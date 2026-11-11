# -*- coding: utf-8 -*-
"""真实检验车辆 Agent 与 RAG 的统一接口，避免 PowerShell 管道损坏中文。"""

from fastapi.testclient import TestClient
from app.main import app

CASES = [
    ("RAG知识问答", "发动机温度过高怎么办？", "knowledge"),
    ("车辆温度查询", "现在车内温度是多少？", "vehicle"),
    ("写操作确认", "请把空调温度设置到23度。", "vehicle"),
    ("不支持的操作", "请帮我打开车门。", "unsupported"),
]


def main():
    print("中文自检：发动机温度过高怎么办？")

    with TestClient(app) as client:
        before = client.get("/vehicle/status").json()["data"]
        pending_token = None

        try:
            for label, message, expected_route in CASES:
                response = client.post(
                    "/assistant/chat",
                    json={"message": message, "mode": "auto", "top_k": 3},
                )
                data = response.json()

                print("\n" + "=" * 40)
                print("场景:", label)
                print("发送内容:", repr(message))
                print("HTTP:", response.status_code)
                print("实际路由:", data.get("route"))
                print("状态:", data.get("status"))
                print("回复:", data.get("reply"))
                print("工具:", data.get("tool_call"))
                print("知识来源:", data.get("sources"))

                assert response.status_code == 200, (
                    f"{label}: HTTP={response.status_code}, data={data}"
                )
                assert data.get("route") == expected_route, (
                    f"{label}: 期望{expected_route}, 实际{data.get('route')}"
                )

                if label == "RAG知识问答":
                    assert data.get("sources"), "RAG没有返回来源"
                elif label == "车辆温度查询":
                    assert data.get("tool_result") is not None
                elif label == "写操作确认":
                    pending_token = data.get("confirmation_token")
                    assert data.get("status") == "confirmation_required"
                    assert pending_token, "未生成确认令牌"
                    assert data.get("tool_call", {}).get("name") == "set_cabin_temp"
                else:
                    assert data.get("tool_call") is None

            after = client.get("/vehicle/status").json()["data"]
            print("\n操作前状态:", before)
            print("操作后状态:", after)
            assert before == after, "未确认的情况下车辆状态发生了变化"
            print("\nPASS: Unified Agent integration")

        finally:
            if pending_token:
                cancelled = client.post(
                    "/agent/confirm",
                    json={"confirmation_token": pending_token, "approved": False},
                )
                print("取消待确认操作 HTTP:", cancelled.status_code)
                assert cancelled.status_code == 400, cancelled.text


if __name__ == "__main__":
    main()
