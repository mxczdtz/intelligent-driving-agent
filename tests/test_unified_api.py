"""统一 Agent 入口测试；使用 Mock，不加载 Qwen 或 BGE。"""

import pytest
from fastapi.testclient import TestClient

import app.main as main
from app.intent_router import classify_message


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("发动机温度持续升高怎么办？", "knowledge"),
        ("空调不制冷是什么原因？", "knowledge"),
        ("机油压力警告灯亮怎么办？", "knowledge"),
        ("现在车内温度是多少？", "vehicle"),
        ("当前车速是多少？", "vehicle"),
        ("请把空调温度设置到23度。", "vehicle"),
        ("请帮我打开车门。", "unsupported"),
        ("给我播放一首音乐。", "unsupported"),
        ("帮我导航去机场。", "unsupported"),
    ],
)
def test_auto_router_classification(message, expected):
    assert classify_message(message) == expected


def test_unified_knowledge_uses_rag_and_sources(monkeypatch):
    """故障咨询必须走RAG，不调用车辆规划。"""
    class FakeRag:
        def answer(self, question, top_k):
            assert question == "发动机温度过高怎么办？"
            assert top_k == 2
            return {
                "answer": "先安全停车，等待冷却后再检查。[KB001]",
                "sources": [
                    {
                        "id": "KB001",
                        "title": "发动机冷却液温度过高",
                        "source": "项目自编教学示例"
                    }
                ],
                "answer_mode": "extractive_high_risk"
            }

    monkeypatch.setattr(main, "get_rag_engine", lambda: FakeRag())

    def forbidden_planner(_):
        raise AssertionError("故障问答不应调用工具规划")

    monkeypatch.setattr(main, "plan_tool", forbidden_planner)

    with TestClient(main.app) as client:
        response = client.post(
            "/assistant/chat",
            json={
                "message": "发动机温度过高怎么办？",
                "mode": "auto",
                "top_k": 2
            }
        )

    assert response.status_code == 200
    data = response.json()
    assert data["route"] == "knowledge"
    assert data["status"] == "answered"
    assert data["sources"][0]["id"] == "KB001"
    assert data["answer_mode"] == "extractive_high_risk"
    assert data["tool_call"] is None


def test_unified_vehicle_read_only(monkeypatch):
    """只读车况请求执行原车辆工具，不调用RAG。"""
    def fake_plan_tool(message):
        assert message == "现在车内温度是多少？"
        return {"name": "get_cabin_temp", "arguments": {}}

    def forbidden_rag():
        raise AssertionError("车辆查询不应该调用RAG")

    monkeypatch.setattr(main, "plan_tool", fake_plan_tool)
    monkeypatch.setattr(main, "get_rag_engine", forbidden_rag)

    with TestClient(main.app) as client:
        response = client.post(
            "/assistant/chat",
            json={"message": "现在车内温度是多少？"}
        )
    assert response.status_code == 200
    data = response.json()
    assert data["route"] == "vehicle"
    assert data["tool_call"]["name"] == "get_cabin_temp"
    assert data["tool_result"]["cabin_temp_c"] == main.vehicle_memory["cabin_temp_c"]


def test_unified_write_requires_confirmation_and_preserves_session(monkeypatch):
    """统一入口保留原来的确认令牌和HttpOnly会话Cookie。"""
    from app.pending_actions import discard_pending_action, SESSION_COOKIE

    monkeypatch.setattr(
        main,
        "plan_tool",
        lambda message: {
            "name": "set_cabin_temp",
            "arguments": {"temperature_c": 23}
        }
    )

    before = dict(main.vehicle_memory)
    with TestClient(main.app) as client:
        result = client.post(
            "/assistant/chat",
            json={"message": "请把空调温度设置到23度。"}
        )

        assert result.status_code == 200, result.text
        data = result.json()
        assert data["route"] == "vehicle"
        assert data["status"] == "confirmation_required"
        assert "confirmation_token" in data
        assert client.cookies.get(SESSION_COOKIE) is not None
        assert main.vehicle_memory == before

        try:
            # 使用同一客户端再次确认，证明会话cookie被正确传递。
            confirmed = client.post(
                "/agent/confirm",
                json={
                    "confirmation_token": data["confirmation_token"],
                    "approved": False
                }
            )
            assert confirmed.status_code == 400
            assert main.vehicle_memory == before
        finally:
            discard_pending_action(data["confirmation_token"])


def test_unified_unsupported_cannot_create_write(monkeypatch):
    """不支持的操作不会进模型，防止其错误选择合法写工具。"""
    def forbidden_planner(message):
        raise AssertionError("开车门不应传递给工具规划")

    def forbidden_rag():
        raise AssertionError("开车门不应触发RAG")

    monkeypatch.setattr(main, "plan_tool", forbidden_planner)
    monkeypatch.setattr(main, "get_rag_engine", forbidden_rag)

    with TestClient(main.app) as client:
        response = client.post(
            "/assistant/chat",
            json={"message": "请帮我打开车门。"}
        )
    assert response.status_code == 200
    data = response.json()
    assert data["route"] == "unsupported"
    assert data["status"] == "no_action"
    assert data["tool_call"] is None


def test_unified_mode_override(monkeypatch):
    class FakeRag:
        def answer(self, question, top_k):
            return {"answer": "资料不足。", "sources": []}

    monkeypatch.setattr(main, "get_rag_engine", lambda: FakeRag())

    with TestClient(main.app) as client:
        response = client.post(
            "/assistant/chat",
            json={
                "message": "这句话需要明确选择知识问答",
                "mode": "knowledge"
            }
        )
    assert response.status_code == 200
    assert response.json()["route"] == "knowledge"


def test_unified_rejects_blank_question():
    with TestClient(main.app) as client:
        response = client.post(
            "/assistant/chat",
            json={"message": "   "}
        )
    assert response.status_code == 422


def test_unified_rejects_vehicle_message_over_200():
    with TestClient(main.app) as client:
        response = client.post(
            "/assistant/chat",
            json={
                "message": "请查询车辆状态" * 30,
                "mode": "vehicle"
            }
        )
    assert response.status_code == 422
