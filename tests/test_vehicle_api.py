
from fastapi.testclient import TestClient
from app.main import app


client = TestClient(app)


def test_reject_invalid_temperature():
    """验证非法温度被拒绝，且车辆状态不变。"""

    # 记录操作前的温度
    before = client.get(
        "/vehicle/status"
    ).json()["data"]["cabin_temp_c"]

    # 尝试设置非法温度50℃
    response = client.post(
        "/tool/call",
        json={
            "name": "set_cabin_temp",
            "arguments": {
                "temperature_c": 50
            }
        }
    )

    # 记录操作后的温度
    after = client.get(
        "/vehicle/status"
    ).json()["data"]["cabin_temp_c"]

    # 验证系统拒绝非法参数
    assert response.status_code == 422

    # 验证模拟车辆状态没有变化
    assert before == after

def test_reject_unknown_tool():
    """验证未知工具被拒绝，且车辆状态不变。"""

    # 记录操作前的车辆状态
    before = client.get(
        "/vehicle/status"
    ).json()["data"].copy()

    # 尝试调用不在白名单中的工具
    response = client.post(
        "/tool/call",
        json={
            "name": "open_car_door",
            "arguments": {}
        }
    )

    # 记录操作后的车辆状态
    after = client.get(
        "/vehicle/status"
    ).json()["data"].copy()

    # 验证未知工具被拒绝
    assert response.status_code == 403

    # 验证拒绝原因
    assert "不在允许列表中" in response.json()["detail"]

    # 验证车辆状态没有变化
    assert before == after



def test_set_valid_temperature():
    """验证合法温度可以设置，并且状态正确更新。"""

    # 记录测试前的温度
    original_temp = client.get(
        "/vehicle/status"
    ).json()["data"]["cabin_temp_c"]

    try:
        # 设置合法温度22℃
        response = client.post(
            "/tool/call",
            json={
                "name": "set_cabin_temp",
                "arguments": {
                    "temperature_c": 22
                }
            }
        )

        # 检查HTTP状态码
        assert response.status_code == 200

        # 检查工具执行结果
        assert response.json()["result"]["cabin_temp_c"] == 22

        # 再次查询车辆状态
        status = client.get(
            "/vehicle/status"
        ).json()["data"]

        # 验证温度确实发生变化
        assert status["cabin_temp_c"] == 22

    finally:
        # 测试结束后恢复原始温度
        client.post(
            "/tool/call",
            json={
                "name": "set_cabin_temp",
                "arguments": {
                    "temperature_c": original_temp
                }
            }
        )



def test_agent_chat_with_mock(monkeypatch):
    """测试 Agent 接口调用工具，不加载真实 Qwen。"""

    # 模拟 Qwen 的工具规划结果
    def fake_plan_tool(message):
        assert message == "现在车内温度是多少？"

        return {
            "name": "get_cabin_temp",
            "arguments": {}
        }

    # 用假函数临时替换真实 Qwen 规划函数
    monkeypatch.setattr(
        "app.main.plan_tool",
        fake_plan_tool
    )

    # 通过 FastAPI 发送聊天请求
    response = client.post(
        "/agent/chat",
        json={
            "message": "现在车内温度是多少？"
        }
    )

    # 检查接口状态
    assert response.status_code == 200

    data = response.json()

    # 检查 Agent 选择的工具
    assert data["tool_call"]["name"] == "get_cabin_temp"

    # 检查工具执行结果
    expected_temp = client.get(
        "/vehicle/status"
    ).json()["data"]["cabin_temp_c"]

    assert data["tool_result"]["cabin_temp_c"] == expected_temp

    # 检查回复内容
    assert "温度" in data["reply"]



def test_agent_invalid_model_output(monkeypatch):
    """验证模型规划失败时，不执行车辆操作。"""

    # 记录操作前的车辆状态
    before = client.get(
        "/vehicle/status"
    ).json()["data"].copy()

    # 模拟 Qwen 输出无法解析的结果
    def fake_plan_tool(message):
        raise ValueError("模型输出不是合法JSON")

    # 临时替换真实的模型规划函数
    monkeypatch.setattr(
        "app.main.plan_tool",
        fake_plan_tool
    )

    # 向 Agent 发送请求
    response = client.post(
        "/agent/chat",
        json={
            "message": "请把空调调到23度"
        }
    )

    # 验证接口正确处理模型规划异常
    assert response.status_code == 502

    assert response.json()["detail"] == (
        "Qwen 工具规划结果无效"
    )

    # 检查车辆状态是否改变
    after = client.get(
        "/vehicle/status"
    ).json()["data"].copy()

    # 模型规划失败时不能执行车辆操作
    assert before == after



def test_agent_reject_invalid_temperature_from_model(monkeypatch):
    """验证模型生成非法温度时，Agent拒绝执行。"""

    # 记录操作前的车辆状态
    before = client.get(
        "/vehicle/status"
    ).json()["data"].copy()

    # 模拟 Qwen 生成非法温度50℃
    def fake_plan_tool(message):
        return {
            "name": "set_cabin_temp",
            "arguments": {
                "temperature_c": 50
            }
        }

    # 替换真实 Qwen 工具规划函数
    monkeypatch.setattr(
        "app.main.plan_tool",
        fake_plan_tool
    )

    # 发送 Agent 请求
    response = client.post(
        "/agent/chat",
        json={
            "message": "请把空调温度设置到50度。"
        }
    )

    # 验证非法温度被拒绝
    assert response.status_code == 422
    assert "16到30" in response.json()["detail"]

    # 验证车辆状态没有变化
    after = client.get(
        "/vehicle/status"
    ).json()["data"].copy()

    assert before == after


def test_agent_reject_unsupported_operation(monkeypatch):
    """验证不支持的操作不会执行任何车辆工具。"""

    # 记录操作前的车辆状态
    before = client.get(
        "/vehicle/status"
    ).json()["data"].copy()

    # 模拟 Qwen 返回 none
    def fake_plan_tool(message):
        assert message == "请帮我打开车门。"

        return {
            "name": "none",
            "arguments": {}
        }

    # 如果程序错误地执行工具，就让测试失败
    def forbidden_execute_tool(*args, **kwargs):
        raise AssertionError(
            "不支持的操作不应该执行任何车辆工具"
        )

    # 替换真实模型规划
    monkeypatch.setattr(
        "app.main.plan_tool",
        fake_plan_tool
    )

    # 监控是否错误进入工具执行层
    monkeypatch.setattr(
        "app.main.execute_tool",
        forbidden_execute_tool
    )

    # 发送不支持的操作请求
    response = client.post(
        "/agent/chat",
        json={
            "message": "请帮我打开车门。"
        }
    )

    # 验证请求得到正常处理
    assert response.status_code == 200

    data = response.json()

    # 验证没有调用任何工具
    assert data["tool_call"] is None
    assert data["tool_result"] is None

    # 验证回复要求进一步澄清
    assert "无法确定" in data["reply"]

    # 验证车辆状态没有改变
    after = client.get(
        "/vehicle/status"
    ).json()["data"].copy()

    assert before == after



def test_reject_malformed_tool_arguments():
    """验证参数类型错误和多余参数被拒绝。"""

    invalid_cases = [
        # 布尔值不能作为温度
        {
            "name": "set_cabin_temp",
            "arguments": {"temperature_c": True}
        },

        # 字符串不能作为温度
        {
            "name": "set_cabin_temp",
            "arguments": {"temperature_c": "23"}
        },

        # 设置温度不能携带额外参数
        {
            "name": "set_cabin_temp",
            "arguments": {
                "temperature_c": 23,
                "force": True
            }
        },

        # 无参数工具不能携带额外参数
        {
            "name": "get_cabin_temp",
            "arguments": {"temperature_c": 23}
        }
    ]

    before = client.get(
        "/vehicle/status"
    ).json()["data"].copy()

    for payload in invalid_cases:
        response = client.post(
            "/tool/call",
            json=payload
        )

        assert response.status_code == 422, payload

        after = client.get(
            "/vehicle/status"
        ).json()["data"].copy()

        assert before == after, payload



def test_agent_does_not_auto_execute_wrong_write(monkeypatch):
    """验证模型误选合法写工具时，不应直接执行。"""

    executed = []

    # 模拟 Qwen 把开门指令错误理解为设置25℃
    def fake_plan_tool(message):
        return {
            "name": "set_cabin_temp",
            "arguments": {
                "temperature_c": 25
            }
        }

    # 用 Mock 记录工具执行，不修改真实模拟状态
    def fake_execute_tool(name, arguments):
        executed.append((name, arguments))
        return {
            "ok": True,
            "cabin_temp_c": 25
        }

    monkeypatch.setattr(
        "app.main.plan_tool",
        fake_plan_tool
    )

    monkeypatch.setattr(
        "app.main.execute_tool",
        fake_execute_tool
    )

    response = client.post(
        "/agent/chat",
        json={
            "message": "请帮我打开车门。"
        }
    )

    # 没有确认机制时，不允许自动执行写操作
    assert executed == [], (
        f"发现未经确认的工具执行: {executed}"
    )


def test_denied_confirmation_cannot_execute(monkeypatch):
    """用户拒绝后，同一令牌不能再执行操作。"""

    from app.pending_actions import consume_pending_action

    # 模拟模型生成设置温度计划
    def fake_plan_tool(message):
        return {
            "name": "set_cabin_temp",
            "arguments": {"temperature_c": 23}
        }

    executed = []

    # 记录执行请求，不修改模拟车辆状态
    def fake_execute_tool(name, arguments):
        executed.append((name, arguments))
        return {"ok": True, "cabin_temp_c": 23}

    monkeypatch.setattr(
        "app.main.plan_tool", fake_plan_tool
    )

    monkeypatch.setattr(
        "app.main.execute_tool", fake_execute_tool
    )

    response = client.post(
        "/agent/chat",
        json={"message": "把空调调到23度"}
    )

    assert response.status_code == 200
    token = response.json()["confirmation_token"]

    try:
        # 用户明确拒绝
        denied = client.post(
            "/agent/confirm",
            json={
                "confirmation_token": token,
                "approved": False
            }
        )

        assert denied.status_code == 400

        # 再用同一个令牌尝试确认
        repeated = client.post(
            "/agent/confirm",
            json={
                "confirmation_token": token,
                "approved": True
            }
        )

        # 拒绝后，令牌必须失效
        assert repeated.status_code == 410

        # 不能执行任何工具
        assert executed == []

    finally:
        # 清理测试中可能残留的令牌
        consume_pending_action(token)



def test_cross_session_confirmation_rejected(monkeypatch):
    """验证不同会话不能确认其他会话的操作。"""

    from app.pending_actions import (
        discard_pending_action,
        SESSION_COOKIE
    )

    executed = []

    # 模拟 Qwen 工具规划
    def fake_plan_tool(message):
        return {
            "name": "set_cabin_temp",
            "arguments": {
                "temperature_c": 23
            }
        }

    # 使用 Mock 记录执行，不修改车辆状态
    def fake_execute_tool(name, arguments):
        executed.append((name, arguments))
        return {
            "ok": True,
            "cabin_temp_c": 23
        }

    monkeypatch.setattr(
        "app.main.plan_tool",
        fake_plan_tool
    )

    monkeypatch.setattr(
        "app.main.execute_tool",
        fake_execute_tool
    )

    # 创建两个独立客户端，分别模拟会话 A 和 B
    with TestClient(app) as client_a, TestClient(app) as client_b:

        # 会话 A 创建待确认操作
        response_a = client_a.post(
            "/agent/chat",
            json={
                "message": "请把空调调到23度"
            }
        )

        assert response_a.status_code == 200
        token_a = response_a.json()["confirmation_token"]

        # 会话 B 创建自己的待确认操作
        response_b = client_b.post(
            "/agent/chat",
            json={
                "message": "请把空调调到23度"
            }
        )

        assert response_b.status_code == 200
        token_b = response_b.json()["confirmation_token"]

        # 确认两个客户端拥有不同的会话 Cookie
        session_a = client_a.cookies.get(SESSION_COOKIE)
        session_b = client_b.cookies.get(SESSION_COOKIE)

        assert session_a is not None
        assert session_b is not None
        assert session_a != session_b

        try:
            # 会话 B 尝试确认 A 的操作
            denied = client_b.post(
                "/agent/confirm",
                json={
                    "confirmation_token": token_a,
                    "approved": True
                }
            )

            # 跨会话操作必须被拒绝
            assert denied.status_code == 403
            assert executed == []

            # 会话 A 确认自己的操作
            approved = client_a.post(
                "/agent/confirm",
                json={
                    "confirmation_token": token_a,
                    "approved": True
                }
            )

            assert approved.status_code == 200
            assert approved.json()["status"] == "executed"

            # 确认只执行了一次正确的工具
            assert executed == [
                (
                    "set_cabin_temp",
                    {"temperature_c": 23}
                )
            ]

            # 会话 A 重复确认必须失败
            repeated = client_a.post(
                "/agent/confirm",
                json={
                    "confirmation_token": token_a,
                    "approved": True
                }
            )

            assert repeated.status_code == 410

        finally:
            # 清理测试产生的令牌
            discard_pending_action(token_a)
            discard_pending_action(token_b)