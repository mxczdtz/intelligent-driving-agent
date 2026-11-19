
import logging
from functools import lru_cache
import math

from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import BaseModel, Field

from app.agent_planner import plan_tool
from app.intent_router import classify_message
from app.pending_actions import (
    create_pending_action,
    consume_pending_action,
    get_or_create_session,
    SessionMismatchError,
    EXPIRE_SECONDS,
    SESSION_EXPIRE_SECONDS,
    SESSION_COOKIE
)


# =====================================
# 1. FastAPI 应用
# =====================================

app = FastAPI(
    title="Vehicle Qwen Agent API",
    description=(
        "基于本地 Qwen 大模型的汽车智能座舱 Agent 教学项目。"
        "支持模拟车辆状态查询、空调温度设置与工具调用。"
        "不连接真实车辆。"
    ),
    version="0.3.0"
)


# =====================================
# 2. 模拟车辆状态
# =====================================

vehicle_memory = {
    "speed_kmh": 0,
    "cabin_temp_c": 24,
    "fan_level": 2
}


# =====================================
# 3. 工具白名单
# =====================================

ALLOWED_TOOLS = {
    "get_vehicle_status",
    "get_cabin_temp",
    "set_cabin_temp"
}


# =====================================
# 4. 请求数据模型
# =====================================

class ToolCallRequest(BaseModel):
    name: str = Field(
        description="工具名称"
    )

    arguments: dict = Field(
        default_factory=dict,
        description="工具参数"
    )


class ChatRequest(BaseModel):
    message: str = Field(
        min_length=1,
        max_length=200,
        description="用户自然语言指令"
    )


class TemperatureRequest(BaseModel):
    temperature_c: float = Field(
        ge=16,
        le=30,
        description="模拟空调温度，范围16到30℃"
    )


# =====================================
# 5. 模拟车辆工具函数
# =====================================

def get_vehicle_status():
    """查询完整模拟车辆状态。"""

    return {
        "speed_kmh": vehicle_memory["speed_kmh"],
        "cabin_temp_c": vehicle_memory["cabin_temp_c"],
        "fan_level": vehicle_memory["fan_level"]
    }


def get_cabin_temp():
    """查询模拟车内温度。"""

    return {
        "cabin_temp_c": vehicle_memory["cabin_temp_c"]
    }


def set_cabin_temp(temperature_c: float):
    """设置模拟车内温度。"""

    vehicle_memory["cabin_temp_c"] = temperature_c

    return {
        "ok": True,
        "cabin_temp_c": temperature_c
    }


# =====================================
# 6. 工具请求安全校验
# =====================================

def validate_tool_request(name: str, arguments: dict):
    """
    只校验工具请求，不执行任何操作。

    检查内容：
    1. 工具白名单
    2. 参数对象类型
    3. 工具参数结构
    4. 参数类型
    5. 温度业务范围
    """

    # 检查工具白名单
    if name not in ALLOWED_TOOLS:
        raise HTTPException(
            status_code=403,
            detail=f"工具 {name} 不在允许列表中"
        )

    # 检查参数对象
    if not isinstance(arguments, dict):
        raise HTTPException(
            status_code=422,
            detail="arguments 必须是对象"
        )

    # ---------------------------------
    # 查询类工具：不允许任何参数
    # ---------------------------------

    if name in (
        "get_vehicle_status",
        "get_cabin_temp"
    ):

        if arguments:
            raise HTTPException(
                status_code=422,
                detail=f"{name} 不接受参数"
            )

        return

    # ---------------------------------
    # 温度设置工具：严格检查参数
    # ---------------------------------

    if name == "set_cabin_temp":

        # 只能有 temperature_c 一个参数
        if set(arguments.keys()) != {"temperature_c"}:
            raise HTTPException(
                status_code=422,
                detail=(
                    "set_cabin_temp 只接受 "
                    "temperature_c 参数"
                )
            )

        temperature_c = arguments["temperature_c"]

        # 拒绝字符串、布尔值、空值等
        if type(temperature_c) not in (int, float):
            raise HTTPException(
                status_code=422,
                detail="temperature_c 必须是有限数字"
            )

        # 拒绝 NaN 和无穷大
        if (
            isinstance(temperature_c, float)
            and not math.isfinite(temperature_c)
        ):
            raise HTTPException(
                status_code=422,
                detail="temperature_c 必须是有限数字"
            )

        # 检查业务范围
        if not 16 <= temperature_c <= 30:
            raise HTTPException(
                status_code=422,
                detail="temperature_c 必须在16到30之间"
            )

        return

    raise HTTPException(
        status_code=500,
        detail="工具校验逻辑未实现"
    )


# =====================================
# 7. 统一工具执行入口
# =====================================

def execute_tool(name: str, arguments: dict):
    """
    校验通过后执行模拟车辆工具。

    注意：
    这个函数具有实际执行能力。
    Agent 写操作必须经过额外确认流程，
    不能直接调用它修改状态。
    """

    # 先进行安全校验
    validate_tool_request(name, arguments)

    # 查询车辆完整状态
    if name == "get_vehicle_status":
        return get_vehicle_status()

    # 查询车内温度
    if name == "get_cabin_temp":
        return get_cabin_temp()

    # 设置模拟温度
    if name == "set_cabin_temp":
        return set_cabin_temp(
            arguments["temperature_c"]
        )

    raise HTTPException(
        status_code=500,
        detail="工具执行逻辑未实现"
    )


# =====================================
# 8. 健康检查接口
# =====================================

@app.get("/health")
def health_check():

    return {
        "status": "ok",
        "service": "vehicle-agent-demo"
    }


# =====================================
# 9. 查询车辆状态接口
# =====================================

@app.get("/vehicle/status")
def vehicle_status():

    return {
        "ok": True,
        "data": get_vehicle_status()
    }


# =====================================
# 10. 直接设置模拟温度接口
# =====================================

@app.post("/vehicle/temperature")
def update_temperature(request: TemperatureRequest):

    result = execute_tool(
        name="set_cabin_temp",
        arguments={
            "temperature_c": request.temperature_c
        }
    )

    return {
        "ok": True,
        "message": "模拟车内温度设置成功",
        "data": result
    }


# =====================================
# 11. 独立 Tool Calling 接口
# =====================================

@app.post("/tool/call")
def tool_call(request: ToolCallRequest):

    result = execute_tool(
        name=request.name,
        arguments=request.arguments
    )

    return {
        "ok": True,
        "tool_name": request.name,
        "result": result
    }


# =====================================
# 12. Qwen Agent 聊天接口
# =====================================

@app.post("/agent/chat")
def agent_chat(
    request: ChatRequest,
    http_request: Request,
    response: Response
):

    message = request.message.strip()

    if not message:
        raise HTTPException(
            status_code=422,
            detail="message 不能为空"
        )

    # ---------------------------------
    # 第一步：调用本地 Qwen 大模型
    # ---------------------------------

    try:
        decision = plan_tool(message)

    except ValueError as exc:
        raise HTTPException(
            status_code=502,
            detail="Qwen 工具规划结果无效"
        ) from exc

    # ---------------------------------
    # 第二步：获取工具规划结果
    # ---------------------------------

    tool_name = decision["name"]
    arguments = decision["arguments"]

    # ---------------------------------
    # 第三步：处理不支持的操作
    # ---------------------------------

    if tool_name == "none":

        return {
            "ok": True,
            "tool_call": None,
            "tool_result": None,
            "reply": (
                "无法确定要执行的模拟车辆操作，"
                "请提供更明确的指令。"
            )
        }

    # ---------------------------------
    # 第四步：创建待确认操作
    # ---------------------------------

    if tool_name == "set_cabin_temp":

        # 1. 先进行参数安全校验
        validate_tool_request(
            name=tool_name,
            arguments=arguments
        )

        # 2. 保存待确认操作，生成随机令牌

        # 获取服务端签发的会话凭证
        session_token = get_or_create_session(
            http_request.cookies.get(SESSION_COOKIE)
        )

        # 将待确认操作与会话绑定
        confirmation_token = create_pending_action(
            name=tool_name,
            arguments=arguments,
            session_token=session_token
        )

        # 通过 HttpOnly Cookie 保存会话凭证
        response.set_cookie(
            key=SESSION_COOKIE,
            value=session_token,
            max_age=SESSION_EXPIRE_SECONDS,
            httponly=True,
            samesite="strict",
            secure=False,
            path="/agent"
        )

        response.headers["Cache-Control"] = "no-store"

        # 3. 返回确认信息，不执行车辆修改
        return {
            "ok": True,
            "status": "confirmation_required",
            "confirmation_token": confirmation_token,
            "expires_in_seconds": EXPIRE_SECONDS,
            "tool_call": decision,
            "tool_result": None,
            "reply": (
                f"模型建议将模拟车内温度设置为"
                f"{arguments['temperature_c']}℃，"
                f"但尚未执行。"
                f"请确认这是否符合你的真实意图。"
            )
        }

    # ---------------------------------
    # 第五步：只读工具正常执行
    # ---------------------------------

    result = execute_tool(
        name=tool_name,
        arguments=arguments
    )

    # ---------------------------------
    # 第六步：根据工具结果构造回复
    # ---------------------------------

    if tool_name == "get_cabin_temp":

        reply = (
            f"当前模拟车内温度为"
            f" {result['cabin_temp_c']}℃。"
        )

    elif tool_name == "get_vehicle_status":

        reply = (
            f"当前模拟车速为 "
            f"{result['speed_kmh']} km/h，"
            f"车内温度为 "
            f"{result['cabin_temp_c']}℃，"
            f"风扇档位为 "
            f"{result['fan_level']}。"
        )

    else:
        raise HTTPException(
            status_code=403,
            detail="不支持的工具"
        )

    # ---------------------------------
    # 第七步：返回 Agent 执行结果
    # ---------------------------------

    return {
        "ok": True,
        "tool_call": decision,
        "tool_result": result,
        "reply": reply
    }



# =====================================
# 13. Agent 用户确认执行接口
# =====================================

class ConfirmRequest(BaseModel):
    confirmation_token: str = Field(
        min_length=1,
        max_length=256,
        description="服务端生成的一次性确认令牌"
    )

    approved: bool = Field(
        description="用户是否明确同意执行"
    )


@app.post("/agent/confirm")
def agent_confirm(request: ConfirmRequest, http_request: Request):
    """仅允许所属会话一次性确认或拒绝待执行操作。"""

    # 从服务端签发的 Cookie 获取当前会话凭证
    session_token = http_request.cookies.get(SESSION_COOKIE)

    # 跨会话访问被拒绝，且不消耗原令牌
    try:
        action = consume_pending_action(
            request.confirmation_token,
            session_token=session_token
        )
    except SessionMismatchError as exc:
        raise HTTPException(
            status_code=403,
            detail="该确认令牌不属于当前会话"
        ) from exc

    # 令牌不存在、已使用或过期
    if action is None:
        raise HTTPException(
            status_code=410,
            detail="确认令牌无效、已过期或已使用"
        )

    # 第二步：用户明确拒绝，令牌已经失效
    if request.approved is not True:
        raise HTTPException(
            status_code=400,
            detail="用户已拒绝执行操作，确认令牌已失效"
        )

    # 第三步：重新进行工具安全校验并执行
    result = execute_tool(
        name=action["name"],
        arguments=action["arguments"]
    )

    # 第四步：根据真正的执行结果回复
    if action["name"] == "set_cabin_temp":
        reply = (
            f"用户已确认，模拟车内温度"
            f"已设置为{result['cabin_temp_c']}℃。"
        )
    else:
        reply = "已完成确认的模拟操作。"

    return {
        "ok": True,
        "status": "executed",
        "tool_name": action["name"],
        "tool_result": result,
        "reply": reply
    }


# =====================================
# 14. 汽车知识库 RAG 问答接口
# =====================================

logger = logging.getLogger(__name__)


class RagAskRequest(BaseModel):
    """RAG 问答请求；与车辆工具执行接口完全分离。"""

    question: str = Field(
        min_length=1,
        max_length=500,
        description="汽车故障或技术知识问题",
        examples=["发动机温度过高应该怎么办？"]
    )

    top_k: int = Field(
        default=3,
        ge=1,
        le=5,
        description="融合检索后返回的知识条目数量"
    )


@lru_cache(maxsize=1)
def get_rag_engine():
    """
    延迟加载 RAG 引擎，同一进程复用 BM25 文档和 Embedding 模型。
    不在 import app.main 时下载模型或创建向量索引。
    """
    from app.rag_engine import VehicleRAG
    return VehicleRAG()


@app.post("/rag/ask", tags=["RAG"])
def rag_ask(request: RagAskRequest):
    """
    独立的汽车知识问答入口：混合检索 -> Qwen/高风险原文回答 -> 返回来源。

    不触发任何模拟车辆写操作。
    """

    question = request.question.strip()
    if not question:
        raise HTTPException(
            status_code=422,
            detail="question 不能为空"
        )

    try:
        result = get_rag_engine().answer(
            question=question,
            top_k=request.top_k
        )
    except Exception as exc:
        # 记录详细错误供开发调试，但不把本地绝对路径/环境信息泄露给客户端。
        logger.exception("RAG 问答失败")
        raise HTTPException(
            status_code=503,
            detail="RAG 服务暂不可用，请检查知识库、索引和本地模型配置"
        ) from exc

    if not isinstance(result, dict):
        logger.error("RAG 引擎返回了非字典结果")
        raise HTTPException(
            status_code=503,
            detail="RAG 服务返回结果异常"
        )

    sources = result.get("sources", [])
    answer_mode = result.get(
        "answer_mode",
        "llm_generated" if sources else "no_results"
    )

    return {
        "ok": True,
        "question": question,
        "answer": result.get("answer", ""),
        "sources": sources,
        "answer_mode": answer_mode
    }


# =====================================
# 15. 统一入口：车辆操作 + 汽车知识问答
# =====================================

from typing import Literal


class UnifiedChatRequest(BaseModel):
    message: str = Field(
        min_length=1,
        max_length=500,
        description="用户自然语言请求"
    )
    mode: Literal["auto", "vehicle", "knowledge"] = Field(
        default="auto",
        description="auto：规则分流；vehicle：车辆工具；knowledge：知识库问答"
    )
    top_k: int = Field(
        default=3,
        ge=1,
        le=5,
        description="知识库检索数量；车辆工具模式下忽略"
    )


@app.post("/assistant/chat", tags=["Assistant"])
def unified_chat(
    request: UnifiedChatRequest,
    http_request: Request,
    response: Response
):
    """
    单入口：
    - 对明确知识问题，复用 /rag/ask 的安全回答逻辑。
    - 对明确支持的车辆请求，复用 /agent/chat 的计划与确认逻辑。
    - 未识别请求默认拒绝，避免让模型猜测不存在的写工具。

    对不明确的问题，客户端可显式指定 mode。
    """
    message = request.message.strip()
    if not message:
        raise HTTPException(status_code=422, detail="message 不能为空")

    selected = (
        classify_message(message)
        if request.mode == "auto"
        else request.mode
    )

    if selected == "unsupported":
        return {
            "ok": True,
            "route": "unsupported",
            "status": "no_action",
            "reply": (
                "当前只支持查询模拟车速/车辆状态/车内温度、"
                "申请设置模拟空调温度，以及汽车故障知识问答。"
                "不会执行其他车辆操作。"
            ),
            "tool_call": None,
            "tool_result": None,
            "sources": []
        }

    if selected == "knowledge":
        rag_result = rag_ask(
            RagAskRequest(question=message, top_k=request.top_k)
        )
        return {
            "ok": True,
            "route": "knowledge",
            "status": "answered",
            "reply": rag_result["answer"],
            "tool_call": None,
            "tool_result": None,
            "sources": rag_result["sources"],
            "answer_mode": rag_result["answer_mode"]
        }

    # 原有 ChatRequest 限制200字符，避免自动入口绕过原接口的长度约束。
    if len(message) > 200:
        raise HTTPException(
            status_code=422,
            detail="车辆工具请求最多200个字符"
        )

    vehicle_result = agent_chat(
        ChatRequest(message=message),
        http_request,
        response
    )

    # 复用已有的确认令牌、会话 Cookie、安全校验和只读工具执行。
    return {
        "route": "vehicle",
        "sources": [],
        "status": vehicle_result.get(
            "status",
            "completed" if vehicle_result.get("tool_result") is not None
            else "no_action"
        ),
        **vehicle_result
    }

# === SCENE_RISK_ROUTER_PHASE1 ===
from app.scene_api import router as scene_risk_router
app.include_router(scene_risk_router)

# === MULTIMODAL_DRIVING_AGENT_PHASE5_1 ===
# Keep the existing /scene/analyze single-image endpoint unchanged.
from app.multimodal_api import router as multimodal_agent_router
app.include_router(multimodal_agent_router)

# === CARLA_LIVE_RGB_PHASE5_2 ===
from app.live_carla_api import router as carla_live_rgb_router
app.include_router(carla_live_rgb_router)


# === UNIFIED_VEHICLE_AI_DASHBOARD_V4 ===
# This block is additive: all existing routes and model/vehicle functions above
# remain unchanged. V3 is embedded using its existing same-origin route.
from fastapi.responses import HTMLResponse


@app.post("/dashboard/validate-tool", tags=["Dashboard V4.1"])
def dashboard_validate_tool(request: ToolCallRequest):
    """Validate a tool contract without calling execute_tool or changing state."""
    validate_tool_request(request.name, request.arguments)
    return {
        "ok": True,
        "validation": "passed",
        "executed": False,
        "tool_name": request.name,
        "arguments": request.arguments,
    }


@app.get("/dashboard", response_class=HTMLResponse, tags=["Dashboard V4.1"])
def unified_dashboard_v4():
    """Unified same-origin research cockpit; keep all model execution in APIs."""
    return HTMLResponse(
        content=DASHBOARD_V4_HTML,
        headers={"Cache-Control": "no-store"},
    )


DASHBOARD_V4_HTML = r'''<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Vehicle AI · Dashboard V4.2</title>
<style>
:root{color-scheme:dark;--bg:#0c1420;--panel:#142232;--panel2:#101d2b;--line:#293c50;--ink:#e9f1f7;--muted:#9eb3c6;--accent:#68ddbd;--blue:#8bbbf7;--warn:#f2c47b;--bad:#ff9e98;--radius:14px}
*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.6 "Microsoft YaHei","PingFang SC",system-ui,sans-serif}
a{color:var(--blue);text-decoration:none}a:hover{text-decoration:underline}button,input,select,textarea{font:inherit}button,a,select,textarea,input{outline-offset:4px}button{cursor:pointer}button:disabled{opacity:.5;cursor:not-allowed}button:focus-visible,a:focus-visible,select:focus-visible,textarea:focus-visible{outline:2px solid var(--accent)}
header{background:#111f2e;border-bottom:1px solid var(--line);padding:21px 28px 17px}header .top{display:flex;align-items:center;justify-content:space-between;gap:18px;flex-wrap:wrap}
.brand{display:flex;align-items:center;gap:13px}.mark{width:42px;height:42px;border:1px solid #42776d;border-radius:12px;background:#1e403c;display:grid;place-items:center;color:var(--accent);font-size:21px;font-weight:700}
.eyebrow{font-size:10px;letter-spacing:.16em;text-transform:uppercase;color:var(--accent);font-weight:700;margin:0}h1{font-size:clamp(19px,2.4vw,28px);line-height:1.25;margin:3px 0 0;font-weight:700}h2{font-size:17px;line-height:1.4;margin:0}h3{font-size:13px;margin:0}
.sub{color:var(--muted);font-size:12px;margin:5px 0 0}.head-links{display:flex;gap:9px;flex-wrap:wrap;align-items:center}.pill,.btn{border:1px solid var(--line);padding:8px 12px;border-radius:8px;background:#192b3c;color:var(--ink);text-decoration:none;display:inline-flex;align-items:center;justify-content:center;gap:6px;font-size:12px;line-height:1.45}
.btn:hover,.pill:hover{border-color:var(--blue);text-decoration:none}.primary{background:#23604f;border-color:#357a66;color:#e6fff6;font-weight:650}.danger{border-color:#705052;color:#ffd0cb;background:#392b35}.plain{background:transparent}.small{padding:6px 9px;font-size:11px}
nav{display:flex;gap:6px;flex-wrap:wrap;margin-top:19px}nav a{padding:7px 13px;border:1px solid #2a3c4e;border-radius:7px;color:#bcd0df;font-size:12px}nav a:hover{border-color:var(--accent);color:var(--accent);text-decoration:none}
main{max-width:1900px;margin:0 auto;padding:20px 26px 34px}.notice{border-left:3px solid var(--warn);background:#272a2e;padding:10px 14px;border-radius:6px;color:#e5d6bb;font-size:12px;margin-bottom:17px}
.workspace{display:grid;grid-template-columns:minmax(0,1.62fr) minmax(345px,1fr);gap:17px;align-items:start}
.panel{border:1px solid var(--line);background:var(--panel);border-radius:var(--radius);min-width:0;overflow:hidden}
.panel-head{padding:15px 17px;display:flex;align-items:flex-start;justify-content:space-between;gap:12px;flex-wrap:wrap;border-bottom:1px solid var(--line)}.panel-head .eyebrow{margin-bottom:3px}
.panel-body{padding:15px 17px}.stack{display:grid;gap:17px;min-width:0}.section-gap{margin-top:17px}
iframe{display:block;width:100%;height:900px;border:0;background:#f5f7f6}.frame-foot{padding:10px 15px;border-top:1px solid var(--line);font-size:11px;color:var(--muted)}
.status-line{display:flex;justify-content:space-between;align-items:center;gap:10px;flex-wrap:wrap}.status-label{font-size:11px;color:var(--muted)}.metrics{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:8px;margin-top:12px}
.metric{padding:10px;border:1px solid var(--line);border-radius:9px;background:var(--panel2);min-width:0}.metric strong{display:block;font-size:22px;line-height:1.3;font-variant-numeric:tabular-nums;color:var(--ink);overflow-wrap:anywhere}.metric span{font-size:10px;color:var(--muted)}
.divider{height:1px;background:var(--line);margin:14px 0}.examples{display:flex;gap:6px;flex-wrap:wrap;margin:10px 0 12px}
.form-row{display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin-bottom:9px}.form-row label{font-size:12px;color:var(--muted)}.form-row select{flex:1;min-width:120px}
select,textarea,input{background:#0c1825;color:var(--ink);border:1px solid #344b60;border-radius:8px;padding:9px}textarea{width:100%;resize:vertical;display:block;min-height:75px;line-height:1.55}textarea::placeholder{color:#7e94a9}
.chat-history{display:flex;flex-direction:column;gap:10px;min-height:115px;max-height:340px;overflow:auto;padding:3px 3px 9px;scrollbar-color:#3a5065 transparent}.message{max-width:95%;padding:10px 12px;background:#1b3043;border:1px solid #2d4459;border-radius:10px;overflow-wrap:anywhere;white-space:pre-wrap;font-size:13px}.message.user{align-self:flex-end;background:#20453e;border-color:#336357}.message.error{border-color:#795254;color:#ffd1cb}.message small{display:block;color:var(--muted);font-size:10px;white-space:normal;margin-bottom:3px}
.actions{display:flex;align-items:center;justify-content:space-between;gap:10px;flex-wrap:wrap;margin-top:9px}.muted{color:var(--muted);font-size:11px}.good{color:var(--accent)}.warn{color:var(--warn)}.bad{color:var(--bad)}
.pending{padding:13px;border:1px solid #796341;background:#322e29;border-radius:10px;margin:12px 0;color:#f5e0bd}.pending p{margin:5px 0;font-size:12px;white-space:pre-wrap;overflow-wrap:anywhere}.pending[hidden]{display:none}.pending .actions{justify-content:flex-start}
.trace{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin:12px 0}.trace div{border:1px solid var(--line);background:var(--panel2);border-radius:8px;padding:8px 10px;min-width:0}.trace span{display:block;color:var(--muted);font-size:10px}.trace strong{display:block;font-size:12px;overflow-wrap:anywhere;margin-top:3px;font-weight:600}
details{border:1px solid var(--line);border-radius:8px;background:var(--panel2);margin-top:9px}summary{cursor:pointer;font-size:12px;color:#c4d8e7;padding:9px 11px}pre{font:11.5px/1.6 Consolas,"SFMono-Regular",monospace;color:#c4e0ef;background:#0b1825;margin:0;padding:11px;overflow:auto;white-space:pre-wrap;overflow-wrap:anywhere;max-height:360px}details pre{border-radius:0 0 7px 7px;border-top:1px solid var(--line)}
.json-editor{font:12px/1.6 Consolas,monospace;min-height:125px}.result-label{margin-top:10px;white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}.result-label:empty{display:none}
.log-layout{display:grid;grid-template-columns:1fr;gap:13px}.log-list{display:flex;flex-direction:column;gap:7px;max-height:300px;overflow:auto}.log-row{width:100%;display:grid;grid-template-columns:minmax(0,1fr) auto;gap:3px 12px;text-align:left;background:#101e2c;border:1px solid var(--line);padding:9px 11px;border-radius:8px;color:var(--ink)}.log-row:hover,.log-row.selected{border-color:#5ba994}.log-row .path{overflow-wrap:anywhere;font:11px/1.5 Consolas,monospace;color:#c9d9e9}.log-row .meta{font-size:10px;color:var(--muted)}.log-row .code{font:11px/1.5 Consolas,monospace;color:var(--accent)}.log-row .code.fail{color:var(--bad)}
.log-detail{border:1px solid var(--line);border-radius:9px;background:var(--panel2);min-width:0;overflow:hidden}.log-detail h3{padding:10px 12px}.log-detail pre{min-height:220px;max-height:365px}.empty{padding:12px;color:var(--muted);font-size:12px}
footer{border-top:1px solid var(--line);color:var(--muted);font-size:11px;padding:14px 26px;display:flex;justify-content:space-between;gap:10px;flex-wrap:wrap}
@media(max-width:1100px){main{padding:14px}.workspace{grid-template-columns:minmax(0,1fr) minmax(320px,.78fr);gap:12px}.panel-body,.panel-head{padding:12px}.metrics{gap:5px}.metric{padding:7px}.metric strong{font-size:19px}}
@media(max-width:900px){.workspace{grid-template-columns:1fr}iframe{height:760px}.log-layout{grid-template-columns:1fr}header{padding:17px}main{padding:12px}.stack{gap:12px}}
@media(prefers-reduced-motion:reduce){html{scroll-behavior:auto}}

/* V4.2: one shared desktop row; both outer cards have the same dimensions. */
*{min-width:0}
main{max-width:none;width:100%;padding:16px 24px 18px}
.workspace{grid-template-columns:repeat(2,minmax(0,1fr));align-items:stretch;gap:16px}
#drive{display:flex;flex-direction:column;min-height:0}
#drive>.panel-head,#drive>.frame-foot{flex:none}
#drive>.panel-head{min-height:74px}
#drive iframe{display:block;width:100%;flex:1 1 0;height:0;min-height:0}
.frame-foot{padding:8px 14px}
.console{display:flex;flex-direction:column;gap:0;min-height:0;overflow:hidden}
.console-tabs{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:7px;padding:10px;border-bottom:1px solid var(--line);background:#101d2b;flex:none}
.console-tabs button{appearance:none;min-width:0;border:1px solid var(--line);border-radius:8px;padding:10px 6px;background:#172839;color:var(--muted);font-weight:600;font-size:13px;white-space:nowrap}
.console-tabs button span{font:10px Consolas,monospace;margin-right:4px;opacity:.72}
.console-tabs button:hover{border-color:#507a86;color:var(--ink)}
.console-tabs button[aria-selected="true"]{border-color:#3d8676;color:var(--accent);background:#20483e}
.console>.console-pane{flex:1 1 0;min-height:0;display:flex;flex-direction:column;overflow:hidden;border:0;border-radius:0;margin:0;background:transparent}
.console>.console-pane[hidden]{display:none!important}
.console-pane>.panel-head{flex:none}
.console-pane>.panel-body{flex:1 1 auto;min-height:0;overflow-y:auto;scrollbar-color:#395165 transparent}
.console-pane .chat-history{min-height:100px;max-height:230px}
.console-pane .log-layout{display:grid;grid-template-columns:minmax(0,1fr);align-content:start}
.console-pane .log-detail pre{min-height:170px;max-height:360px}
.notice{margin-bottom:14px}
header{padding:16px 24px 12px}
header nav{margin-top:12px}
footer{padding:9px 24px}
@media(min-width:901px) and (min-height:640px){
  html,body{height:100%;overflow:hidden}
  body{display:flex;flex-direction:column;height:100vh;height:100dvh}
  header,footer{flex:none}
  main{flex:1 1 auto;min-height:0;display:flex;flex-direction:column}
  main>.notice{flex:none}
  main>.workspace{flex:1 1 0;min-height:0}
  .workspace>#drive,.workspace>#console{height:100%;min-height:0}
}
@media(min-width:901px) and (max-height:639px){
  .workspace{min-height:700px}
  #drive,#console{height:700px}
}
@media(max-width:900px){
  body{height:auto;overflow:auto}
  main{padding:12px}
  header{padding:15px}
  .workspace{grid-template-columns:minmax(0,1fr);gap:12px}
  #drive iframe{flex:none;height:680px;height:clamp(520px,78dvh,820px)}
  .console>.console-pane{flex:none;min-height:0}
  .console-pane>.panel-body{overflow:visible}
  .console-tabs button{font-size:12px}
}
@media(max-width:420px){.console-tabs button span{display:none}}

</style>
</head>
<body>
<header>
  <div class="top">
    <div class="brand"><div class="mark" aria-hidden="true">V</div><div><p class="eyebrow">Vehicle AI / Research cockpit</p><h1>智能汽车大模型统一交互平台 <span style="color:var(--accent)">V4.2</span></h1><p class="sub">驾驶场景理解 · 座舱 Agent · VehicleRAG · 结构化证据</p></div></div>
    <div class="head-links"><a class="pill" href="/scene/live/dashboard" target="_blank" rel="noopener">驾驶 V3 ↗</a><a class="pill" href="/docs" target="_blank" rel="noopener">Swagger ↗</a><a class="pill" href="/openapi.json" target="_blank" rel="noopener">OpenAPI JSON ↗</a></div>
  </div>
  <nav aria-label="功能导航"><a href="#drive">01 智能驾驶</a><a href="#assistant">02 座舱 / 知识问答</a><a href="#validator">03 JSON 校验</a><a href="#logs">04 API 实验日志</a></nav>
</header>
<main>
  <div class="notice">仿真研究与教学演示：左侧 CARLA 场景与右侧座舱模拟状态是不同的数据源。风险建议不会执行转向、油门或制动；座舱温度写操作必须经过用户确认。仅限本地可信环境使用。</div>
  <div class="workspace">
    <section class="panel" id="drive">
      <div class="panel-head"><div><p class="eyebrow">01 / Perception &amp; reasoning</p><h2>智能驾驶 · 多模态分析</h2><p class="sub">保留 V3 实时 YOLO、异步关键帧和离线报告</p></div><a class="btn small plain" href="/scene/live/dashboard" target="_blank" rel="noopener">独立打开 ↗</a></div>
      <iframe title="CARLA 实时与历史多模态 Dashboard V3" src="/scene/live/dashboard" loading="eager"></iframe>
      <div class="frame-foot">V3 原页面在同源窗口中显示，可独立滚动；实时检测与历史关键帧推理的时间口径保持不变。</div>
    </section>
    <div class="stack console panel" id="console">
      <div class="console-tabs" role="tablist" aria-label="座舱工作台">
        <button type="button" id="tab-assistant" role="tab" aria-selected="true" aria-controls="assistant" tabindex="0" data-tab="assistant"><span>01</span> 座舱 Agent</button>
        <button type="button" id="tab-validator" role="tab" aria-selected="false" aria-controls="validator" tabindex="-1" data-tab="validator"><span>02</span> JSON 校验</button>
        <button type="button" id="tab-logs" role="tab" aria-selected="false" aria-controls="logs" tabindex="-1" data-tab="logs"><span>03</span> API 日志</button>
      </div>
      <section class="panel console-pane" id="assistant" role="tabpanel" aria-labelledby="tab-assistant">
        <div class="panel-head"><div><p class="eyebrow">02 / Agent &amp; knowledge</p><h2>智能座舱 · Qwen Agent</h2><p class="sub">真实后端意图分流、工具调用与知识问答</p></div><span class="muted" id="connection" role="status">未读取状态</span></div>
        <div class="panel-body">
          <div class="status-line"><span class="status-label">模拟车辆状态 · 非 CARLA 自车遥测</span><button type="button" id="refresh" class="btn small">读取状态</button></div>
          <div class="metrics" aria-live="polite"><div class="metric"><strong id="speed">—</strong><span>模拟车速 / km/h</span></div><div class="metric"><strong id="temp">—</strong><span>模拟温度 / °C</span></div><div class="metric"><strong id="fan">—</strong><span>模拟风扇档位</span></div></div>
          <div class="divider"></div>
          <div class="chat-history" id="history" aria-live="polite" aria-label="座舱对话"><div class="empty" id="chat-empty">选择示例或输入自然语言指令，结果来自真实 API。页面不会自动调用模型。</div></div>
          <div class="examples">
            <button class="btn small" data-message="现在车内温度是多少？" data-mode="auto">查询温度</button>
            <button class="btn small" data-message="帮我把空调温度调到26度" data-mode="auto">设置 26℃</button>
            <button class="btn small" data-message="查询当前车辆状态" data-mode="auto">车辆状态</button>
            <button class="btn small" data-message="发动机温度过高应该怎么办？" data-mode="knowledge">知识库示例</button>
          </div>
          <form id="chat-form">
            <div class="form-row"><label for="mode">路由模式</label><select id="mode"><option value="auto">Auto · 规则自动分流</option><option value="vehicle">Vehicle · Qwen 工具规划</option><option value="knowledge">Knowledge · RAG 知识问答</option></select><label for="topk">Top K</label><select id="topk" style="flex:0 1 68px;min-width:54px"><option>1</option><option>2</option><option selected>3</option><option>4</option><option>5</option></select></div>
            <label class="muted" for="message">自然语言请求</label>
            <textarea id="message" rows="2" maxlength="500" required placeholder="例如：帮我把空调温度调到 26 度"></textarea>
            <div class="actions"><span id="chat-status" class="muted" role="status">/assistant/chat</span><button class="btn primary" type="submit" id="send">发送请求</button></div>
          </form>
          <div class="pending" id="pending" hidden>
            <h3>待确认 · 尚未执行模拟写操作</h3><p id="pending-text"></p><p id="countdown" class="muted"></p>
            <div class="actions"><button type="button" class="btn primary" id="approve">确认执行</button><button type="button" class="btn danger" id="reject">拒绝执行</button></div>
          </div>
          <div class="divider"></div>
          <h3>最新真实路由与工具结果</h3>
          <div class="trace"><div><span>后端 route</span><strong id="route">—</strong></div><div><span>后端 status</span><strong id="agent-status">—</strong></div><div><span>工具名称</span><strong id="tool-name">—</strong></div><div><span>HTTP / 请求总耗时</span><strong id="elapsed">—</strong></div></div>
          <details open><summary>Tool Call · 结构化规划</summary><pre id="tool-json">暂无数据</pre></details>
          <details><summary>Tool Result · 执行结果</summary><pre id="result-json">暂无数据</pre></details>
          <details><summary>RAG · 回答模式与知识来源</summary><pre id="source-json">暂无数据</pre></details>
          <p class="muted">显示 API 实际返回的结构化规划和结果，不等同于 Qwen 原始生成文本，也不伪造内部推理步骤。</p>
        </div>
      </section>
      <section class="panel console-pane" id="validator" role="tabpanel" aria-labelledby="tab-validator" hidden>
        <div class="panel-head"><div><p class="eyebrow">03 / Contract validation</p><h2>JSON 工具请求校验实验</h2><p class="sub">复用后端工具白名单及参数校验，不执行工具</p></div><span class="muted good">VALIDATE ONLY</span></div>
        <div class="panel-body">
          <div class="examples"><button type="button" class="btn small" id="sample-good">合法 26℃</button><button type="button" class="btn small" id="sample-range">越界 40℃</button><button type="button" class="btn small" id="sample-type">错误类型</button><button type="button" class="btn small" id="sample-plan" disabled>载入上次规划</button></div>
          <label for="json-input" class="muted">ToolCallRequest JSON</label>
          <textarea class="json-editor" id="json-input" spellcheck="false" rows="5">{
  "name": "set_cabin_temp",
  "arguments": {
    "temperature_c": 26
  }
}</textarea>
          <div class="actions"><span class="muted">不改变模拟温度</span><button type="button" class="btn primary" id="validate">检查 JSON</button></div>
          <p id="validation-label" class="result-label" role="status"></p>
          <details open><summary>后端校验响应 / 错误</summary><pre id="validation-result">尚未提交校验</pre></details>
          <p class="muted">先检查 JSON 语法，再调用只校验接口；合法工具应返回 200，非法参数或工具应返回 422 / 403。</p>
        </div>
      </section>
  <section class="panel console-pane" id="logs" role="tabpanel" aria-labelledby="tab-logs" hidden>
    <div class="panel-head"><div><p class="eyebrow">04 / Request inspector</p><h2>API 实验日志</h2><p class="sub">本页面实际 HTTP 请求、返回码与浏览器观察的请求耗时 · 当前页面会话</p></div><button class="btn small plain" type="button" id="clear-logs">清空本页日志</button></div>
    <div class="panel-body log-layout">
      <div id="log-list" class="log-list" aria-label="请求记录"><div class="empty">暂无请求记录</div></div>
      <div class="log-detail"><h3>选中记录 · Request / Response</h3><pre id="log-detail">点击左侧记录查看实际 JSON。</pre></div>
    </div>
  </section>
    </div>
  </div>
</main>
<footer><span>VEHICLE AI / Dashboard V4.2 · 本地仿真研究</span><span>保留既有 Agent、RAG、CARLA V3 与 Swagger · 模型结果须人工复核</span></footer>
<script>
"use strict";
const $ = id => document.getElementById(id);
const pretty = value => value === undefined ? "—" : JSON.stringify(value, null, 2);
let busy = false, refreshing = false, validating = false, pending = null, lastPlan = null;
let logs = [], activeLog = null, logSequence = 0;
const maxLogs = 40;

// Sensitive one-time tokens are retained only in the pending-action object.
function redact(value) {
  if (Array.isArray(value)) return value.map(redact);
  if (value && typeof value === "object") {
    return Object.fromEntries(Object.entries(value).map(([key, val]) =>
      [key, /(?:token|cookie|authorization|password|secret|api[_-]?key)/i.test(key) ? "[redacted]" : redact(val)]));
  }
  return value;
}
function renderLogs() {
  const list = $("log-list"); list.replaceChildren();
  if (!logs.length) {
    const empty = document.createElement("div"); empty.className = "empty"; empty.textContent = "暂无请求记录"; list.append(empty);
    $("log-detail").textContent = "点击左侧记录查看实际 JSON。"; return;
  }
  logs.forEach(entry => {
    const row = document.createElement("button"); row.type = "button";
    row.className = "log-row" + (entry.id === activeLog ? " selected" : "");
    const path = document.createElement("span"); path.className = "path"; path.textContent = entry.method + " " + entry.path;
    const code = document.createElement("span"); code.className = "code" + (entry.ok ? "" : " fail"); code.textContent = entry.http_status;
    const meta = document.createElement("span"); meta.className = "meta"; meta.textContent = entry.time + " · " + entry.elapsed_ms + " ms";
    row.append(path, code, meta);
    row.addEventListener("click", () => { activeLog = entry.id; renderLogs(); });
    list.append(row);
  });
  const selected = logs.find(x => x.id === activeLog);
  $("log-detail").textContent = selected ? pretty({
    method:selected.method,path:selected.path,http_status:selected.http_status,elapsed_ms:selected.elapsed_ms,
    request:selected.request,response:selected.response
  }) : "点击左侧记录查看实际 JSON。";
}
async function api(path, method = "GET", body) {
  const start = performance.now();
  let responseData, code = "NETWORK", ok = false;
  try {
    const options = {method, credentials:"same-origin", cache:"no-store",
      headers:{"Accept":"application/json"}};
    if (body !== undefined) {
      options.headers["Content-Type"] = "application/json";
      options.body = JSON.stringify(body);
    }
    const response = await fetch(path, options);
    code = response.status;
    const raw = await response.text();
    try { responseData = JSON.parse(raw); }
    catch (_) { responseData = {detail:"响应不是 JSON", text:raw.slice(0, 2000)}; }
    ok = response.ok;
  } catch (error) {
    responseData = {detail:"网络请求失败：" + String(error.message || error)};
  }
  const ms = Math.round(performance.now() - start);
  const entry = {id:++logSequence,time:new Date().toLocaleTimeString("zh-CN",{hour12:false}),
    method,path,http_status:code,elapsed_ms:ms,ok,request:body===undefined?null:redact(body),response:redact(responseData)};
  logs.unshift(entry); logs = logs.slice(0,maxLogs); activeLog = entry.id; renderLogs();
  return {ok,code,data:responseData,ms};
}
function errorText(r) {
  const detail = r.data && r.data.detail;
  return "HTTP " + r.code + " · " + (typeof detail === "string" ? detail : pretty(detail ?? r.data));
}
function addMessage(kind, text) {
  $("chat-empty")?.remove();
  const node = document.createElement("div"); node.className = "message " + kind;
  const who = document.createElement("small"); who.textContent = kind==="user" ? "用户" : kind==="error" ? "请求状态" : "助手 · API 响应";
  node.append(who, document.createTextNode(String(text)));
  $("history").append(node);
  $("history").scrollTop = $("history").scrollHeight;
}
function syncControls() {
  $("send").disabled = busy || !!pending;
  $("approve").disabled = busy || !pending;
  $("reject").disabled = busy || !pending;
  $("refresh").disabled = refreshing;
  $("validate").disabled = validating;
  $("sample-plan").disabled = !lastPlan;
}
function setValue(id, value) { $(id).textContent = value === null || value === undefined ? "—" : String(value); }
async function refreshVehicle() {
  if (refreshing) return;
  refreshing = true; syncControls();
  $("connection").textContent = "读取中…"; $("connection").className = "muted";
  try {
    const r = await api("/vehicle/status");
    const d = r.data && r.data.data;
    if (r.ok && r.data.ok === true && d && typeof d === "object") {
      setValue("speed",d.speed_kmh); setValue("temp",d.cabin_temp_c); setValue("fan",d.fan_level);
      $("connection").textContent = "模拟状态已读取"; $("connection").className = "muted good";
    } else {
      ["speed","temp","fan"].forEach(id=>setValue(id,null));
      $("connection").textContent = "读取失败 · " + r.code; $("connection").className = "muted bad";
    }
  } finally { refreshing = false; syncControls(); }
}
function showTrace(r, confirmation = false) {
  const d = r.data || {};
  if (!confirmation) {
    setValue("route",d.route); setValue("tool-name",d.tool_call?.name);
    lastPlan = d.tool_call && typeof d.tool_call==="object" ? d.tool_call : null;
    $("tool-json").textContent = pretty(lastPlan);
    $("source-json").textContent = pretty({answer_mode:d.answer_mode??null,sources:d.sources??[]});
  } else if (d.tool_name) { setValue("tool-name",d.tool_name); }
  setValue("agent-status",d.status ?? (r.ok ? "未返回 status" : "HTTP error"));
  setValue("elapsed",`${r.code} / ${r.ms} ms`);
  $("result-json").textContent = pretty(d.tool_result??null);
  syncControls();
}
function updatePending() {
  if (!pending) { $("pending").hidden = true; syncControls(); return; }
  const left = pending.deadline === null ? null : Math.max(0,Math.ceil((pending.deadline-Date.now())/1000));
  $("pending").hidden = false;
  $("countdown").textContent = left===null ? "有效期由服务端判断。" : `约剩余 ${left} 秒（以服务端为准）`;
  if (left === 0 && !busy) {
    pending=null; $("pending").hidden=true;
    addMessage("error","本地确认倒计时已结束，未提交执行；请重新发送指令。");
    $("chat-status").textContent="确认已过期（本地）";
  }
  syncControls();
}
function setPending(d) {
  const token = d.confirmation_token;
  if (typeof token !== "string" || !token) {
    addMessage("error","后端要求确认，但未返回有效令牌；不会执行。"); return;
  }
  const seconds = d.expires_in_seconds;
  pending = {token,deadline:typeof seconds==="number"&&Number.isFinite(seconds)&&seconds>=0 ? Date.now()+seconds*1000 : null};
  $("pending-text").textContent = String(d.reply??"请核对后端返回的工具及参数。")+"\n"+pretty(redact(d.tool_call??null));
  updatePending();
}
async function sendChat(event) {
  event.preventDefault();
  if (busy || pending) return;
  const message = $("message").value.trim();
  if (!message) return;
  const mode = $("mode").value, top_k = Number($("topk").value);
  busy=true;syncControls();$("chat-status").textContent="正在请求后端…";
  addMessage("user",message);$("message").value="";
  try {
    const r = await api("/assistant/chat","POST",{message,mode,top_k});
    showTrace(r);
    if (!r.ok || r.data?.ok !== true) {
      addMessage("error",errorText(r)); $("chat-status").textContent="请求未成功 · "+r.code; return;
    }
    const d=r.data;
    addMessage("assistant",d.reply??"后端未返回 reply 字段，请查看原始响应。");
    $("chat-status").textContent="已收到真实响应 · "+r.code;
    if (d.status==="confirmation_required") setPending(d);
    else if (d.route==="vehicle" && d.tool_result!==null && d.tool_result!==undefined) await refreshVehicle();
  } finally {busy=false;updatePending();}
}
async function resolvePending(approved) {
  if (busy || !pending) return;
  if (pending.deadline!==null && Date.now()>=pending.deadline) { updatePending(); return; }
  // Only an explicit click on the confirm/reject buttons can reach this request.
  const token=pending.token;
  busy=true;syncControls();$("chat-status").textContent=approved?"正在提交用户确认…":"正在提交用户拒绝…";
  try {
    const r=await api("/agent/confirm","POST",{confirmation_token:token,approved});
    showTrace(r,true);
    if (r.ok && r.data?.ok===true) {
      addMessage("assistant",r.data.reply??"已收到确认接口响应，请查看执行结果。");
      $("chat-status").textContent="确认响应 · "+r.code;
    } else if (!approved && r.code===400 && typeof r.data?.detail==="string" && r.data.detail.includes("用户已拒绝执行操作")) {
      addMessage("assistant",r.data.detail);
      $("chat-status").textContent="后端已拒绝操作 · HTTP 400（原接口约定）";
    } else {
      addMessage("error",errorText(r)+(r.code==="NETWORK"?"；执行结果无法确认，请读取模拟状态，不要自动重试。":""));
      $("chat-status").textContent="确认请求未成功 · "+r.code;
    }
    await refreshVehicle();
  } finally {pending=null;busy=false;updatePending();}
}
function loadSample(value) {
  $("json-input").value=pretty(value);
  $("validation-label").textContent="";$("validation-result").textContent="尚未提交此 JSON 的校验";
}
async function validateJSON() {
  if (validating) return;
  let value;
  try {value=JSON.parse($("json-input").value);}
  catch(error) {
    $("validation-label").textContent="语法错误 · 浏览器 JSON.parse 未通过（未发送 HTTP 请求）";
    $("validation-label").className="result-label bad";
    $("validation-result").textContent=String(error.message||error);return;
  }
  validating=true;syncControls();$("validation-label").textContent="JSON 语法通过，正在请求后端校验…";$("validation-label").className="result-label";
  try {
    const r=await api("/dashboard/validate-tool","POST",value);
    const passed=r.ok && r.data?.validation==="passed" && r.data?.executed===false;
    $("validation-label").textContent=passed?"PASS · 后端校验通过，未执行工具":r.code==="NETWORK"?"无法核实 · 网络请求失败":"未通过 / 未确认 · HTTP "+r.code;
    $("validation-label").className="result-label "+(passed?"good":"bad");
    $("validation-result").textContent=pretty(redact(r.data));
  } finally {validating=false;syncControls();}
}
$("refresh").addEventListener("click",refreshVehicle);
$("chat-form").addEventListener("submit",sendChat);
$("approve").addEventListener("click",()=>resolvePending(true));
$("reject").addEventListener("click",()=>resolvePending(false));
document.querySelectorAll("[data-message]").forEach(btn=>btn.addEventListener("click",()=>{
  $("message").value=btn.dataset.message;$("mode").value=btn.dataset.mode;$("message").focus();
}));
$("sample-good").addEventListener("click",()=>loadSample({name:"set_cabin_temp",arguments:{temperature_c:26}}));
$("sample-range").addEventListener("click",()=>loadSample({name:"set_cabin_temp",arguments:{temperature_c:40}}));
$("sample-type").addEventListener("click",()=>loadSample({name:"set_cabin_temp",arguments:{temperature_c:"26"}}));
$("sample-plan").addEventListener("click",()=>{if(lastPlan)loadSample(lastPlan);});
$("validate").addEventListener("click",validateJSON);
$("clear-logs").addEventListener("click",()=>{logs=[];activeLog=null;renderLogs();});

const tabIds = ["assistant","validator","logs"];
function selectTab(id, focus = false) {
  if (!tabIds.includes(id)) return;
  for (const name of tabIds) {
    const selected = name === id;
    $(name).hidden = !selected;
    const button = $("tab-" + name);
    button.setAttribute("aria-selected", String(selected));
    button.tabIndex = selected ? 0 : -1;
  }
  if (focus) $("tab-" + id).focus();
}
document.querySelectorAll(".console-tabs [data-tab]").forEach(button => {
  button.addEventListener("click", () => selectTab(button.dataset.tab));
  button.addEventListener("keydown", event => {
    const index = tabIds.indexOf(button.dataset.tab);
    let next = index;
    if (event.key === "ArrowRight") next = (index + 1) % tabIds.length;
    else if (event.key === "ArrowLeft") next = (index + tabIds.length - 1) % tabIds.length;
    else if (event.key === "Home") next = 0;
    else if (event.key === "End") next = tabIds.length - 1;
    else return;
    event.preventDefault();
    selectTab(tabIds[next], true);
  });
});
document.querySelectorAll("header nav a").forEach(link => {
  link.addEventListener("click", event => {
    const id = link.getAttribute("href").slice(1);
    if (tabIds.includes(id)) {
      event.preventDefault();
      selectTab(id);
      if (window.matchMedia("(max-width: 900px)").matches) {
        $("console").scrollIntoView({block:"start"});
      }
    }
  });
});
selectTab(tabIds.includes(location.hash.slice(1)) ? location.hash.slice(1) : "assistant");

setInterval(updatePending,1000);
syncControls();
refreshVehicle();
</script>
</body>
</html>'''
