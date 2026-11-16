
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
