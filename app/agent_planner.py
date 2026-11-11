
import json

from app.qwen_engine import generate_text


# 模型可以选择的工具
ALLOWED_NAMES = {
    "get_vehicle_status",
    "get_cabin_temp",
    "set_cabin_temp",
    "none"
}


# Agent 工具选择提示词 V3
SYSTEM_PROMPT = """
你是汽车智能座舱教学模拟系统的工具选择器。
所有车辆状态均为模拟数据，不连接真实车辆。

你只能从下面4个名称中选择一个：
get_vehicle_status、get_cabin_temp、set_cabin_temp、none。
禁止创造或输出其他工具名称。

可用工具：

1. get_vehicle_status
查询车辆状态，包括车速、车内温度和风扇档位。
不接受任何参数，arguments必须为{}。

2. get_cabin_temp
查询当前车内温度。
不接受任何参数，arguments必须为{}。

3. set_cabin_temp
设置模拟车内温度。
arguments只能包含temperature_c一个数字参数。
温度允许范围是16到30摄氏度。
不得添加额外参数，不得擅自更改用户要求的温度。
实际安全校验由Python执行层负责。

4. none
当用户要求的操作不属于上述3个车辆工具时，选择none。
当用户意图不明确、缺少必要参数或上下文时，也选择none。
arguments必须为{}。

重要规则：
只能选择已提供的工具，不能编造新工具。
无参数工具的arguments必须为{}。
禁止添加任何未定义的参数字段。
不支持的操作必须选择none。

只输出一个合法JSON对象。
顶层只能有name和arguments两个字段。
不要输出Markdown、解释或其他文字。
不要声称已经执行车辆操作。
"""


def plan_tool(message: str) -> dict:
    """让 Qwen 选择工具，但不实际执行。"""

    raw = generate_text([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": message}
    ])


    # 清理模型输出前后的空白字符
    raw = raw.strip()

    # 如果模型输出被完整的 Markdown 代码块包裹，移除外层标记
    lines = raw.splitlines()

    if (
        len(lines) >= 3
        and lines[0].strip().lower() in ("```json", "```")
        and lines[-1].strip() == "```"
    ):
        raw = "\n".join(lines[1:-1]).strip()


    # 模型直接输出 none 时，转换为标准的拒绝执行结果
    if raw.strip().lower() == "none":
        return {
            "name": "none",
            "arguments": {}
        }

    # 解析模型生成的 JSON
    try:
        result = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"模型输出不是合法JSON: {raw}"
        ) from exc

    # 检查 JSON 基本结构
    if (
        not isinstance(result, dict)
        or set(result.keys()) != {"name", "arguments"}
    ):
        raise ValueError("模型输出结构不正确")

    name = result["name"]

    if not isinstance(name, str) or name not in ALLOWED_NAMES:
        raise ValueError("模型选择了未知工具")

    if not isinstance(result["arguments"], dict):
        raise ValueError("工具参数必须是JSON对象")

    return result