"""统一入口使用的保守路由规则。

说明：
- 仅做有限的意图分流，绝不是通用大模型意图理解。
- 不能从这里调用或执行车辆工具。
- 不认识的请求默认拒绝，而不是猜测一个写操作。
"""

import re
from typing import Literal

Route = Literal["vehicle", "knowledge", "unsupported"]

# 涉及危险或不存在的车辆操作，不调用模型来“猜测”一个允许的工具。
UNSUPPORTED_COMMANDS = (
    "打开车门", "关闭车门", "开车门", "关车门",
    "播放音乐", "播放歌曲", "放首歌", "切歌",
    "导航去", "导航到", "设导航", "打开天窗",
    "打开后备箱", "启动发动机", "熄火",
)

# 这些词表明用户更可能是在咨询故障、原理或维修知识。
KNOWLEDGE_CUES = (
    "怎么办", "怎么处理", "如何处理", "为什么", "原因",
    "排查", "检查", "故障", "异常", "报警", "警告",
    "不制冷", "不启动", "无法启动", "泄漏", "漏液",
    "过热", "过高", "抖动", "维修", "保养", "损坏",
    "刮不干净", "高压系统", "气味", "异响",
)

# 汽车故障类名词，用于识别不包含“怎么办”的知识问题。
KNOWLEDGE_TERMS = (
    "发动机", "冷却液", "机油", "制动", "刹车", "胎压",
    "轮胎", "蓄电池", "方向盘", "雨刮", "雨刷",
    "传感器", "自适应巡航", "散热器", "新能源汽车",
    "电池绝缘", "压缩机",
)

TEMP_TARGETS = ("空调", "车内温度", "座舱温度", "车厢温度", "舱内温度")
SET_VERBS = ("设置", "设为", "调到", "调成", "调至", "调为", "调节", "调高", "调低")
STATUS_TARGETS = ("车速", "车辆状态", "风扇档位", "风量档位", "车内温度", "座舱温度", "车厢温度")
QUERY_VERBS = ("查询", "查一下", "查看", "多少", "现在", "当前", "几度", "状态", "告诉我", "显示")


def classify_message(message: str) -> Route:
    """用规则区分明确的知识问题、支持的车辆请求和其他请求。"""
    text = re.sub(r"\s+", "", message).lower()
    if not text:
        return "unsupported"

    # 首先识别明确不支持的命令，不能让误选写工具的模型自动进入确认流程。
    if any(term in text for term in UNSUPPORTED_COMMANDS):
        return "unsupported"

    # 故障/安全问答优先于温度等车辆关键词。
    if any(term in text for term in KNOWLEDGE_CUES):
        return "knowledge"

    # 明确的空调温度修改。
    if any(term in text for term in TEMP_TARGETS) and any(
        verb in text for verb in SET_VERBS
    ):
        return "vehicle"

    # 明确的状态查询。
    if any(term in text for term in STATUS_TARGETS) and any(
        verb in text for verb in QUERY_VERBS
    ):
        return "vehicle"

    # 未出现明确请求类型但涉及汽车技术名词，进入知识查询。
    if any(term in text for term in KNOWLEDGE_TERMS):
        return "knowledge"

    return "unsupported"
