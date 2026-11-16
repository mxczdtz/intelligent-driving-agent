"""基于二维检测框的教学级风险提示规则，不执行车辆控制。"""
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field, ConfigDict


class RiskLevel(str, Enum):
    UNKNOWN = "unknown"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Detection(BaseModel):
    model_config = ConfigDict(extra="forbid")
    class_name: str
    confidence: float = Field(ge=0, le=1)
    box_xyxy: tuple[float, float, float, float]


class SceneInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    image_width: int = Field(gt=0, le=8192)
    image_height: int = Field(gt=0, le=8192)
    ego_speed_kmh: Optional[float] = Field(default=None, ge=0, le=200)
    detections: list[Detection] = Field(default_factory=list)


class RiskReport(BaseModel):
    model_config = ConfigDict(extra="forbid")
    risk_level: RiskLevel
    key_factors: list[str]
    evidence: list[str]
    driving_suggestion: str
    method: str = "2D bounding-box heuristic (educational prototype)"
    limitations: list[str] = Field(default_factory=list)


class SceneResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    image_width: int
    image_height: int
    ego_speed_kmh: Optional[float]
    detections: list[Detection]
    risk: RiskReport
    inference_ms: float = Field(ge=0)


def assess_scene(scene: SceneInput) -> RiskReport:
    """有条件的视觉风险提示，不估计真实距离、TTC、车道或灯色。"""
    general_limits = [
        "单帧二维检测框不能可靠判断目标实际距离、相对速度、车道或碰撞时间。",
        "本结果仅用于离线仿真与软件测试，不能控制真实车辆。",
        "未检测出目标不代表道路安全，存在漏检、遮挡和域偏移风险。",
    ]
    if scene.ego_speed_kmh is None:
        general_limits.append("未提供自车速度，风险分级缺少动态上下文。")

    if not scene.detections:
        return RiskReport(
            risk_level=RiskLevel.UNKNOWN,
            key_factors=["当前帧未检测出目标，不能据此推断道路安全。"],
            evidence=[],
            driving_suggestion="无法依据当前帧判断风险；应结合更多传感器和连续帧核查。",
            limitations=general_limits,
        )

    severity = 1  # 观察到目标但缺少明显图像警示线索，保守标为 low
    factors: list[str] = []
    evidence: list[str] = []

    for idx, det in enumerate(scene.detections):
        x1, y1, x2, y2 = det.box_xyxy
        if not (0 <= x1 < x2 <= scene.image_width and 0 <= y1 < y2 <= scene.image_height):
            raise ValueError(f"detections[{idx}].box_xyxy 超出图像或坐标顺序错误")

        name = det.class_name.lower().strip()
        center_x = (x1 + x2) / (2 * scene.image_width)
        bottom_y = y2 / scene.image_height
        width_ratio = (x2 - x1) / scene.image_width

        # 这些只是画面区域启发式，不应称为“本车道”或“碰撞距离”
        in_center = 0.35 <= center_x <= 0.65
        visually_low = bottom_y >= 0.58

        if name == "person" and in_center and visually_low:
            proposed = 3 if scene.ego_speed_kmh is not None and scene.ego_speed_kmh >= 15 else 2
            severity = max(severity, proposed)
            factors.append("行人检测框位于画面中央偏下位置，需关注潜在冲突。")
            evidence.append(f"detection_{idx}: person, conf={det.confidence:.2f}")
        elif name in {"car", "truck", "bus", "motorcycle", "bicycle"} and in_center and bottom_y >= 0.63 and width_ratio >= 0.24:
            proposed = 3 if scene.ego_speed_kmh is not None and scene.ego_speed_kmh >= 30 else 2
            severity = max(severity, proposed)
            factors.append("车辆或骑行目标的检测框在画面中央偏下且较大；不等同于距离估计。")
            evidence.append(f"detection_{idx}: {name}, conf={det.confidence:.2f}")
        elif name in {"traffic light", "stop sign"}:
            severity = max(severity, 2)
            factors.append("识别到交通控制设施，应进一步核验信号状态和道路规则。")
            evidence.append(f"detection_{idx}: {name}, conf={det.confidence:.2f}")

    if not factors:
        factors = ["检测到道路相关目标，但当前规则未发现明显的二维图像警示线索。"]
        evidence = [f"detection_{idx}: {det.class_name}, conf={det.confidence:.2f}"
                    for idx, det in enumerate(scene.detections[:5])]

    level = {1: RiskLevel.LOW, 2: RiskLevel.MEDIUM, 3: RiskLevel.HIGH}[severity]
    advice = {
        RiskLevel.LOW: "保持观察；单帧结果不支持作出安全通行结论。",
        RiskLevel.MEDIUM: "建议人工复核连续帧、车速与道路信息，不自动生成车辆控制指令。",
        RiskLevel.HIGH: "检测到较强的图像风险提示；应由更完整的感知与安全系统复核，不自动刹车或转向。",
    }[level]
    return RiskReport(
        risk_level=level,
        key_factors=list(dict.fromkeys(factors)),
        evidence=evidence,
        driving_suggestion=advice,
        limitations=general_limits,
    )
