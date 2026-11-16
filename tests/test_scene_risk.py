"""不需 YOLO 模型权重，直接测试风险提示规则。"""
import pytest
from app.scene_risk import Detection, RiskLevel, SceneInput, assess_scene


def make_scene(*objects, speed=25):
    return SceneInput(
        image_width=1000,
        image_height=600,
        ego_speed_kmh=speed,
        detections=list(objects),
    )


def det(kind, box=(400, 200, 580, 570)):
    return Detection(class_name=kind, confidence=0.9, box_xyxy=box)


def test_empty_detections_is_unknown():
    report = assess_scene(make_scene())
    assert report.risk_level == RiskLevel.UNKNOWN
    assert not report.evidence


def test_pedestrian_center_lower_high_with_speed():
    report = assess_scene(make_scene(det("person"), speed=30))
    assert report.risk_level == RiskLevel.HIGH
    assert "person" in report.evidence[0]


def test_pedestrian_without_speed_not_high():
    report = assess_scene(make_scene(det("person"), speed=None))
    assert report.risk_level == RiskLevel.MEDIUM
    assert any("未提供" in x for x in report.limitations)


def test_large_center_car_high():
    report = assess_scene(make_scene(det("car", (350, 160, 700, 570)), speed=50))
    assert report.risk_level == RiskLevel.HIGH


def test_traffic_light_cannot_claim_red_signal():
    report = assess_scene(make_scene(det("traffic light", (850, 50, 900, 170))))
    assert report.risk_level == RiskLevel.MEDIUM
    assert "红灯" not in str(report.model_dump())


def test_peripheral_object_is_not_asserted_collision():
    report = assess_scene(make_scene(det("car", (30, 70, 200, 220))))
    assert report.risk_level == RiskLevel.LOW
    assert "碰撞时间" in str(report.limitations)


def test_invalid_boxes_fail_closed():
    with pytest.raises(ValueError):
        assess_scene(make_scene(det("person", (900, 10, 1200, 300))))


def test_schema_rejects_extra_fields():
    with pytest.raises(Exception):
        SceneInput(
            image_width=1000, image_height=600, ego_speed_kmh=30,
            detections=[], controls={"brake": 1},
        )
