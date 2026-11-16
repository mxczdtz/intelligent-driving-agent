"""需要 python-multipart 和 Pillow，但无需下载真实 YOLO 模型。"""
import io
import pytest

pytest.importorskip("multipart")
PIL = pytest.importorskip("PIL")
from PIL import Image
from fastapi.testclient import TestClient
from app.scene_service import app
from app.scene_risk import Detection
import app.scene_api as scene_api


def jpeg_bytes():
    output = io.BytesIO()
    Image.new("RGB", (640, 480), (180, 190, 200)).save(output, format="JPEG")
    return output.getvalue()


def test_scene_endpoint_mock_detector(monkeypatch):
    monkeypatch.setattr(
        scene_api, "detect_objects",
        lambda image: [Detection(
            class_name="person",
            confidence=0.95,
            box_xyxy=(260, 120, 400, 460),
        )],
    )
    response = TestClient(app).post(
        "/scene/analyze",
        data={"speed_kmh": "35"},
        files={"file": ("road.jpg", jpeg_bytes(), "image/jpeg")},
    )
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["risk"]["risk_level"] == "high"
    assert result["detections"][0]["class_name"] == "person"
    assert result["image_width"] == 640


def test_scene_endpoint_rejects_bad_image():
    response = TestClient(app).post(
        "/scene/analyze",
        files={"file": ("broken.png", b"not an image", "image/png")},
    )
    assert response.status_code == 422


def test_scene_endpoint_rejects_bad_speed():
    response = TestClient(app).post(
        "/scene/analyze",
        data={"speed_kmh": "-10"},
        files={"file": ("road.jpg", jpeg_bytes(), "image/jpeg")},
    )
    assert response.status_code == 422
