"""YOLO 图片目标检测，运行时延迟加载模型。"""
import os
from functools import lru_cache
from PIL import Image
from app.scene_risk import Detection

RELEVANT = {
    "person", "car", "truck", "bus", "motorcycle", "bicycle",
    "traffic light", "stop sign",
}


@lru_cache(maxsize=1)
def get_yolo():
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("缺少 ultralytics；请安装 requirements_scene.txt") from exc
    # 推荐先使用本地已下载权重；仅传文件名时，Ultralytics 可能联网下载
    model_path = os.getenv("SCENE_YOLO_MODEL", "yolo11n.pt")
    return YOLO(model_path)


def detect_objects(image: Image.Image) -> list[Detection]:
    """仅返回 COCO 常见道路目标；不同预训练模型的类别需单独核验。"""
    model = get_yolo()
    kwargs = {"source": image, "conf": 0.35, "imgsz": 640, "verbose": False}
    device = os.getenv("SCENE_YOLO_DEVICE")
    if device:
        kwargs["device"] = device
    results = model.predict(**kwargs)
    if not results:
        raise RuntimeError("YOLO 未返回推理结果")
    result = results[0]
    if result.boxes is None:
        return []
    names = result.names
    coords = result.boxes.xyxy.cpu().tolist()
    confidences = result.boxes.conf.cpu().tolist()
    class_ids = result.boxes.cls.cpu().tolist()
    detections = []
    for xyxy, conf, class_id in zip(coords, confidences, class_ids):
        index = int(class_id)
        name = names[index] if not isinstance(names, dict) else names.get(index, str(index))
        if name not in RELEVANT:
            continue
        detections.append(
            Detection(
                class_name=name,
                confidence=float(conf),
                box_xyxy=tuple(float(v) for v in xyxy),
            )
        )
    return detections
