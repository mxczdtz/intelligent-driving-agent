"""新增视觉分析接口：仅接收图片进行离线演示，不调用真实车辆。"""
import io
import logging
from time import perf_counter

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from app.scene_detector import detect_objects
from app.scene_risk import SceneInput, SceneResponse, assess_scene

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/scene", tags=["Scene Perception (prototype)"])

MAX_FILE_BYTES = 8 * 1024 * 1024
MAX_PIXELS = 16_000_000


@router.post("/analyze", response_model=SceneResponse)
async def analyze_scene(
    file: UploadFile = File(..., description="JPEG / PNG 道路场景图像"),
    speed_kmh: float | None = Form(default=None, ge=0, le=200),
):
    start = perf_counter()
    raw = await file.read(MAX_FILE_BYTES + 1)
    if len(raw) > MAX_FILE_BYTES:
        raise HTTPException(status_code=413, detail="图像文件超过 8 MB")
    if not raw:
        raise HTTPException(status_code=422, detail="图像为空")

    try:
        with Image.open(io.BytesIO(raw)) as source:
            if source.format not in {"JPEG", "PNG"}:
                raise HTTPException(status_code=415, detail="只支持 JPEG / PNG 图像")
            w, h = source.size
            if w <= 0 or h <= 0 or w * h > MAX_PIXELS:
                raise HTTPException(status_code=413, detail="图像像素超过限制")
            image = source.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise HTTPException(status_code=422, detail="图像损坏或格式无效") from exc

    try:
        detected = detect_objects(image)
        scene = SceneInput(
            image_width=w,
            image_height=h,
            ego_speed_kmh=speed_kmh,
            detections=detected,
        )
        report = assess_scene(scene)
    except (RuntimeError, ImportError) as exc:
        logger.warning("scene model unavailable: %s", exc)
        raise HTTPException(status_code=503, detail="视觉模型不可用，请检查本地模型文件与安装环境") from exc
    except ValueError as exc:
        logger.exception("invalid detection result")
        raise HTTPException(status_code=500, detail="视觉结果格式校验失败") from exc

    return SceneResponse(
        image_width=w,
        image_height=h,
        ego_speed_kmh=speed_kmh,
        detections=detected,
        risk=report,
        inference_ms=round((perf_counter() - start) * 1000, 2),
    )
