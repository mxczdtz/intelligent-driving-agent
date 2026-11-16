"""可独立运行的视觉服务：python -m uvicorn app.scene_service:app --port 8001"""
from fastapi import FastAPI
from app.scene_api import router

app = FastAPI(
    title="Intelligent Driving - Scene Risk Prototype",
    version="0.1.0",
    description="单帧 YOLO 目标检测与启发式风险提示，完全不连接真实车辆。",
)
app.include_router(router)


@app.get("/health")
def health():
    return {"ok": True, "service": "scene-risk-prototype"}
