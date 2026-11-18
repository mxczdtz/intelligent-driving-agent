"""Phase 5.2.1: local CARLA RGB ingress + per-frame YOLO perception.

This module receives RGB camera frames from a separate CARLA client process.
It does not read CARLA actor state, simulator speed, metric range, or issue
vehicle-control commands. Only image pixels are passed to YOLO and the existing
2D educational scene-risk heuristic.
"""
from __future__ import annotations

import io
import json
from pathlib import Path
import logging
import math
import re
import threading
import time
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, Response
from PIL import Image, UnidentifiedImageError
from starlette.concurrency import run_in_threadpool

from app.scene_detector import detect_objects
from app.scene_risk import SceneInput, assess_scene

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/scene/live", tags=["CARLA Live RGB"])

MAX_FILE_BYTES = 5 * 1024 * 1024
MAX_PIXELS = 8_000_000
SESSION_RE = re.compile(r"^[A-Za-z0-9_-]{8,64}$")
INFERENCE_LOCK = threading.Lock()
STATE_LOCK = threading.RLock()

_STATE: dict[str, Any] = {
    "session_id": None,
    "active": False,
    "started_at": None,
    "stopped_at": None,
    "last_seen_monotonic": 0.0,
    "frames_received": 0,
    "frames_processed": 0,
    "last_frame_id": None,
    "latest": None,
    "latest_jpeg": None,
    "last_error": None,
}


def _reset_state(session_id: str) -> None:
    _STATE.update({
        "session_id": session_id,
        "active": True,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "stopped_at": None,
        "last_seen_monotonic": time.monotonic(),
        "frames_received": 0,
        "frames_processed": 0,
        "last_frame_id": None,
        "latest": None,
        "latest_jpeg": None,
        "last_error": None,
    })


def _infer(image: Image.Image) -> tuple[list[Any], Any, float]:
    start = time.perf_counter()
    # The client is single-stream and rate-limited; serialize inference in case
    # another local request hits the existing YOLO model concurrently.
    with INFERENCE_LOCK:
        detections = detect_objects(image)
        scene = SceneInput(
            image_width=image.width,
            image_height=image.height,
            ego_speed_kmh=None,  # CARLA speed/actor truth is deliberately excluded.
            detections=detections,
        )
        risk = assess_scene(scene)
    return detections, risk, (time.perf_counter() - start) * 1000.0


@router.post("/frame", summary="接收一帧 CARLA RGB 并运行 YOLO")
async def receive_live_frame(
    file: UploadFile = File(..., description="CARLA RGB camera JPEG/PNG"),
    session_id: str = Form(..., min_length=8, max_length=64),
    frame_id: int = Form(..., ge=0),
    sim_timestamp: float = Form(..., ge=0),
):
    if not SESSION_RE.fullmatch(session_id):
        raise HTTPException(status_code=422, detail="session_id 格式无效。")
    if not math.isfinite(sim_timestamp):
        raise HTTPException(status_code=422, detail="sim_timestamp 必须是有限数值。")

    raw = await file.read(MAX_FILE_BYTES + 1)
    if len(raw) > MAX_FILE_BYTES:
        raise HTTPException(status_code=413, detail="单帧图像超过 5 MB。")
    if not raw:
        raise HTTPException(status_code=422, detail="收到空图像。")

    try:
        with Image.open(io.BytesIO(raw)) as source:
            if source.format not in {"JPEG", "PNG"}:
                raise HTTPException(status_code=415, detail="只支持 JPEG / PNG。")
            width, height = source.size
            if width < 16 or height < 16 or width * height > MAX_PIXELS:
                raise HTTPException(status_code=413, detail="图像尺寸无效或超过 8 MP。")
            image = source.convert("RGB")
            image.load()
    except HTTPException:
        raise
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise HTTPException(status_code=422, detail="图像损坏或格式无效。") from exc

    with STATE_LOCK:
        current_session = _STATE["session_id"]
        if current_session != session_id:
            # Do not silently steal a currently active stream. A crashed client
            # can be replaced after 15 seconds without a received frame.
            stale = time.monotonic() - float(_STATE["last_seen_monotonic"] or 0.0) > 15.0
            if _STATE["active"] and current_session and not stale:
                raise HTTPException(
                    status_code=409,
                    detail=f"已有实时会话 {current_session} 正在运行；先停止旧客户端。",
                )
            _reset_state(session_id)
        else:
            previous_id = _STATE["last_frame_id"]
            if previous_id is not None and frame_id <= previous_id:
                raise HTTPException(
                    status_code=409,
                    detail=f"帧号必须递增；上一帧为 {previous_id}，收到 {frame_id}。",
                )
            _STATE["last_seen_monotonic"] = time.monotonic()
        _STATE["frames_received"] += 1

    try:
        detections, risk, inference_ms = await run_in_threadpool(_infer, image)
    except (RuntimeError, ImportError, FileNotFoundError) as exc:
        logger.exception("Live YOLO inference unavailable")
        with STATE_LOCK:
            _STATE["last_error"] = f"{type(exc).__name__}: {exc}"
        raise HTTPException(
            status_code=503,
            detail=f"YOLO 实时推理不可用：{type(exc).__name__}: {exc}",
        ) from exc
    except Exception as exc:
        logger.exception("Unexpected live perception error")
        with STATE_LOCK:
            _STATE["last_error"] = f"{type(exc).__name__}: {exc}"
        raise HTTPException(status_code=500, detail="实时感知处理失败。") from exc

    image_buffer = io.BytesIO()
    image.save(image_buffer, format="JPEG", quality=82, optimize=True)
    jpeg = image_buffer.getvalue()

    detection_rows = [
        {
            "class_name": item.class_name,
            "confidence": round(float(item.confidence), 4),
            "box_xyxy": [round(float(v), 2) for v in item.box_xyxy],
        }
        for item in detections
    ]
    risk_data = risk.model_dump(mode="json")
    received_at = datetime.now(timezone.utc).isoformat()
    result = {
        "ok": True,
        "session_id": session_id,
        "frame_id": frame_id,
        "sim_timestamp": round(sim_timestamp, 4),  # metadata only; not a model input
        "received_at": received_at,
        "image_width": width,
        "image_height": height,
        "inference_ms": round(inference_ms, 2),
        "detection_count": len(detection_rows),
        "detections": detection_rows,
        "risk": risk_data,
        "source_isolation": {
            "camera_rgb_only_for_inference": True,
            "carla_actor_state_used": False,
            "carla_speed_or_metric_distance_used": False,
            "sim_timestamp_used_for_risk": False,
            "vehicle_control_enabled": False,
        },
        "interpretation": (
            "Live per-frame YOLO plus an uncalibrated 2D heuristic. "
            "Not a validated hazard classifier or vehicle-control system."
        ),
    }

    with STATE_LOCK:
        _STATE["active"] = True
        _STATE["last_seen_monotonic"] = time.monotonic()
        _STATE["frames_processed"] += 1
        _STATE["last_frame_id"] = frame_id
        _STATE["latest"] = result
        _STATE["latest_jpeg"] = jpeg
        _STATE["last_error"] = None

    return result


@router.get("/status", summary="获取 CARLA 实时感知状态")
def live_status():
    with STATE_LOCK:
        latest = _STATE["latest"]
        return {
            "ok": True,
            "session_id": _STATE["session_id"],
            "active": bool(_STATE["active"]),
            "started_at": _STATE["started_at"],
            "stopped_at": _STATE["stopped_at"],
            "frames_received": _STATE["frames_received"],
            "frames_processed": _STATE["frames_processed"],
            "last_frame_id": _STATE["last_frame_id"],
            "last_error": _STATE["last_error"],
            "latest_summary": (
                {
                    "frame_id": latest["frame_id"],
                    "received_at": latest["received_at"],
                    "inference_ms": latest["inference_ms"],
                    "detection_count": latest["detection_count"],
                    "risk_level": latest["risk"]["risk_level"],
                }
                if latest else None
            ),
            "dashboard_url": "/scene/live/dashboard",
        }


@router.get("/latest", summary="获取最近一帧 YOLO 与风险结果")
def live_latest():
    with STATE_LOCK:
        if _STATE["latest"] is None:
            raise HTTPException(status_code=404, detail="尚未收到实时图像帧。")
        return _STATE["latest"]


@router.get("/latest.jpg", summary="获取最近一帧 RGB 图像")
def live_latest_image():
    with STATE_LOCK:
        jpeg = _STATE["latest_jpeg"]
    if jpeg is None:
        raise HTTPException(status_code=404, detail="尚未收到实时图像帧。")
    return Response(
        content=jpeg,
        media_type="image/jpeg",
        headers={"Cache-Control": "no-store"},
    )


@router.post("/stop", summary="停止当前实时会话")
def stop_live_session(payload: dict):
    session_id = payload.get("session_id") if isinstance(payload, dict) else None
    if not isinstance(session_id, str) or not SESSION_RE.fullmatch(session_id):
        raise HTTPException(status_code=422, detail="需要有效的 session_id。")
    with STATE_LOCK:
        if _STATE["session_id"] != session_id:
            raise HTTPException(status_code=404, detail="实时会话不存在或已被替换。")
        _STATE["active"] = False
        _STATE["stopped_at"] = datetime.now(timezone.utc).isoformat()
    return {"ok": True, "session_id": session_id, "active": False}



# Research reports are OFFLINE artifacts, independent from the live camera.
# Never call VLM/Qwen from live /frame: a 6 GiB GPU cannot safely sustain them
# concurrently with the simulator and the existing YOLO stream.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESEARCH_REPORT_DIR = (PROJECT_ROOT / "evidence" / "scene_llm_risk").resolve()
MVP_REPLAY_DIR = (PROJECT_ROOT / "evidence" / "driving_agent_mvp").resolve()
RESEARCH_REPORT_PATTERN = re.compile(r"llm_frame_\d{8}_\d{8}_\d{6}_\d{6}\.json\Z")
REPLAY_PATTERN = re.compile(r"agent_mvp_\d{8}_\d{6}_\d{6}\.html\Z")
MAX_RESEARCH_JSON_BYTES = 2 * 1024 * 1024
_VALID_RISK = {"unknown", "low", "medium", "high"}


def _text_only(value: Any, maximum: int = 200) -> str:
    return value[:maximum] if isinstance(value, str) else ""


def _short_strings(value: Any, count: int = 8, maximum: int = 180) -> list[str]:
    if not isinstance(value, list):
        return []
    return [
        item[:maximum]
        for item in value[:count]
        if isinstance(item, str) and item.strip()
    ]


def _json_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        numeric = float(value)
    except (ValueError, TypeError):
        return None
    if not math.isfinite(numeric) or numeric < 0:
        return None
    return round(numeric, 2)


def _research_files() -> list[Path]:
    if not RESEARCH_REPORT_DIR.is_dir():
        return []
    candidates = []
    for candidate in RESEARCH_REPORT_DIR.glob("llm_frame_*.json"):
        if not RESEARCH_REPORT_PATTERN.fullmatch(candidate.name):
            continue
        try:
            resolved = candidate.resolve(strict=True)
            if (resolved.is_file() and resolved.is_relative_to(RESEARCH_REPORT_DIR)
                    and resolved.stat().st_size <= MAX_RESEARCH_JSON_BYTES):
                candidates.append(resolved)
        except OSError:
            continue
    return sorted(candidates, key=lambda item: (item.stat().st_mtime_ns, item.name),
                  reverse=True)


def _read_research(path: Path) -> dict[str, Any]:
    try:
        resolved = path.resolve(strict=True)
        if (not resolved.is_file() or not resolved.is_relative_to(RESEARCH_REPORT_DIR)
                or not RESEARCH_REPORT_PATTERN.fullmatch(resolved.name)
                or resolved.stat().st_size > MAX_RESEARCH_JSON_BYTES):
            raise HTTPException(status_code=404, detail="研究报告不存在或超出读取限制。")
        with resolved.open("r", encoding="utf-8-sig") as handle:
            report = json.load(handle)
    except HTTPException:
        raise
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=422, detail="研究报告无法读取。") from exc
    if (not isinstance(report, dict)
            or report.get("schema_version") != "scene_llm_risk.v1"
            or not isinstance(report.get("final_research_decision"), dict)
            or not isinstance(report.get("input_to_llm"), dict)):
        raise HTTPException(status_code=422, detail="研究报告结构不符合 scene_llm_risk.v1。")
    return report


def _risk_level(raw: Any) -> str:
    return raw if isinstance(raw, str) and raw in _VALID_RISK else "unknown"


def _summarize_research(report: dict[str, Any], name: str) -> dict[str, Any]:
    decision = report["final_research_decision"]
    model_input = report["input_to_llm"]
    candidate = report.get("candidate_model_output_UNVERIFIED")
    runtime = report.get("runtime") or {}
    sources = report.get("sources") or {}
    if not isinstance(runtime, dict):
        runtime = {}
    if not isinstance(sources, dict):
        sources = {}

    observations = []
    for obj in model_input.get("YOLO_predictions", [])[:100]:
        if not isinstance(obj, dict):
            continue
        eid = _text_only(obj.get("id"), 32)
        if not re.fullmatch(r"yolo:\d{1,3}", eid):
            continue
        observations.append({
            "id": eid,
            "class": _text_only(obj.get("class"), 60),
            "confidence": _json_number(obj.get("confidence")),
            "status": "YOLO_model_prediction_not_truth",
        })

    vlm_claims = []
    for obj in model_input.get("VLM_claims_unverified", [])[:100]:
        if not isinstance(obj, dict):
            continue
        eid = _text_only(obj.get("id"), 32)
        if not re.fullmatch(r"vlm:\d{1,3}", eid):
            continue
        value = obj.get("value")
        if isinstance(value, list):
            value = [_text_only(str(v), 60) for v in value[:5]]
        elif isinstance(value, (str, int, bool)):
            value = _text_only(value, 120) if isinstance(value, str) else value
        else:
            value = None
        vlm_claims.append({
            "id": eid,
            "type": _text_only(obj.get("type"), 80),
            "value": value,
            "status": "VLM_claim_unverified",
        })

    checks = []
    for check in model_input.get("cross_checks", [])[:20]:
        if isinstance(check, dict):
            checks.append({
                "comparison": _text_only(check.get("comparison"), 60),
                "status": _text_only(check.get("status"), 120),
            })

    frame = report.get("frame")
    frame = frame if type(frame) is int and frame >= 0 else None
    reference_ids = {item["id"] for item in observations + vlm_claims}
    if isinstance(model_input.get("ego_telemetry"), dict):
        reference_ids.add("ego:speed")
    refs = [
        ref for ref in _short_strings(decision.get("evidence_refs"), 8, 32)
        if ref in reference_ids
    ]
    return {
        "ok": True,
        "mode": "saved_offline_report",
        "matches_live_session": False,
        "vehicle_control_enabled": False,
        "report_id": name,
        "world_frame": frame,
        "created_at": _text_only(report.get("created_at"), 60),
        "validation_status": _text_only(report.get("validation_status"), 40),
        "baseline_yolo_risk_for_comparison_only": _risk_level(
            sources.get("baseline_yolo_rule_risk_for_comparison_only")
        ),
        "qwen": {
            "final_risk": _risk_level(decision.get("risk_level")),
            "candidate_risk": (
                _risk_level(candidate.get("risk_level"))
                if isinstance(candidate, dict) else None
            ),
            "key_factors": _short_strings(decision.get("key_factors"), 5),
            "evidence_refs": refs,
            "safe_suggestion": _text_only(decision.get("driving_suggestion"), 300),
            "uncertainty": _short_strings(decision.get("uncertainty"), 5),
            "generation_ms": _json_number(runtime.get("generate_ms")),
            "model_load_ms": _json_number(runtime.get("model_load_ms")),
        },
        "yolo_evidence": observations,
        "vlm_claims_unverified": vlm_claims,
        "cross_checks": checks,
        "safety_actions": _short_strings(report.get("policy_actions"), 8, 200),
        "interpretation": (
            "Saved offline single-frame Qwen research result. "
            "It is NOT a prediction for the current live RGB frame. "
            "Neither YOLO detections nor VLM text are ground truth."
        ),
    }


@router.get("/research/reports", summary="列出已有的离线 Qwen 风险报告")
def list_research_reports():
    reports = []
    for path in _research_files()[:30]:
        try:
            report = _read_research(path)
        except HTTPException:
            continue
        frame = report.get("frame")
        reports.append({
            "report_id": path.name,
            "world_frame": frame if type(frame) is int and frame >= 0 else None,
            "created_at": _text_only(report.get("created_at"), 60),
            "validation_status": _text_only(report.get("validation_status"), 40),
        })
    return JSONResponse(
        {"ok": True, "mode": "saved_offline_reports", "reports": reports},
        headers={"Cache-Control": "no-store"},
    )


@router.get("/research/report/{report_id}", summary="读取一份离线 Qwen 风险报告摘要")
def get_research_report(report_id: str):
    if not RESEARCH_REPORT_PATTERN.fullmatch(report_id):
        raise HTTPException(status_code=404, detail="研究报告文件名无效。")
    path = RESEARCH_REPORT_DIR / report_id
    report = _read_research(path)
    return JSONResponse(
        _summarize_research(report, report_id),
        headers={"Cache-Control": "no-store"},
    )


@router.get("/research/replay", summary="获取现有 RAG 多模态时序回放入口")
def research_replay():
    """Link to the ALREADY REGISTERED /scene/multimodal/{id}/player route.

    No model inference is triggered by reading this endpoint.
    """
    if not MVP_REPLAY_DIR.is_dir():
        return {"ok": True, "available": False}
    files = []
    for p in MVP_REPLAY_DIR.glob("agent_mvp_*.html"):
        if not REPLAY_PATTERN.fullmatch(p.name):
            continue
        try:
            resolved = p.resolve(strict=True)
            if resolved.is_file() and resolved.is_relative_to(MVP_REPLAY_DIR):
                files.append(resolved)
        except OSError:
            continue
    if not files:
        return {"ok": True, "available": False}
    latest = max(files, key=lambda p: (p.stat().st_mtime_ns, p.name))
    run_id = latest.stem
    # Only expose replays whose JSON exists and is served by multimodal_api.
    if not (MVP_REPLAY_DIR / (run_id + ".json")).is_file():
        return {"ok": True, "available": False}
    return {
        "ok": True,
        "available": True,
        "mode": "saved_offline_replay",
        "run_id": run_id,
        "player_url": f"/scene/multimodal/{run_id}/player",
    }


@router.get("/snapshot", summary="????????????")
def live_snapshot():
    import base64
    with STATE_LOCK:
        if _STATE["latest"] is None or _STATE["latest_jpeg"] is None:
            raise HTTPException(
                status_code=404,
                detail="??????????"
            )
        data = dict(_STATE["latest"])
        jpeg = _STATE["latest_jpeg"]

    data["image_data_url"] = (
        "data:image/jpeg;base64,"
        + base64.b64encode(jpeg).decode("ascii")
    )
    return data


@router.get("/dashboard", response_class=HTMLResponse, summary="实时驾驶感知与离线多模态研究面板")
def live_dashboard():
    return HTMLResponse(r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="color-scheme" content="light">
<title>CARLA · 智能驾驶场景理解</title>
<style>
:root{
  --paper:#f4f5f0;--card:#ffffff;--ink:#17383a;--muted:#637c7c;
  --line:#dce6e1;--teal:#13766c;--mint:#eaf4ef;--orange:#dd754c;
  --red:#b94c44;--shadow:0 9px 30px rgba(26,64,57,.055);
  font-family:"Segoe UI","Microsoft YaHei",Arial,sans-serif;
  color:var(--ink);background:var(--paper)
}
*{box-sizing:border-box}
body{margin:0;min-width:300px}
main{max-width:1340px;padding:28px 25px 48px;margin:auto}
header{display:flex;align-items:flex-end;justify-content:space-between;gap:12px;
  border-bottom:1px solid #bfcfc7;padding-bottom:20px;margin-bottom:22px}
.brand{font-size:12px;letter-spacing:.2em;font-weight:800;color:var(--teal)}
h1{font-size:clamp(23px,3.3vw,38px);letter-spacing:-.025em;margin:9px 0 7px}
h2{font-size:18px;margin:0 0 14px}h3{font-size:14px;margin:22px 0 9px}
p{line-height:1.65}small,.muted{color:var(--muted)}
.muted{font-size:13px}
.pill{font-size:11px;letter-spacing:.045em;font-weight:750;
  background:var(--mint);border:1px solid #cce5d9;border-radius:99px;
  padding:7px 11px;color:#176659;display:inline-block}
.pill.offline{background:#fff5e9;color:#956335;border-color:#e9d6bf}
.card{background:var(--card);border:1px solid var(--line);border-radius:17px;
  padding:21px;box-shadow:var(--shadow)}
.live-grid{display:grid;grid-template-columns:minmax(0,1.7fr) minmax(285px,1fr);
  gap:17px;margin-bottom:20px}
.camera{background:#102c31;border-radius:12px;overflow:hidden}
.frame-wrap{position:relative;width:100%;line-height:0}
#frame{display:block;width:100%;height:auto;min-height:180px;object-fit:contain;
  background:#102c31;color:#fff}
#bbox-layer{position:absolute;inset:0;width:100%;height:100%;pointer-events:none}
#stream{margin:11px 0 0;min-height:21px}
.kpis{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px}
.kpi{background:#edf5f0;border:1px solid #e4efea;border-radius:12px;padding:12px}
.kpi span{color:#627c7a;font-size:11px}
.kpi b{display:block;font-size:22px;margin:5px 0 0;overflow-wrap:anywhere}
.risk{font-size:31px;font-weight:850;letter-spacing:.035em}
.low{color:#198362}.medium{color:#b77624}.high{color:#ba4c46}.unknown{color:#667983}
.dets,.refs{display:flex;flex-wrap:wrap;gap:7px}
.chip{font-size:12px;padding:6px 9px;border-radius:7px;background:#edf4f0}
.chip.selected{background:#dceee7;color:#136659;border:1px solid #acd2c2}
.mono{font-family:Consolas,ui-monospace,monospace;overflow-wrap:anywhere}
.divider{height:1px;background:var(--line);margin:17px 0}
.section-label{font-size:12px;font-weight:750;letter-spacing:.09em;color:#4c7974}
.report-top{display:flex;justify-content:space-between;gap:12px;align-items:flex-start}
.report-grid{display:grid;grid-template-columns:minmax(0,1fr) minmax(270px,.78fr);gap:21px}
.keybox{background:#f3f7f4;border:1px solid var(--line);border-radius:12px;padding:15px}
.report-vals{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:9px}
.report-val{border:1px solid var(--line);border-radius:10px;padding:11px}
.report-val small{font-size:11px;display:block}
.report-val b{font-size:18px;display:block;margin-top:5px}
ol,ul{padding-left:21px;margin:7px 0;line-height:1.6}
li{margin:5px 0}
.quiet{background:#fffaf2;border-left:3px solid #d6a96c;
  padding:12px 14px;border-radius:4px;font-size:13px;line-height:1.55}
.report-toolbar{display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin:13px 0}
select,button{border:1px solid #bfcfc8;border-radius:9px;padding:9px 12px;
  background:#fff;color:var(--ink);font:inherit;font-size:13px}
select{max-width:100%;flex:1;min-width:260px}
button{cursor:pointer;background:#ecf5ef;font-weight:650}
button:hover{background:#dceddf}button:focus-visible,select:focus-visible,a:focus-visible{
  outline:2px solid #147a70;outline-offset:2px}
.details{border:1px solid var(--line);border-radius:11px;padding:12px;margin-top:12px}
.details summary{cursor:pointer;font-size:13px;font-weight:750}
.evidence-row{display:flex;gap:8px;align-items:baseline;padding:8px 0;
  border-bottom:1px dashed #e2ebe6;font-size:13px}
.evidence-row:last-child{border:0}
.evidence-row code{flex:none;color:#15756c}
.footer{margin-top:23px;border-top:1px solid #d4e1d8;padding-top:13px;font-size:12px;color:#68807c}
a{color:#086e67}
#research-error{color:#ae4b3c}
.status{font-size:12px;color:var(--muted)}
@media(max-width:840px){
  main{padding:17px 13px 30px}
  .live-grid,.report-grid{grid-template-columns:1fr}
  .report-vals{grid-template-columns:repeat(2,minmax(0,1fr))}
  header{align-items:flex-start;flex-direction:column}
  .card{padding:15px}
}
</style>
</head>
<body><main>
<header>
  <div>
    <div class="brand">CARLA / MULTIMODAL DRIVING RESEARCH</div>
    <h1>智能驾驶场景理解与风险分析</h1>
    <div class="muted">实时视觉感知 · 离线 Qwen 证据推理 · 仿真研究演示</div>
  </div>
  <span class="pill">NO VEHICLE CONTROL</span>
</header>

<div class="live-grid">
  <section class="card">
    <div class="report-top">
      <h2>实时 CARLA 摄像头</h2>
      <span class="pill">LIVE · YOLO</span>
    </div>
    <div class="camera frame-wrap">
      <img id="frame" alt="等待 CARLA RGB 画面" width="960" height="540">
      <svg id="bbox-layer" viewBox="0 0 960 540" preserveAspectRatio="xMidYMid meet"
           role="presentation" aria-hidden="true"></svg>
    </div>
    <p id="stream" class="muted">等待 CARLA RGB 客户端连接……</p>
  </section>
  <aside class="card">
    <h2>实时单帧感知</h2>
    <div class="kpis">
      <div class="kpi"><span>Session</span><b id="session">—</b></div>
      <div class="kpi"><span>World frame</span><b id="frameid">—</b></div>
      <div class="kpi"><span>YOLO latency</span><b id="latency">—</b></div>
      <div class="kpi"><span>Detections</span><b id="count">—</b></div>
    </div>
    <h3>二维启发式风险（非 Qwen）</h3>
    <div class="risk unknown" id="risk">UNKNOWN</div>
    <p class="muted" id="advice">等待图像帧。</p>
    <div class="divider"></div>
    <h3>YOLO 目标检测</h3>
    <div id="detections" class="dets"><span class="muted">暂无目标</span></div>
    <p class="muted">此区域仅使用图像像素，不读取 CARLA 物理真值。</p>
  </aside>
</div>

<section class="card" id="research">
  <div class="report-top">
    <div>
      <div class="section-label">SAVED MULTIMODAL EVIDENCE</div>
      <h2 style="margin-top:6px;margin-bottom:5px">YOLO + VLM + Qwen · 研究报告</h2>
      <div class="muted">从已保存的 <span class="mono">scene_llm_risk.v1</span> 报告加载；不会启动模型。</div>
    </div>
    <span class="pill offline">OFFLINE · 不对应实时帧</span>
  </div>
  <div class="quiet" style="margin-top:14px">
    <strong>来源隔离：</strong>下方展示的是保存的历史仿真帧分析。
    无论 Frame ID 是否碰巧相同，都不代表当前 CARLA 实时画面的风险。
    YOLO 预测和 VLM 声明也不是物理真值。
  </div>
  <div class="report-toolbar">
    <label for="report-select" class="muted">实验报告</label>
    <select id="report-select" aria-label="选择已保存的 Qwen 实验报告">
      <option value="">读取现有报告中……</option>
    </select>
    <button type="button" id="report-reload">刷新报告</button>
  </div>
  <p id="research-error" class="status" role="status">正在检查已保存的研究结果……</p>
  <div id="report-content" hidden>
    <div class="report-vals">
      <div class="report-val"><small>离线 CARLA Frame</small><b id="r-frame">—</b></div>
      <div class="report-val"><small>JSON validation</small><b id="r-valid">—</b></div>
      <div class="report-val"><small>Qwen generation</small><b id="r-time">—</b></div>
    </div>
    <div class="report-grid" style="margin-top:20px">
      <div>
        <h3>最终研究风险等级（安全规则后）</h3>
        <div class="risk unknown" id="r-risk">UNKNOWN</div>
        <p class="muted" id="r-comparison">—</p>
        <h3>关键风险因素</h3>
        <ol id="r-factors"></ol>
        <h3>本次引用的视觉证据</h3>
        <div class="refs" id="r-refs"></div>
        <div class="details">
          <details>
            <summary>查看全部 YOLO 检测与 VLM 未验证声明</summary>
            <div id="r-evidence"></div>
          </details>
        </div>
      </div>
      <div>
        <div class="keybox">
          <div class="section-label">SAFETY-FILTERED SUGGESTION</div>
          <h3>最终安全提示</h3>
          <p id="r-advice">—</p>
          <p class="muted">由安全规则约束；不生成车辆控制指令。</p>
        </div>
        <h3>不确定性</h3>
        <ul id="r-uncertainty"></ul>
        <h3>规则处理记录</h3>
        <ul id="r-guards"></ul>
      </div>
    </div>
    <div class="divider"></div>
    <p class="muted" id="r-provenance">—</p>
  </div>
  <div class="divider"></div>
  <div class="report-top">
    <div>
      <h3 style="margin:0 0 5px">RAG + VLM + Qwen 时序实验回放</h3>
      <div class="muted">独立的历史离线 Agent 回放，不会启动推理任务。</div>
    </div>
    <div id="replay-area" class="status">正在检查回放文件……</div>
  </div>
</section>

<section class="card" style="margin-top:20px">
  <h2>研究适用边界</h2>
  <ul class="muted">
    <li>实时 YOLO 规则仅基于二维图像，不具备经过校准的距离、TTC 或车辆控制能力。</li>
    <li>保存的 Qwen 报告属于独立的离线仿真帧；此页面不会逐帧运行 SmolVLM2/Qwen。</li>
    <li>VLM 交通灯颜色、车辆运动等描述未经物理真值核验；UNKNOWN 不代表安全。</li>
    <li>所有建议仅供仿真研究和人工复核，不适用于真实驾驶决策或自动驾驶控制。</li>
  </ul>
</section>
<div class="footer">CARLA RESEARCH PROTOTYPE · LIVE PERCEPTION / OFFLINE REASONING · NO ACTUATION</div>

<script>
"use strict";
const $ = id => document.getElementById(id);
let lastLiveKey = "";
let liveBusy = false;
let reportsBusy = false;
let latestReport = "";
let chosenReport = "";

function setText(id, value, fallback="—") {
  $(id).textContent = (value === null || value === undefined || value === "")
    ? fallback : String(value);
}
function makeList(id, items, empty="暂无内容") {
  const element = $(id);
  element.replaceChildren();
  if (!Array.isArray(items) || !items.length) {
    const li = document.createElement("li");
    li.textContent = empty;
    element.appendChild(li);
    return;
  }
  for (const item of items) {
    const li = document.createElement("li");
    li.textContent = String(item);
    element.appendChild(li);
  }
}
function drawBoxes(items, w, h) {
  const svg = $("bbox-layer");
  svg.setAttribute("viewBox", `0 0 ${w || 960} ${h || 540}`);
  svg.replaceChildren();
  for (const item of (items || [])) {
    const b = item.box_xyxy;
    if (!Array.isArray(b) || b.length !== 4) continue;
    const a = b.map(Number);
    if (!a.every(Number.isFinite)) continue;
    const [x1, y1, x2, y2] = a;
    if (x2 <= x1 || y2 <= y1) continue;
    const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
    rect.setAttribute("x", x1);
    rect.setAttribute("y", y1);
    rect.setAttribute("width", x2 - x1);
    rect.setAttribute("height", y2 - y1);
    rect.setAttribute("fill", "none");
    rect.setAttribute("stroke", "#ee8754");
    rect.setAttribute("stroke-width", "3");
    rect.setAttribute("vector-effect", "non-scaling-stroke");
    svg.appendChild(rect);
  }
}
async function refreshLive() {
  if (liveBusy) return;
  liveBusy = true;
  try {
    const response = await fetch("/scene/live/status", {cache:"no-store"});
    if (!response.ok) throw new Error("status HTTP " + response.status);
    const status = await response.json();
    const active = Boolean(status.active);
    setText("stream", active ? "实时 RGB/YOLO 正在运行" :
      (status.latest_summary ? "实时会话已停止 · 保留最后接收的画面" :
       "等待 CARLA RGB 客户端连接……"));
    setText("session", status.session_id ? status.session_id.slice(0,10) : null);
    const latest = status.latest_summary;
    if (!latest) return;
    const key = String(status.session_id) + ":" + String(latest.frame_id);
    if (key === lastLiveKey) return;
    const snapRes = await fetch("/scene/live/snapshot", {cache:"no-store"});
    if (!snapRes.ok) throw new Error("snapshot HTTP " + snapRes.status);
    const data = await snapRes.json();
    const preview = new Image();
    preview.src = data.image_data_url;
    await preview.decode();
    $("frame").src = preview.src;
    setText("frameid", data.frame_id);
    setText("latency", data.inference_ms + " ms");
    setText("count", data.detection_count);
    drawBoxes(data.detections, data.image_width, data.image_height);
    const risk = data.risk || {};
    const level = ["low","medium","high","unknown"].includes(risk.risk_level)
      ? risk.risk_level : "unknown";
    setText("risk", level.toUpperCase());
    $("risk").className = "risk " + level;
    setText("advice", risk.driving_suggestion);
    const dets = $("detections");
    dets.replaceChildren();
    for (const obj of (data.detections || [])) {
      const chip = document.createElement("span");
      chip.className = "chip";
      chip.textContent = obj.class_name + " " +
        Math.round(Number(obj.confidence) * 100) + "%";
      dets.appendChild(chip);
    }
    if (!(data.detections || []).length) {
      const node = document.createElement("span");
      node.className = "muted";
      node.textContent = "本帧无检测目标（不代表安全）";
      dets.appendChild(node);
    }
    lastLiveKey = key;
  } catch (error) {
    setText("stream", "实时 API 暂不可用：" + String(error));
  } finally {
    liveBusy = false;
  }
}
function showReport(report) {
  $("report-content").hidden = false;
  setText("research-error", "");
  const q = report.qwen || {};
  setText("r-frame", report.world_frame);
  setText("r-valid", report.validation_status);
  setText("r-time", q.generation_ms == null ? null : q.generation_ms + " ms");
  const level = ["low","medium","high","unknown"].includes(q.final_risk)
    ? q.final_risk : "unknown";
  setText("r-risk", level.toUpperCase());
  $("r-risk").className = "risk " + level;
  setText("r-comparison", "Qwen 候选：" +
    (q.candidate_risk || "无有效候选") +
    " · YOLO 原规则：" + report.baseline_yolo_risk_for_comparison_only +
    "（仅供同帧离线比较）");
  makeList("r-factors", q.key_factors, "模型未提供关键因素");
  makeList("r-uncertainty", q.uncertainty, "未提供不确定性描述");
  makeList("r-guards", report.safety_actions, "无附加规则记录");
  setText("r-advice", q.safe_suggestion, "仅供仿真研究人员复核。");

  const refs = new Set(q.evidence_refs || []);
  const refArea = $("r-refs");
  refArea.replaceChildren();
  for (const id of refs) {
    const span = document.createElement("span");
    span.className = "chip selected mono";
    span.textContent = id;
    refArea.appendChild(span);
  }
  if (!refs.size) {
    const node = document.createElement("span");
    node.className = "muted";
    node.textContent = "本次未引用具体视觉证据";
    refArea.appendChild(node);
  }

  const evidenceArea = $("r-evidence");
  evidenceArea.replaceChildren();
  const addEvidence = (id, desc) => {
    const row = document.createElement("div");
    row.className = "evidence-row";
    const tag = document.createElement("code");
    tag.textContent = id;
    const body = document.createElement("span");
    body.textContent = desc + (refs.has(id) ? " · 已引用" : "");
    row.append(tag, body);
    evidenceArea.appendChild(row);
  };
  for (const obj of report.yolo_evidence || []) {
    const conf = obj.confidence == null ? "未知" :
      Math.round(Number(obj.confidence) * 100) + "%";
    addEvidence(obj.id, "YOLO 预测 · " + obj.class + " · 置信度 " + conf);
  }
  for (const obj of report.vlm_claims_unverified || []) {
    const value = Array.isArray(obj.value) ? obj.value.join("、") : String(obj.value);
    addEvidence(obj.id, "VLM 未核验声明 · " + obj.type + "：" + value);
  }
  if (!evidenceArea.childElementCount) {
    evidenceArea.textContent = "本份报告未包含可展示的视觉证据。";
  }
  setText("r-provenance", "来源：" + report.report_id +
    " · 创建于 " + report.created_at +
    " · 仅历史仿真样本，不关联实时会话");
}
async function loadReport(reportId) {
  if (!reportId) return;
  setText("research-error", "正在读取离线 Qwen 报告……");
  try {
    const response = await fetch(
      "/scene/live/research/report/" + encodeURIComponent(reportId),
      {cache:"no-store"});
    if (!response.ok) throw new Error("报告 HTTP " + response.status);
    const report = await response.json();
    if (chosenReport !== reportId) return;
    showReport(report);
  } catch (error) {
    $("report-content").hidden = true;
    setText("research-error", "研究报告暂不可用：" + String(error));
  }
}
async function loadReportIndex() {
  if (reportsBusy) return;
  reportsBusy = true;
  try {
    const response = await fetch("/scene/live/research/reports", {cache:"no-store"});
    if (!response.ok) throw new Error("目录 HTTP " + response.status);
    const data = await response.json();
    const files = Array.isArray(data.reports) ? data.reports : [];
    const select = $("report-select");
    const prev = chosenReport;
    select.replaceChildren();
    for (const item of files) {
      const option = document.createElement("option");
      option.value = item.report_id;
      option.textContent = "Frame " + item.world_frame + " · " +
        item.created_at + " · " + item.validation_status;
      select.appendChild(option);
    }
    if (!files.length) {
      const option = document.createElement("option");
      option.value = "";
      option.textContent = "暂无已保存的 scene_llm_risk 报告";
      select.appendChild(option);
      chosenReport = "";
      latestReport = "";
      $("report-content").hidden = true;
      setText("research-error", "尚未找到历史 Qwen 报告；无需启动模型即可保持实时 YOLO 展示。");
      return;
    }
    latestReport = files[0].report_id;
    const exists = files.some(item => item.report_id === prev);
    chosenReport = exists ? prev : latestReport;
    select.value = chosenReport;
    if (chosenReport !== prev || $("report-content").hidden) await loadReport(chosenReport);
  } catch (error) {
    setText("research-error", "离线报告目录暂不可用：" + String(error));
  } finally {
    reportsBusy = false;
  }
}
async function loadReplay() {
  const area = $("replay-area");
  try {
    const response = await fetch("/scene/live/research/replay", {cache:"no-store"});
    if (!response.ok) throw new Error("HTTP " + response.status);
    const data = await response.json();
    area.replaceChildren();
    if (!data.available || !data.player_url) {
      area.textContent = "暂无已保存的时序回放";
      return;
    }
    const link = document.createElement("a");
    link.href = data.player_url;
    link.target = "_blank";
    link.rel = "noopener noreferrer";
    link.textContent = "打开现有离线 Agent 回放";
    area.appendChild(link);
  } catch (error) {
    area.textContent = "回放入口暂不可用";
  }
}
$("report-select").addEventListener("change", async event => {
  chosenReport = event.target.value;
  $("report-content").hidden = true;
  await loadReport(chosenReport);
});
$("report-reload").addEventListener("click", async () => {
  chosenReport = "";
  $("report-content").hidden = true;
  await loadReportIndex();
  await loadReplay();
});
refreshLive();
loadReportIndex();
loadReplay();
setInterval(refreshLive, 1000);
setInterval(loadReportIndex, 15000);
</script>
</main></body></html>""")
