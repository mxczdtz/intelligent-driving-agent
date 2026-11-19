# -*- coding: utf-8 -*-
"""
Live YOLO -> VLM -> fusion -> Qwen bridge (CARLA research only).

Runs separately from the existing FastAPI process. Polls its published live
endpoints, snapshots a *matched* YOLO JSON + JPEG keyframe, then launches VLM
and Qwen SEQUENTIALLY as subprocesses. No model is called inside /frame.
Nothing here issues vehicle-control commands.

Default CPU inference protects a 6 GiB laptop GPU while CARLA/YOLO are live.
CPU inference may be slow. CUDA is opt-in and may fail if VRAM is insufficient.

Run from the repository root:
    python -m scripts.live_multimodal_bridge --qwen-model-dir PATH_TO_LOCAL_QWEN_MODEL --once
For continuous operation, omit --once. For collection without inference:
    python -m scripts.live_multimodal_bridge --capture-only
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from html import escape
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen
from uuid import uuid4

PROJECT = Path(__file__).resolve().parent.parent
EVIDENCE = PROJECT / "evidence"
BRIDGE_ROOT = EVIDENCE / "live_multimodal_bridge"
LIVE_DEFAULT = "http://127.0.0.1:8000"
SESSION = re.compile(r"^[A-Za-z0-9_-]{8,64}$")
IMPORTANT_CLASSES = {"person", "car", "truck", "bus", "motorcycle",
                     "bicycle", "traffic light", "stop sign"}
ALLOWED_RISKS = {"unknown", "low", "medium", "high"}
MAX_JPEG = 5 * 1024 * 1024


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object: " + str(path))
    return value


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    with path.open("x", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def replace_json(path: Path, data: dict[str, Any]) -> None:
    """Atomically replace a local display/status file, never a source report."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.parent / (path.name + "." + uuid4().hex + ".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp, path)


def http_get(url: str, *, limit: int) -> bytes:
    req = Request(url, headers={"Cache-Control": "no-store", "User-Agent": "LocalCARLABridge/1"})
    with urlopen(req, timeout=3.5) as response:
        payload = response.read(limit + 1)
    if len(payload) > limit:
        raise ValueError("Unexpectedly large live response")
    return payload


def get_latest(base: str) -> dict[str, Any]:
    raw = http_get(base + "/scene/live/latest", limit=300_000)
    data = json.loads(raw.decode("utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Malformed live JSON")
    return data


def live_key(data: dict[str, Any]) -> tuple[str, int]:
    session = data.get("session_id")
    frame_id = data.get("frame_id")
    if not isinstance(session, str) or not SESSION.fullmatch(session):
        raise ValueError("Unexpected live session_id")
    if not isinstance(frame_id, int) or isinstance(frame_id, bool) or frame_id < 0:
        raise ValueError("Unexpected live frame_id")
    return session, frame_id


def matched_snapshot(base: str) -> tuple[dict[str, Any], bytes] | None:
    """JSON(A) -> JPEG -> JSON(B); accept only if both JSON keys agree.

    CARLA frame IDs strictly increase within a session. A mismatch indicates
    the camera advanced while fetching its JPEG, so the snapshot is rejected.
    """
    before = get_latest(base)
    key_before = live_key(before)
    raw = http_get(base + "/scene/live/latest.jpg", limit=MAX_JPEG)
    after = get_latest(base)
    if key_before != live_key(after):
        return None
    if not (raw.startswith(b"\xff\xd8\xff") and raw.rstrip().endswith(b"\xff\xd9")):
        raise ValueError("Latest image is not a complete JPEG")
    if not isinstance(after.get("detections"), list) or not isinstance(after.get("risk"), dict):
        raise ValueError("Live response missing YOLO detections or risk")
    return after, raw


def live_frame_fresh(data: dict[str, Any], max_age_sec: float = 8.0) -> bool:
    """Do not automatically analyze a stale /latest frame after CARLA stopped."""
    received_at = data.get("received_at")
    if not isinstance(received_at, str):
        return False
    try:
        parsed = datetime.fromisoformat(received_at.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            return False
        age = (datetime.now(timezone.utc) - parsed).total_seconds()
        return -2.0 <= age <= max_age_sec
    except ValueError:
        return False


def interesting(data: dict[str, Any], include_empty: bool) -> bool:
    if include_empty:
        return True
    return any(
        isinstance(row, dict)
        and str(row.get("class_name", "")).strip().lower() in IMPORTANT_CLASSES
        for row in data.get("detections", [])
    )


def capture(data: dict[str, Any], jpeg: bytes) -> Path:
    session, frame = live_key(data)
    folder = BRIDGE_ROOT / "sessions" / session
    folder.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    stem = f"frame_{frame:08d}_{stamp}"
    image_path = folder / (stem + ".jpg")
    metadata_path = folder / (stem + ".json")
    image_path.write_bytes(jpeg)
    record = {
        "schema_version": "live_multimodal_capture.v1",
        "captured_at": utc_now(),
        "session_id": session,
        "frame_id": frame,
        "image_path": str(image_path.resolve()),
        "image_width": data.get("image_width"),
        "image_height": data.get("image_height"),
        "detections": data["detections"],
        "risk": data["risk"],
        "yolo_inference_ms": data.get("inference_ms"),
        "source_isolation": {
            "simulator_truth_used_as_visual_evidence": False,
            "model_vehicle_control": False,
        },
    }
    write_json(metadata_path, record)
    return metadata_path


def run_module(module: str, args: list[str], *, timeout: int, label: str) -> None:
    command = [sys.executable, "-m", module, *args]
    print(f"[{label}] 子进程启动（独立于实时 YOLO 服务）", flush=True)
    try:
        completed = subprocess.run(
            command, cwd=str(PROJECT), stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, errors="replace",
            timeout=timeout, check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"{label} exceeded {timeout} seconds") from exc
    if completed.returncode:
        suffix = (completed.stdout or "")[-1600:]
        raise RuntimeError(f"{label} failed (exit {completed.returncode}): {suffix}")
    print(f"[{label}] 已完成", flush=True)


def newly_written_reports(folder: Path, pattern: str) -> set[str]:
    return {p.name for p in folder.glob(pattern)}


def find_vlm_report(before: set[str], target_image: Path) -> Path:
    folder = EVIDENCE / "vlm_smolvlm2"
    matches = []
    for path in folder.glob("vlm_single_image_*.json"):
        if path.name in before:
            continue
        try:
            report = read_json(path)
            image = Path(str(report.get("image_path", ""))).resolve()
            description = report.get("output_text")
            if image == target_image.resolve() and isinstance(description, str) and description.strip():
                matches.append(path)
        except (OSError, ValueError, json.JSONDecodeError):
            pass
    if len(matches) != 1:
        raise RuntimeError(f"Expected one new VLM report for same JPEG, found {len(matches)}")
    return matches[0]


def prepare_fusion(capture_record: dict[str, Any], vlm_report: Path) -> Path:
    """Reuse the project's real scene_fusion.fuse_visual_evidence contract."""
    from app.scene_fusion import fuse_visual_evidence

    descriptions = read_json(vlm_report)
    text = descriptions.get("output_text")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("VLM report contains no usable text")
    classes = []
    for row in capture_record["detections"]:
        if not isinstance(row, dict):
            raise ValueError("Invalid live detection")
        name = row.get("class_name")
        score = row.get("confidence")
        bbox = row.get("box_xyxy")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Live detection has no class_name")
        classes.append({"class": name, "confidence": score, "bbox": bbox})
    risk = capture_record["risk"].get("risk_level", "unknown")
    if risk not in ALLOWED_RISKS:
        raise ValueError("Invalid original YOLO risk level")

    aligned = {
        "world_frame": capture_record["frame_id"],
        "image": capture_record["image_path"],
        "sources": {
            "yolo": {"mode": "live_rgb", "session_id": capture_record["session_id"],
                     "frame_id": capture_record["frame_id"]},
            "vlm": {"report": str(vlm_report.resolve()), "same_rgb_snapshot": True},
        },
        "yolo_detection_info": {"count": len(classes), "objects": classes},
        "vlm_unverified_description": text,
        "yolo_original_rule_risk": risk,
        # Important: CARLA metadata/truth is not supplied at all.
    }
    fusion = fuse_visual_evidence(aligned)
    fusion["live_keyframe_origin"] = {
        "session_id": capture_record["session_id"],
        "frame_id": capture_record["frame_id"],
        "captured_at": capture_record["captured_at"],
        "image_path": capture_record["image_path"],
        "analysis_mode": "asynchronous_saved_keyframe",
        "not_current_live_frame": True,
    }
    path = EVIDENCE / "scene_fusion" / (
        f"fusion_frame_{capture_record['frame_id']:08d}_"
        f"{datetime.now():%Y%m%d_%H%M%S_%f}.json"
    )
    write_json(path, fusion)
    return path


def find_qwen_report(before: set[str], frame: int, fusion: Path) -> Path:
    folder = EVIDENCE / "scene_llm_risk"
    matches = []
    for path in folder.glob(f"llm_frame_{frame:08d}_*.json"):
        if path.name in before:
            continue
        try:
            value = read_json(path)
            source = value.get("sources", {}).get("fusion_report", "")
            if Path(str(source)).resolve() == fusion.resolve():
                matches.append(path)
        except (OSError, ValueError, json.JSONDecodeError, TypeError, AttributeError):
            continue
    if len(matches) != 1:
        raise RuntimeError(f"Expected one new Qwen report from the same fusion, found {len(matches)}")
    return matches[0]


def process_capture(capture_file: Path, *, qwen_model: Path,
                    device: str, vlm_timeout: int, qwen_timeout: int) -> dict[str, Any]:
    record = read_json(capture_file)
    frame = record["frame_id"]
    session = record["session_id"]
    image_path = Path(record["image_path"]).resolve(strict=True)
    result = {
        "schema_version": "live_multimodal_link.v1",
        "session_id": session, "frame_id": frame,
        "captured_at": record["captured_at"],
        "analyzed_at": None,
        "image_path": str(image_path),
        "yolo_detections": record["detections"],
        "yolo_rule_risk": record["risk"].get("risk_level", "unknown"),
        "vlm_unverified_text": None,
        "qwen_risk": "unknown",
        "qwen_validation": None,
        "evidence_refs": [],
        "key_factors": [],
        "uncertainty": [],
        "status": "processing",
        "fusion_report": None, "vlm_report": None, "qwen_report": None,
        "limitations": "Asynchronous research on a saved live keyframe; no vehicle control.",
    }
    try:
        vlm_folder = EVIDENCE / "vlm_smolvlm2"
        existing_vlm = newly_written_reports(vlm_folder, "vlm_single_image_*.json")
        run_module("scripts.vlm_batch_compare",
                   ["--images", str(image_path), "--device", device,
                    "--max-new-tokens", "96"], timeout=vlm_timeout, label="VLM")
        vlm_report = find_vlm_report(existing_vlm, image_path)
        result["vlm_report"] = str(vlm_report)
        result["vlm_unverified_text"] = read_json(vlm_report)["output_text"]
        fusion_file = prepare_fusion(record, vlm_report)
        result["fusion_report"] = str(fusion_file)

        existing_qwen = newly_written_reports(
            EVIDENCE / "scene_llm_risk", f"llm_frame_{frame:08d}_*.json")
        run_module("app.scene_llm_risk",
                   ["--fusion-report", str(fusion_file), "--model-dir", str(qwen_model),
                    "--device", device], timeout=qwen_timeout, label="Qwen")
        qwen_file = find_qwen_report(existing_qwen, frame, fusion_file)
        result["qwen_report"] = str(qwen_file)
        qwen = read_json(qwen_file)
        final = qwen.get("final_research_decision", {})
        result["qwen_risk"] = final.get("risk_level", "unknown")
        result["qwen_validation"] = qwen.get("validation_status")
        result["key_factors"] = final.get("key_factors", [])
        result["evidence_refs"] = final.get("evidence_refs", [])
        result["uncertainty"] = final.get("uncertainty", [])
        result["status"] = (
            "complete" if result["qwen_validation"] in ("valid", "valid_after_retry")
            else "conservative_fallback"
        )
    except Exception as exc:
        result["status"] = "failed"
        result["error"] = f"{type(exc).__name__}: {str(exc)[:650]}"
    result["analyzed_at"] = utc_now()
    result_path = capture_file.with_name(capture_file.stem + "_result.json")
    write_json(result_path, result)
    result["result_path"] = str(result_path)
    return result


def render_monitor(history: list[dict[str, Any]]) -> None:
    def cell(value: Any) -> str:
        return escape(str(value)) if value is not None else "—"

    rows = []
    for data in reversed(history[-12:]):
        src = Path(str(data.get("image_path", "")))
        try:
            rel = src.relative_to(BRIDGE_ROOT).as_posix()
        except ValueError:
            rel = ""
        image = f'<img src="{escape(rel, quote=True)}" alt="Saved CARLA keyframe">' if rel else ""
        detected = ", ".join(
            f"{v.get('class_name','?')} ({v.get('confidence',0):.2f})"
            for v in data.get("yolo_detections", []) if isinstance(v, dict)
        )
        factors = "；".join(data.get("key_factors", [])[:3])
        note = cell(data.get("vlm_unverified_text") or "VLM 未完成")
        rows.append(
            "<article><div class='meta'>"
            f"Session {cell(data.get('session_id'))} / Frame {cell(data.get('frame_id'))}"
            f"　·　{cell(data.get('status'))}　·　{cell(data.get('analyzed_at'))}"
            "</div><div class='content'><div>" + image + "</div><div>"
            f"<p><b>YOLO：</b>{cell(detected or '无目标')}</p>"
            f"<p><b>二维规则：</b>{cell(data.get('yolo_rule_risk'))}</p>"
            f"<p><b>VLM（未经核实）：</b>{note}</p>"
            f"<p><b>Qwen 分析：</b><strong>{cell(data.get('qwen_risk'))}</strong>"
            f"　校验：{cell(data.get('qwen_validation'))}</p>"
            f"<p><b>证据引用：</b>{cell(', '.join(data.get('evidence_refs', [])))}</p>"
            f"<p><b>因素：</b>{cell(factors)}</p>"
            f"<p><b>状态：</b>{cell(data.get('error','已完成或等待处理'))}</p>"
            "</div></div></article>"
        )
    html = """<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta http-equiv="refresh" content="8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>CARLA Multimodal Keyframe Research</title><style>
body{background:#f8f5ef;color:#173c45;font:15px/1.55 system-ui,sans-serif;max-width:1120px;margin:auto;padding:32px}
h1{font-size:2rem;margin:0 0 6px}aside{color:#586c70;margin:0 0 24px}
article{background:white;border:1px solid #d9e3e1;border-radius:14px;margin:16px 0;overflow:hidden}
.meta{background:#e6f0ed;padding:12px 18px;font-weight:700}
.content{display:grid;grid-template-columns:minmax(240px,42%) 1fr;gap:20px;padding:18px}
img{width:100%;border-radius:8px}p{margin:0 0 10px;overflow-wrap:anywhere}
strong{color:#bb592d}footer{margin-top:30px;color:#60767b;font-size:13px}
@media(max-width:650px){.content{grid-template-columns:1fr}body{padding:16px}}
</style></head><body><h1>CARLA · YOLO → VLM → Qwen</h1>
<aside>自动关联同一关键帧的研究结果 · 每 8 秒刷新一次 · 非逐帧实时驾驶控制</aside>
""" + ("\n".join(rows) if rows else "<p>等待关键帧采集…</p>") + """
<footer>CARLA 物理真值不进入视觉证据；VLM 声明未经独立验证；UNKNOWN 不等于安全。
仅供本地仿真研究与人工复核。</footer></body></html>"""
    path = BRIDGE_ROOT / "monitor.html"
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name("monitor." + uuid4().hex + ".html.tmp")
    temp.write_text(html, encoding="utf-8")
    os.replace(temp, path)


def update_history(history: list[dict[str, Any]], entry: dict[str, Any]) -> None:
    history.append(entry)
    history[:] = history[-30:]
    replace_json(BRIDGE_ROOT / "status.json", {
        "updated_at": utc_now(), "latest": entry, "recent_events": history,
    })
    render_monitor(history)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=LIVE_DEFAULT)
    parser.add_argument("--qwen-model-dir", type=Path,
                        default=Path(os.environ["QWEN_MODEL_DIR"])
                        if os.environ.get("QWEN_MODEL_DIR") else None)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--poll-sec", type=float, default=1.0)
    parser.add_argument("--min-gap-sec", type=float, default=25.0)
    parser.add_argument("--vlm-timeout", type=int, default=600)
    parser.add_argument("--qwen-timeout", type=int, default=600)
    parser.add_argument("--include-empty", action="store_true")
    parser.add_argument("--capture-only", action="store_true")
    parser.add_argument("--process-record", type=Path, default=None,
                        help="Analyze a previously captured JSON after CARLA is stopped.")
    parser.add_argument("--once", action="store_true",
                        help="Process just one eligible keyframe, then stop.")
    args = parser.parse_args()
    parsed = urlsplit(args.base_url)
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        parser.error("Only local loopback HTTP base URLs are allowed.")
    if not 0.3 <= args.poll_sec <= 60 or not 0 <= args.min_gap_sec <= 3600:
        parser.error("poll-sec/min-gap-sec out of range.")
    if not args.capture_only and (
        args.qwen_model_dir is None or not args.qwen_model_dir.expanduser().is_dir()
    ):
        parser.error("Set --qwen-model-dir to an existing local model directory.")
    qwen_model = (args.qwen_model_dir.expanduser().resolve()
                  if args.qwen_model_dir else Path("."))
    history: list[dict[str, Any]] = []
    if args.process_record is not None:
        result = process_capture(args.process_record.resolve(strict=True),
                                 qwen_model=qwen_model, device=args.device,
                                 vlm_timeout=args.vlm_timeout, qwen_timeout=args.qwen_timeout)
        update_history(history, result)
        print("结果：", result["status"], "|", result.get("result_path"))
        return 0 if result["status"] in ("complete", "conservative_fallback") else 1

    print("实时摄像头继续运行 YOLO；本进程按关键帧依次执行 VLM → 融合 → Qwen。")
    print("设备：", args.device, "| 监视页面：", BRIDGE_ROOT / "monitor.html")
    render_monitor(history)
    last_key = None
    last_capture_at = float("-inf")
    errors_seen = set()
    while True:
        try:
            live = get_latest(args.base_url)
            key = live_key(live)
            if not live_frame_fresh(live):
                time.sleep(args.poll_sec)
                continue
            if key == last_key or not interesting(live, args.include_empty):
                time.sleep(args.poll_sec)
                continue
            if time.monotonic() - last_capture_at < args.min_gap_sec:
                time.sleep(args.poll_sec)
                continue
            aligned = matched_snapshot(args.base_url)
            if aligned is None:
                time.sleep(args.poll_sec)
                continue
            snapshot_data, jpeg = aligned
            if not live_frame_fresh(snapshot_data):
                time.sleep(args.poll_sec)
                continue
            if not interesting(snapshot_data, args.include_empty):
                time.sleep(args.poll_sec)
                continue
            snapshot_key = live_key(snapshot_data)
            if snapshot_key == last_key:
                time.sleep(args.poll_sec)
                continue
            capture_path = capture(snapshot_data, jpeg)
            last_key = snapshot_key
            last_capture_at = time.monotonic()
            print("已保存同帧关键帧：", snapshot_key, flush=True)
            if args.capture_only:
                entry = {
                    "session_id": snapshot_key[0], "frame_id": snapshot_key[1],
                    "image_path": read_json(capture_path)["image_path"],
                    "yolo_detections": snapshot_data["detections"],
                    "yolo_rule_risk": snapshot_data["risk"].get("risk_level"),
                    "status": "captured_only", "analyzed_at": utc_now(),
                    "capture_file": str(capture_path),
                }
            else:
                entry = process_capture(capture_path, qwen_model=qwen_model,
                                        device=args.device,
                                        vlm_timeout=args.vlm_timeout,
                                        qwen_timeout=args.qwen_timeout)
            update_history(history, entry)
            print("处理结果：", entry["status"], "Session/Frame:", snapshot_key, flush=True)
            if entry.get("error"):
                print("原因：", entry["error"], flush=True)
            if args.once:
                return 0 if entry["status"] in (
                    "complete", "conservative_fallback", "captured_only") else 1
        except KeyboardInterrupt:
            print("\n已停止采集，实时 FastAPI 不受影响。")
            return 0
        except (HTTPError, URLError, TimeoutError, ConnectionError,
                OSError, json.JSONDecodeError, ValueError) as exc:
            msg = f"{type(exc).__name__}: {str(exc)[:150]}"
            if msg not in errors_seen:
                print("等待可用实时帧：", msg, flush=True)
                errors_seen.add(msg)
            time.sleep(max(2.0, args.poll_sec))


if __name__ == "__main__":
    raise SystemExit(main())
