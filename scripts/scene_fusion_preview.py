"""
Phase 3.2: Offline evidence alignment for ONE existing CARLA frame.

Read-only inputs: CARLA metadata, YOLO risk report, and VLM single-image report.
CARLA ground truth is for post-hoc human AUDIT ONLY; it is not passed to a model.
No model inference, new risk score, or driving action.
"""
import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))

def frame_matches(record, frame):
    if not isinstance(record, dict):
        return False
    for key in ("world_frame", "image_frame", "frame", "frame_id"):
        value = record.get(key)
        if not isinstance(value, bool):
            try:
                if value is not None and int(value) == frame:
                    return True
            except (TypeError, ValueError):
                pass
    for key in ("image_file", "image_path"):
        m = re.search(r"frame_(\d+)\.(png|jpe?g)$",
                      str(record.get(key, "")), re.I)
        if m and int(m.group(1)) == frame:
            return True
    return False

def find_records(data, frame, depth=0):
    if depth > 12:
        return
    if isinstance(data, dict):
        if frame_matches(data, frame):
            yield data
        for key, value in data.items():
            if str(key).isdigit() and int(key) == frame and isinstance(value, dict):
                yield value
            if isinstance(value, (dict, list)):
                yield from find_records(value, frame, depth + 1)
    elif isinstance(data, list):
        for value in data:
            if isinstance(value, (dict, list)):
                yield from find_records(value, frame, depth + 1)

def nested(data, keys, depth=0):
    if depth > 8:
        return
    if isinstance(data, dict):
        for key in keys:
            if key in data:
                yield data[key]
        for value in data.values():
            if isinstance(value, (dict, list)):
                yield from nested(value, keys, depth + 1)
    elif isinstance(data, list):
        for value in data:
            if isinstance(value, (dict, list)):
                yield from nested(value, keys, depth + 1)

def original_risk(data):
    levels = {"unknown", "low", "medium", "high"}
    for value in nested(data, ("risk_level", "risk", "overall_risk", "level")):
        if isinstance(value, str) and value.lower() in levels:
            return value.lower()
        if isinstance(value, dict):
            v = value.get("risk_level", value.get("level"))
            if isinstance(v, str) and v.lower() in levels:
                return v.lower()
    return None

def detection_info(data):
    for value in nested(data, ("detections",)):
        if isinstance(value, list) and all(isinstance(o, dict) for o in value):
            items = []
            for obj in value[:20]:
                items.append({
                    "class": next((obj[k] for k in
                                   ("class_name", "label", "name", "cls", "class_id")
                                   if k in obj), None),
                    "confidence": obj.get("confidence", obj.get("conf")),
                    "bbox": obj.get("xyxy", obj.get("bbox"))
                })
            return {"count": len(value), "objects": items}
    for value in nested(data, ("detection_count", "detections_count", "num_detections")):
        if isinstance(value, int) and not isinstance(value, bool):
            return {"count": value, "objects": None}
    return {"count": None, "objects": None}

def latest_yolo(capture_dir):
    items = sorted(capture_dir.glob("yolo_risk_*.json"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    if not items:
        raise ValueError("YOLO report not found in capture directory.")
    return items[0]

def norm(text):
    return str(text).replace("\\", "/").casefold()

def matching_vlm(root, image, explicit):
    folder = root / "evidence" / "vlm_smolvlm2"
    files = [explicit] if explicit else sorted(
        folder.glob("vlm_single_image_*.json"),
        key=lambda p: p.stat().st_mtime, reverse=True)
    for f in files:
        content = load(f)
        actual = norm(content.get("image_path", "")).split("/")
        expected = norm(image).split("/")
        if len(actual) >= 3 and actual[-3:] == expected[-3:]:
            return f, content
    raise ValueError("No VLM report for this exact capture run and image. "
                     "Use --vlm-report if needed.")

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--world-frame", type=int, required=True)
    parser.add_argument("--yolo-report", type=Path)
    parser.add_argument("--vlm-report", type=Path)
    a = parser.parse_args()
    capture = a.capture_dir.resolve()
    meta_path = capture / "metadata.json"
    meta = load(meta_path)
    frames = meta.get("frames") if isinstance(meta, dict) else None
    if not isinstance(frames, list):
        raise ValueError("Unexpected metadata format: missing frames list.")
    found = [f for f in frames if frame_matches(f, a.world_frame)]
    if len(found) != 1:
        raise ValueError(f"Expected one metadata frame {a.world_frame}; found {len(found)}.")
    truth_frame = found[0]
    image_name = truth_frame.get("image_file")
    if not isinstance(image_name, str):
        raise ValueError("Metadata frame lacks image_file.")
    image = (capture / image_name).resolve()
    if not image.is_file():
        raise ValueError(f"Image missing: {image}")
    yolo_path = a.yolo_report or latest_yolo(capture)
    yolo_matches = list(find_records(load(yolo_path), a.world_frame))
    if not yolo_matches:
        raise ValueError("Could not locate matching frame in YOLO JSON. "
                         "Please send its top-level keys so we can adapt safely.")
    yolo = max(yolo_matches, key=lambda d: len(json.dumps(d, ensure_ascii=False)))
    root = Path(__file__).resolve().parent.parent
    vlm_path, vlm = matching_vlm(root, image, a.vlm_report)
    description = vlm.get("output_text")
    if not isinstance(description, str) or not description.strip():
        raise ValueError("VLM report contains no output_text.")
    audit_keys = (
        "walker_lane_center_offset_m",
        "walker_inside_lane_lateral_band",
        "walker_relative_longitudinal_m",
        "walker_relative_lateral_m",
        "walker_planar_origin_distance_m",
    )
    audit = {k: truth_frame[k] for k in audit_keys if k in truth_frame}
    result = {
        "type": "offline_single_frame_evidence_alignment",
        "world_frame": a.world_frame,
        "image": str(image),
        "sources": {
            "metadata": str(meta_path), "yolo": str(yolo_path), "vlm": str(vlm_path)
        },
        "yolo_original_rule_risk": original_risk(yolo),
        "yolo_detection_info": detection_info(yolo),
        "yolo_raw_frame_record": yolo,
        "vlm_unverified_description": description,
        "vlm_generation_ms": vlm.get("generation_ms"),
        "carla_truth_for_human_audit_only": audit,
        "claim_status": "VLM details not visually verified",
        "limitations": [
            "This does not create a new risk decision.",
            "YOLO and VLM predictions are not ground truth.",
            "CARLA metadata has NOT been supplied to the models.",
            "Single-frame motion and light-color claims require verification."
        ]
    }
    out_dir = root / "evidence" / "scene_fusion_preview"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"aligned_frame_{a.world_frame:08d}_{datetime.now():%Y%m%d_%H%M%S}.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("=== Scene evidence alignment ===")
    print("Frame:", a.world_frame)
    print("Original YOLO risk:", result["yolo_original_rule_risk"])
    print("YOLO detection count:", result["yolo_detection_info"]["count"])
    print("YOLO detections:", json.dumps(result["yolo_detection_info"]["objects"],
                                          ensure_ascii=False))
    print("CARLA walker lane-center offset (AUDIT only):",
          audit.get("walker_lane_center_offset_m", "not available"))
    print("CARLA walker inside lane (AUDIT only):",
          audit.get("walker_inside_lane_lateral_band", "not available"))
    print("VLM unverified:", description)
    print("Report:", out)
    print("PASS: existing CARLA/YOLO/VLM evidence aligned; no new inference")
    print("IMPORTANT: visually verify VLM claims using the saved PNG.")

if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(1)
