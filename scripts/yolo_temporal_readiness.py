r"""
Phase 4.2 — YOLO box-schema and vision-only temporal readiness audit.

Inputs (read-only):
- ORIGINAL yolo_risk_*.json from the existing 25-frame capture
- existing frame_*.png/jpg CARLA RGB files (filename and size only)
No CARLA metadata, actor truth, speed, distance, VLM, Qwen, or new inference
is read. No model loads, simulator connection, network or downloads.

Outputs in evidence/yolo_temporal_readiness/ (new files only):
- YOLO schema/box audit JSON
- per-frame CSV
- offline HTML summary

NOTE: Even a parsed YOLO box is only a model prediction. Image-space boxes
and object classes alone cannot establish lead-vehicle identity, distance,
closing rate, TTC or road danger.

PowerShell (from project root):
  & $PY -m scripts.yolo_temporal_readiness `
      --capture-dir ".\evidence\carla_lead_brake\20261116_140208_590540" `
      --compare-frames 2358 2373
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime
from html import escape
import io
import json
import math
from pathlib import Path
import re
import sys
from typing import Any

from PIL import Image
from scripts.scene_fusion_preview import find_records, latest_yolo, original_risk

SCHEMA_VERSION = "yolo_temporal_readiness.v1"
# Explicitly encoded *pixel* XYXY formats; do not guess from bare bbox arrays.
XYXY_ARRAY_KEYS = ("bbox_xyxy", "box_xyxy", "xyxy", "coordinates_xyxy")
BOX_CONTAINER_KEYS = ("bbox", "box", "bounding_box", "boundingBox", "coordinates")
DETECTION_CONTAINER_KEYS = ("detections",)
CLASS_KEYS = ("class_name", "label", "name", "class", "classname")
CONF_KEYS = ("confidence", "conf", "score")
CSV_FIELDS = (
    "world_frame", "original_yolo_rule_risk", "detection_count",
    "classes", "parsed_pixel_xyxy_count", "bbox_format_unknown_count",
    "bbox_field_absent_count", "detection_path", "image_path",
    "image_width", "image_height", "record_status",
)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def capture_images(capture_dir: Path) -> list[dict[str, Any]]:
    photos = []
    for item in sorted(capture_dir.iterdir()):
        if not item.is_file() or item.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            continue
        found = re.fullmatch(r"frame_(\d+)\.(?:png|jpe?g)", item.name, flags=re.I)
        if not found:
            continue
        with Image.open(item) as image:
            image.verify()
        with Image.open(item) as image:
            width, height = image.size
        if not 1 <= width <= 8192 or not 1 <= height <= 8192:
            raise ValueError(f"Unsupported image dimensions in {item}.")
        photos.append({
            "world_frame": int(found.group(1)),
            "path": str(item.resolve()),
            "width": width,
            "height": height,
        })
    photos.sort(key=lambda e: e["world_frame"])
    ids = [row["world_frame"] for row in photos]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate RGB files for the same world_frame.")
    if not photos:
        raise ValueError(f"No frame_*.png/jpg images in {capture_dir}.")
    return photos


def find_detection_arrays(obj: Any, path: str = "$", depth: int = 0):
    """Return only explicit 'detections' arrays; retain their JSON paths."""
    if depth > 12:
        return
    if isinstance(obj, dict):
        for key, value in obj.items():
            child_path = f"{path}.{key}"
            if key in DETECTION_CONTAINER_KEYS and isinstance(value, list):
                yield child_path, value
            if isinstance(value, (dict, list)):
                yield from find_detection_arrays(value, child_path, depth + 1)
    elif isinstance(obj, list):
        for i, value in enumerate(obj):
            if isinstance(value, (dict, list)):
                yield from find_detection_arrays(value, f"{path}[{i}]", depth + 1)


def detection_array_from_record(record: dict[str, Any]) -> tuple[str | None, list[Any] | None]:
    candidates = list(find_detection_arrays(record))
    if not candidates:
        return None, None
    # Favor the deepest frame-owned explicit array rather than an aggregate.
    # If distinct arrays remain, refuse to fabricate a unique detection source.
    nonempty = [(p, d) for p, d in candidates if len(d) > 0]
    if len(nonempty) == 1:
        return nonempty[0]
    if len(nonempty) == 0 and len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        # Exact duplicates in surrounding wrappers are not reliable enough to
        # claim unique ownership of detections.
        raise ValueError(
            "Multiple 'detections' arrays in one frame record; manual schema review required."
        )
    return candidates[0]


def xyxy_tuple(raw: Any, image_width: int, image_height: int
               ) -> tuple[list[float] | None, str]:
    """Parse only a clearly XYXY-marked value; never infer bbox xywh vs xyxy."""
    if not isinstance(raw, (list, tuple)) or len(raw) != 4:
        return None, "malformed_named_xyxy"
    if any(isinstance(v, bool) or not isinstance(v, (int, float))
           or not math.isfinite(v) for v in raw):
        return None, "malformed_named_xyxy"
    points = [float(v) for v in raw]
    x1, y1, x2, y2 = points
    # Four coordinates in [0,1] could be *normalized* XYXY.
    # Never convert them to pixels or assume they are tiny pixel boxes.
    if 0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1:
        return None, "named_xyxy_normalized_or_ambiguous"
    if 0 <= x1 < x2 <= image_width and 0 <= y1 < y2 <= image_height:
        return points, "pixel_xyxy_parsed"
    return None, "named_xyxy_outside_image_or_normalized"


def parse_bbox(obj: dict[str, Any], width: int, height: int
               ) -> tuple[list[float] | None, str, str | None]:
    """Return coordinates, status, and originating field without guessing units."""
    for key in XYXY_ARRAY_KEYS:
        if key in obj:
            box, status = xyxy_tuple(obj[key], width, height)
            return box, status, key

    for key in BOX_CONTAINER_KEYS:
        if key not in obj:
            continue
        value = obj[key]
        if isinstance(value, dict):
            if "xyxy" in value:
                box, status = xyxy_tuple(value["xyxy"], width, height)
                return box, status, f"{key}.xyxy"
            for names in (
                ("x1", "y1", "x2", "y2"),
                ("xmin", "ymin", "xmax", "ymax"),
                ("x_min", "y_min", "x_max", "y_max"),
                ("left", "top", "right", "bottom"),
            ):
                if all(field in value for field in names):
                    box, status = xyxy_tuple([value[field] for field in names],
                                              width, height)
                    return box, status, f"{key}.{{{','.join(names)}}}"
        # Unlabeled bbox arrays could be xywh/center_xywh/normalized.
        return None, "bbox_format_unknown", key
    return None, "bbox_field_absent", None


def detection_summary(obj: Any, width: int, height: int, idx: int) -> dict[str, Any]:
    if not isinstance(obj, dict):
        raise ValueError(f"Detection #{idx} is not a JSON object.")
    raw_cls = next((obj[key] for key in CLASS_KEYS if key in obj), None)
    if not isinstance(raw_cls, str) or not raw_cls.strip():
        # Numeric 'class_id' or 'cls' is not a reliable human-readable class.
        label = None
    else:
        label = raw_cls.strip().lower()
    confidence = next((obj[key] for key in CONF_KEYS if key in obj), None)
    if confidence is not None:
        if isinstance(confidence, bool) or not isinstance(confidence, (float, int)) \
                or not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError(f"Invalid confidence at detection #{idx}.")
        confidence = round(float(confidence), 6)
    bbox, status, bbox_field = parse_bbox(obj, width, height)
    return {
        "index": idx,
        "class_predicted": label,
        "confidence": confidence,
        "bbox_pixel_xyxy": bbox,
        "bbox_parse_status": status,
        "bbox_source_field": bbox_field,
        "raw_detection_keys_and_types": {
            key: type(value).__name__
            for key, value in obj.items()
            if isinstance(key, str)
        },
    }


def best_matching_record(document: Any, world_frame: int) -> dict[str, Any] | None:
    matches = list(find_records(document, world_frame))
    if not matches:
        return None
    # Favor a matching frame object with an explicit detections list,
    # not an enclosing aggregate, while preserving uncertainty on duplicates.
    scored = []
    for record in matches:
        arrays = list(find_detection_arrays(record))
        if arrays:
            smallest_depth = min(p.count(".") for p, d in arrays)
            scored.append((smallest_depth, len(json.dumps(record, ensure_ascii=False)), record))
    if scored:
        scored.sort(key=lambda x: (x[0], x[1]))
        return scored[0][2]
    return min(matches, key=lambda obj: len(json.dumps(obj, ensure_ascii=False)))


def audit_yolo(document: Any, images: list[dict[str, Any]],
               compare_frames: tuple[int, int] = (2358, 2373)) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for image in images:
        frame = image["world_frame"]
        record = best_matching_record(document, frame)
        if record is None:
            rows.append({
                "world_frame": frame, "image_path": image["path"],
                "image_width": image["width"], "image_height": image["height"],
                "original_yolo_rule_risk": None,
                "detections": None, "detection_count": None, "detection_path": None,
                "record_status": "yolo_frame_not_found",
                "raw_frame_top_level_keys": [],
            })
            continue
        path, raw = detection_array_from_record(record)
        if raw is None:
            rows.append({
                "world_frame": frame, "image_path": image["path"],
                "image_width": image["width"], "image_height": image["height"],
                "original_yolo_rule_risk": original_risk(record),
                "detections": None, "detection_count": None, "detection_path": None,
                "record_status": "no_explicit_detections_array",
                "raw_frame_top_level_keys": sorted(record.keys()),
            })
            continue
        detections = [
            detection_summary(obj, image["width"], image["height"], i)
            for i, obj in enumerate(raw)
        ]
        rows.append({
            "world_frame": frame,
            "image_path": image["path"],
            "image_width": image["width"],
            "image_height": image["height"],
            "original_yolo_rule_risk": original_risk(record),
            "detections": detections,
            "detection_count": len(detections),
            "detection_path": path,
            "record_status": "detections_array_parsed",
            "raw_frame_top_level_keys": sorted(record.keys()),
        })

    statuses = Counter(row["record_status"] for row in rows)
    successful = [r for r in rows if r["detections"] is not None]
    boxes = Counter(
        det["bbox_parse_status"]
        for row in successful for det in row["detections"]
    )
    classes = Counter(
        det["class_predicted"] or "<class label unavailable>"
        for row in successful for det in row["detections"]
    )
    originals = Counter(
        str(r["original_yolo_rule_risk"]) for r in rows
    )

    by_id = {row["world_frame"]: row for row in rows}
    examples = {}
    for frame in compare_frames:
        if frame not in by_id:
            examples[str(frame)] = {"status": "image_not_found"}
        else:
            r = by_id[frame]
            examples[str(frame)] = {
                "status": r["record_status"],
                "yolo_risk": r["original_yolo_rule_risk"],
                "detection_count": r["detection_count"],
                "detections": r["detections"],
                "image_path": r["image_path"],
                "raw_frame_top_level_keys": r.get("raw_frame_top_level_keys", []),
                "detection_path": r.get("detection_path"),
            }

    parsed = boxes.get("pixel_xyxy_parsed", 0)
    raw_count = sum(row["detection_count"] or 0 for row in rows)
    if parsed == 0:
        readiness = "no_trustworthy_pixel_boxes_in_existing_report"
    elif parsed < raw_count:
        readiness = "partial_pixel_boxes_for_tracking_research_only"
    else:
        readiness = "pixel_boxes_present_but_identity_tracking_unverified"
    return {
        "schema_version": SCHEMA_VERSION,
        "image_count": len(images),
        "yolo_frame_status_counts": dict(sorted(statuses.items())),
        "risk_counts_from_original_yolo": dict(sorted(originals.items())),
        "raw_detection_count": raw_count,
        "raw_predicted_class_counts": dict(sorted(classes.items())),
        "bbox_status_counts": dict(sorted(boxes.items())),
        "valid_pixel_xyxy_count": parsed,
        "bbox_availability_status": readiness,
        "selected_frame_audit": examples,
        "rows": rows,
        "source_isolation": {
            "CARLA_metadata_or_actor_truth_read": False,
            "VLM_or_Qwen_used": False,
            "models_rerun": False,
        },
        "limitations": [
            "Model detections and boxes are unverified predictions.",
            "Only explicitly named pixel XYXY arrays or clear corner dictionaries are parsed.",
            "Unlabeled bbox arrays are never assumed to be XYXY or pixel coordinates.",
            "Having boxes does not establish lead-vehicle identity across frames.",
            "A missing YOLO box is not proof that the original scene lacks a vehicle.",
            "An image is not a calibrated depth map: no distance, speed or TTC is calculated.",
            "This report is about input readiness, NOT a new danger label or risk model.",
        ],
    }


def csv_report(rows: list[dict[str, Any]]) -> str:
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        detections = row["detections"]
        stats = Counter(x["bbox_parse_status"] for x in detections or [])
        writer.writerow({
            "world_frame": row["world_frame"],
            "original_yolo_rule_risk": row["original_yolo_rule_risk"],
            "detection_count": row["detection_count"],
            "classes": (
                json.dumps([d["class_predicted"] for d in detections],
                           ensure_ascii=False) if detections is not None else ""
            ),
            "parsed_pixel_xyxy_count": stats["pixel_xyxy_parsed"],
            "bbox_format_unknown_count": stats["bbox_format_unknown"],
            "bbox_field_absent_count": stats["bbox_field_absent"],
            "detection_path": row["detection_path"],
            "image_path": row["image_path"],
            "image_width": row["image_width"],
            "image_height": row["image_height"],
            "record_status": row["record_status"],
        })
    return stream.getvalue()


def html_report(audit: dict[str, Any]) -> str:
    trs = []
    for row in audit["rows"]:
        counts = Counter(x["bbox_parse_status"] for x in row["detections"] or [])
        classes = ", ".join(
            str(x["class_predicted"] or "(class unknown)")
            for x in (row["detections"] or [])
        )
        trs.append(
            f'<tr><td>{row["world_frame"]}</td>'
            f'<td>{escape(str(row["original_yolo_rule_risk"]))}</td>'
            f'<td>{escape(str(row["detection_count"]))}</td>'
            f'<td>{escape(classes)}</td>'
            f'<td>{counts["pixel_xyxy_parsed"]}</td>'
            f'<td>{counts["bbox_format_unknown"] + counts["bbox_field_absent"]}</td>'
            f'<td>{escape(row["record_status"])}</td></tr>'
        )
    pairs = []
    for frame, info in audit["selected_frame_audit"].items():
        txt = json.dumps(info, indent=2, ensure_ascii=False)
        pairs.append(
            f'<section class="card"><h2>Frame {escape(frame)} — raw keys and boxes</h2>'
            f'<pre>{escape(txt)}</pre></section>'
        )
    return """<!DOCTYPE html><html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>YOLO / Bounding-box readiness</title>
<style>
:root{font-family:Segoe UI,Arial,sans-serif;background:#f6f6f1;color:#253d43}
body{max-width:1100px;margin:auto;padding:27px 20px 70px}
h1{font-size:32px}h2{font-size:20px}
small,p{line-height:1.5;color:#4e6763}
.card{padding:20px;background:white;border:1px solid #dce6e0;
border-radius:16px;margin:20px 0}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:10px}
.metric{background:#e9f0ed;padding:16px;border-radius:11px}
.metric strong{display:block;font-size:28px}
table{border-collapse:collapse;width:100%;font-size:13px}
td,th{border-bottom:1px solid #e5eeea;padding:9px;text-align:left}
th{background:#eaf1ee}
.scroll{overflow:auto}pre{overflow:auto;white-space:pre-wrap;
word-break:break-word;font-size:12px;max-height:480px}
.warn{background:#fff2e6;border-left:3px solid #c17b4f;padding:14px}
</style></head><body>
<h1>YOLO bounding-box readiness</h1>
<p>Audit of ORIGINAL image and YOLO output artifacts only. No simulator ground truth or model inference.
A parsed box is not automatically a correct detection or tracked lead car.</p>
<div class="grid">
<div class="metric">RGB frames<strong>""" + str(audit["image_count"]) + """</strong></div>
<div class="metric">Detections<strong>""" + str(audit["raw_detection_count"]) + """</strong></div>
<div class="metric">Pixel XYXY boxes<strong>""" + str(audit["valid_pixel_xyxy_count"]) + """</strong></div>
</div>
<section class="card"><h2>Readiness finding</h2>
<p><strong>""" + escape(audit["bbox_availability_status"]) + """</strong></p>
<p class="warn">This report does not measure braking safety, classify hazards,
or calculate vehicle TTC. Missing boxes may be a report-schema issue.</p></section>
<section class="card"><h2>Frame-by-frame YOLO output</h2><div class="scroll"><table>
<thead><tr><th>Frame</th><th>Risk</th><th>Detections</th><th>Predicted classes</th>
<th>Parsed boxes</th><th>Absent / unknown boxes</th><th>Status</th></tr></thead>
<tbody>""" + "".join(trs) + """</tbody></table></div></section>
""" + "".join(pairs) + """
<section class="card"><h2>Limitations</h2><ul>""" + "".join(
    "<li>" + escape(x) + "</li>" for x in audit["limitations"]
) + """</ul></section></body></html>"""


def new_file(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as f:
        f.write(content)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--capture-dir", required=True, type=Path)
    ap.add_argument("--yolo-report", type=Path)
    ap.add_argument("--compare-frames", nargs=2, type=int, default=[2358, 2373])
    args = ap.parse_args()
    capture = args.capture_dir.resolve()
    if not capture.is_dir():
        raise FileNotFoundError(f"No capture directory: {capture}")
    yolo = args.yolo_report.resolve() if args.yolo_report else latest_yolo(capture)
    if not yolo.is_file():
        raise FileNotFoundError(f"No YOLO report: {yolo}")
    inputs = capture_images(capture)
    audit = audit_yolo(
        load_json(yolo), inputs, tuple(args.compare_frames)
    )
    audit["source_yolo_report"] = str(yolo)
    audit["source_capture_dir"] = str(capture)
    audit["created_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    project = Path(__file__).resolve().parent.parent
    dest = project / "evidence" / "yolo_temporal_readiness"
    stem = f"yolo_schema_{datetime.now():%Y%m%d_%H%M%S_%f}"
    paths = [
        (dest / (stem + ".json"), json.dumps(audit, ensure_ascii=False, indent=2)),
        (dest / (stem + ".csv"), csv_report(audit["rows"])),
        (dest / (stem + ".html"), html_report(audit)),
    ]
    for path, content in paths:
        new_file(path, content)
    print("=== Phase 4.2: YOLO visual temporal readiness ===")
    print("Capture RGB frames:", audit["image_count"])
    print("YOLO matching statuses:", audit["yolo_frame_status_counts"])
    print("Original YOLO risk counts:", audit["risk_counts_from_original_yolo"])
    print("Raw YOLO detections:", audit["raw_detection_count"])
    print("Pixel XYXY boxes parsed:", audit["valid_pixel_xyxy_count"])
    print("BBox field/status counts:", audit["bbox_status_counts"])
    print("Readiness:", audit["bbox_availability_status"])
    for frame, info in audit["selected_frame_audit"].items():
        print(f"Frame {frame}: risk={info.get('yolo_risk')} "
              f"detections={info.get('detection_count')} status={info['status']}")
        print("  frame top-level keys=", info.get("raw_frame_top_level_keys"))
        print("  detections source path=", info.get("detection_path"))
        for item in info.get("detections") or []:
            print("  class=", item["class_predicted"],
                  "bbox_status=", item["bbox_parse_status"],
                  "bbox_field=", item["bbox_source_field"],
                  "raw_keys=", sorted(item["raw_detection_keys_and_types"]))
    for path, _ in paths:
        print(path.suffix[1:].upper() + ":", path)
    print("PASS: detector schema inspected without simulator truth or model rerun")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
