r"""
Phase 4.3 — vision-only, conservative vehicle box association (research only).

INPUT: Phase 4.2 yolo_schema_*.json containing saved YOLO per-frame pixel
XYXY boxes. This tool DOES NOT open CARLA metadata or actor truth, nor use
simulator speeds, distances, original risk logic, VLM, Qwen or any new model.
It uses the saved RGB files ONLY to draw human-review overlays.

Vehicle classes car, truck and bus are in one broad association category, so
car->truck label changes CAN be examined without declaring identity. Matching
uses adjacent frame pixel-box IoU and normalized center displacement; ambiguous
matches are rejected and new tentative tracks are started. Frame gaps are not
bridged. This is a tracking HYPOTHESIS, not a stable physical object ID.

No TTC, real range, safety/collision classification or vehicle control.

Input:
  & $PY -m scripts.yolo_vehicle_box_tracks `
    --readiness-report ".\evidence\yolo_temporal_readiness\yolo_schema_20261116_180128_313613.json" `
    --compare-frames 2358 2373

Output under evidence/yolo_vehicle_box_tracks:
  vehicle_tracks_*.json, vehicle_tracks_*.csv, vehicle_tracks_*.html
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime
from html import escape
import io
import json
import math
from pathlib import Path
import sys
from typing import Any
import base64

from PIL import Image, ImageDraw, ImageFont

INPUT_SCHEMA = "yolo_temporal_readiness.v1"
OUTPUT_SCHEMA = "yolo_vehicle_box_tracks.v1"
VEHICLE_CLASSES = frozenset({"car", "truck", "bus"})
CSV_COLUMNS = (
    "world_frame", "hypothesis_track_id", "detection_index", "class_predicted",
    "confidence", "x1", "y1", "x2", "y2", "normalized_center_x",
    "normalized_center_y", "bbox_area_fraction", "association_state",
    "match_iou_from_prior_frame", "match_center_shift_normalized",
    "prior_class", "label_changed_from_prior", "center_region_heuristic",
)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as fp:
        data = json.load(fp)
    if not isinstance(data, dict):
        raise ValueError("Expected top-level JSON object.")
    return data


def finite_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric.")
    val = float(value)
    if not math.isfinite(val):
        raise ValueError(f"{name} must be finite.")
    return val


def box_metrics(box: list[float], width: int, height: int) -> dict[str, float]:
    x1, y1, x2, y2 = box
    return {
        "normalized_center_x": round((x1+x2) / (2*width), 6),
        "normalized_center_y": round((y1+y2) / (2*height), 6),
        "bbox_area_fraction": round(
            (x2-x1) * (y2-y1) / (width*height), 8
        ),
    }


def normalize_frames(readiness: dict[str, Any]) -> list[dict[str, Any]]:
    if readiness.get("schema_version") != INPUT_SCHEMA:
        raise ValueError(f"Expected {INPUT_SCHEMA} input report.")
    isolation = readiness.get("source_isolation")
    if not isinstance(isolation, dict) or isolation.get("CARLA_metadata_or_actor_truth_read") is not False:
        raise ValueError("Input provenance cannot establish separation from CARLA truth.")
    if isolation.get("models_rerun") is not False:
        raise ValueError("Input provenance cannot establish offline behavior.")
    rows = readiness.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("No YOLO frame records.")
    if readiness.get("image_count") != len(rows):
        raise ValueError("Image count differs from saved YOLO rows.")
    normalized: list[dict[str, Any]] = []
    last_frame = -1
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"Malformed input row #{index}.")
        frame = row.get("world_frame")
        if isinstance(frame, bool) or not isinstance(frame, int) or frame <= last_frame:
            raise ValueError("Frames must be increasing, unique, nonnegative integers.")
        last_frame = frame
        width, height = row.get("image_width"), row.get("image_height")
        if (
            isinstance(width, bool) or not isinstance(width, int) or
            isinstance(height, bool) or not isinstance(height, int) or
            not (16 <= width <= 8192 and 16 <= height <= 8192)
        ):
            raise ValueError(f"Invalid image dimensions for frame {frame}.")
        path = row.get("image_path")
        if not isinstance(path, str) or not path.strip():
            raise ValueError(f"No image path in frame {frame}.")
        raw_detections = row.get("detections")
        if raw_detections is not None and not isinstance(raw_detections, list):
            raise ValueError(f"Invalid detection array in frame {frame}.")
        count = row.get("detection_count")
        if raw_detections is not None and (
            isinstance(count, bool) or not isinstance(count, int)
            or count != len(raw_detections)
        ):
            raise ValueError(f"Detector count mismatch in frame {frame}.")

        objects: list[dict[str, Any]] = []
        for det_index, det in enumerate(raw_detections or []):
            if not isinstance(det, dict):
                raise ValueError(f"Invalid detection #{det_index} in frame {frame}.")
            cls = det.get("class_predicted")
            if not isinstance(cls, str):
                # Keep missing class as missing; don't fabricate a vehicle class.
                cls = None
            confidence = det.get("confidence")
            if confidence is not None:
                confidence = finite_float(confidence, "confidence")
                if not 0 <= confidence <= 1:
                    raise ValueError("YOLO confidence must be from 0 to 1.")

            status = det.get("bbox_parse_status")
            raw_box = det.get("bbox_pixel_xyxy")
            bbox = None
            if status == "pixel_xyxy_parsed":
                if not isinstance(raw_box, list) or len(raw_box) != 4:
                    raise ValueError(f"Parsed bbox absent at frame {frame}, detection {det_index}.")
                bbox = [finite_float(x, "bbox coordinate") for x in raw_box]
                x1, y1, x2, y2 = bbox
                if not 0 <= x1 < x2 <= width or not 0 <= y1 < y2 <= height:
                    raise ValueError(f"bbox outside image in frame {frame}.")
            elif raw_box is not None:
                # Do not accept opaque bounding-box formats.
                raise ValueError("Non-XYXY bbox must not have interpreted pixel coordinates.")

            if cls not in VEHICLE_CLASSES or bbox is None:
                continue
            metrics = box_metrics(bbox, width, height)
            objects.append({
                "world_frame": frame,
                "detection_index": det_index,
                "class_predicted": cls,
                "confidence": confidence,
                "box_pixel_xyxy": bbox,
                "image_width": width,
                "image_height": height,
                **metrics,
                # This is a visual image-center heuristic, NOT a lane detector.
                "center_region_heuristic": (
                    .35 <= metrics["normalized_center_x"] <= .65
                    and metrics["normalized_center_y"] >= .20
                ),
            })
        normalized.append({
            "world_frame": frame, "image_path": path,
            "image_width": width, "image_height": height,
            "original_yolo_rule_risk": row.get("original_yolo_rule_risk"),
            "all_detection_count": count,
            "vehicle_detections": objects,
            "yolo_record_status": row.get("record_status"),
        })
    return normalized


def iou(a: list[float], b: list[float]) -> float:
    ix1, iy1 = max(a[0],b[0]), max(a[1],b[1])
    ix2, iy2 = min(a[2],b[2]), min(a[3],b[3])
    intersection = max(0,ix2-ix1)*max(0,iy2-iy1)
    area_a = (a[2]-a[0])*(a[3]-a[1])
    area_b = (b[2]-b[0])*(b[3]-b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def candidate_edge(prior: dict[str, Any], current: dict[str, Any],
                   max_center_shift: float = .12,
                   min_iou: float = .12) -> dict[str, float] | None:
    if prior["image_width"] != current["image_width"] or prior["image_height"] != current["image_height"]:
        return None
    dx = current["normalized_center_x"]-prior["normalized_center_x"]
    dy = current["normalized_center_y"]-prior["normalized_center_y"]
    distance = math.hypot(dx, dy)
    if distance > max_center_shift:
        return None
    area_a, area_b = prior["bbox_area_fraction"], current["bbox_area_fraction"]
    if area_a <= 0 or area_b <= 0:
        return None
    ratio = max(area_a, area_b) / min(area_a, area_b)
    overlap = iou(prior["box_pixel_xyxy"], current["box_pixel_xyxy"])
    # Either meaningful overlap, OR very small centroid motion + moderate size drift.
    if not (overlap >= min_iou or (distance <= .025 and ratio <= 1.75)):
        return None
    if ratio > 3:
        return None
    score = (
        .70 * overlap + .25 * max(0,1-distance/max_center_shift)
        + .05 * (1.0 if prior["class_predicted"] == current["class_predicted"] else 0)
    )
    return {"score": round(score, 6), "iou": round(overlap, 6),
            "center_distance": round(distance, 6)}


def associate_two_frames(
    last: list[dict[str, Any]],
    current: list[dict[str, Any]],
    *,
    ambiguity_margin: float = .06,
) -> tuple[list[tuple[int, int, dict[str, float]]], set[int]]:
    """Greedy matching; discard near-ties rather than invent a stable identity."""
    links: list[tuple[int,int,dict[str,float]]] = []
    for i, a in enumerate(last):
        for j, b in enumerate(current):
            edge = candidate_edge(a,b)
            if edge:
                links.append((i,j,edge))
    links.sort(key=lambda t: t[2]["score"], reverse=True)

    ambiguous: set[int] = set()
    # If one new object is similarly compatible with two prior boxes, reject it.
    for j in range(len(current)):
        scores = sorted([e[2]["score"] for e in links if e[1] == j], reverse=True)
        if len(scores) >= 2 and scores[0] - scores[1] <= ambiguity_margin:
            ambiguous.add(j)

    # If one prior box is similarly compatible with multiple new detections,
    # reject those new matches as ambiguous too.
    for i in range(len(last)):
        related = [e for e in links if e[0] == i]
        if len(related) >= 2 and (
            related[0][2]["score"] -
            sorted([e[2]["score"] for e in related], reverse=True)[1]
            <= ambiguity_margin
        ):
            ambiguous.update(e[1] for e in related)

    assigned_old: set[int] = set()
    assigned_new: set[int] = set()
    chosen = []
    for i,j,edge in links:
        if j in ambiguous or i in assigned_old or j in assigned_new:
            continue
        assigned_old.add(i)
        assigned_new.add(j)
        chosen.append((i,j,edge))
    return chosen, ambiguous


def track_frames(readiness: dict[str, Any], *,
                 compare_frames: tuple[int,int] = (2358,2373),
                 ambiguity_margin: float = .06) -> dict[str, Any]:
    if not 0 <= ambiguity_margin <= .3:
        raise ValueError("ambiguity_margin must be in [0,.3].")
    frames = normalize_frames(readiness)
    all_entries: list[dict[str, Any]] = []
    next_id = 1
    previous_frame: dict[str, Any] | None = None
    prior_tracked: list[dict[str, Any]] = []
    for frame in frames:
        current = [dict(obj) for obj in frame["vehicle_detections"]]
        matched = {}
        ambiguous = set()
        if previous_frame is not None and frame["world_frame"] == previous_frame["world_frame"]+1:
            chosen, ambiguous = associate_two_frames(
                prior_tracked, current, ambiguity_margin=ambiguity_margin
            )
            for i,j,edge in chosen:
                matched[j] = (prior_tracked[i], edge)
        for j, item in enumerate(current):
            if j in matched:
                prior, edge = matched[j]
                item.update({
                    "hypothesis_track_id": prior["hypothesis_track_id"],
                    "association_state": "adjacent_frame_match_tentative",
                    "match_iou_from_prior_frame": edge["iou"],
                    "match_center_shift_normalized": edge["center_distance"],
                    "prior_class": prior["class_predicted"],
                    "label_changed_from_prior": (
                        prior["class_predicted"] != item["class_predicted"]
                    ),
                })
            else:
                item.update({
                    "hypothesis_track_id": f"T{next_id:03d}",
                    "association_state": (
                        "unresolved_ambiguous_match_new_track"
                        if j in ambiguous else "new_unlinked_track"
                    ),
                    "match_iou_from_prior_frame": None,
                    "match_center_shift_normalized": None,
                    "prior_class": None,
                    "label_changed_from_prior": False,
                })
                next_id += 1
        frame["tracked_vehicle_detections"] = current
        frame["vehicle_prediction_count"] = len(current)
        frame["center_region_vehicle_count"] = sum(
            item["center_region_heuristic"] for item in current
        )
        all_entries.extend(current)
        previous_frame = frame
        prior_tracked = current

    grouped: dict[str,list[dict[str,Any]]] = defaultdict(list)
    for item in all_entries:
        grouped[item["hypothesis_track_id"]].append(item)
    summaries = []
    for track_id, history in grouped.items():
        first, last = history[0], history[-1]
        labels = [e["class_predicted"] for e in history]
        summaries.append({
            "hypothesis_track_id": track_id,
            "first_world_frame": first["world_frame"],
            "last_world_frame": last["world_frame"],
            "frames_observed": len(history),
            "predicted_classes_in_order": labels,
            "class_changes_in_adjacent_matches": sum(
                item["label_changed_from_prior"] for item in history
            ),
            "last_to_first_bbox_area_fraction_ratio": round(
                last["bbox_area_fraction"] / first["bbox_area_fraction"], 4
            ) if first["bbox_area_fraction"] > 0 else None,
            "identity_status": "tentative_image_space_box_association_hypothesis_only",
        })
    by_frame = {r["world_frame"]:r for r in frames}
    comparisons = {}
    for frame in compare_frames:
        if frame in by_frame:
            row = by_frame[frame]
            comparisons[str(frame)] = {
                "original_yolo_rule_risk": row["original_yolo_rule_risk"],
                "all_detection_count": row["all_detection_count"],
                "vehicle_detection_count": row["vehicle_prediction_count"],
                "center_region_vehicle_count": row["center_region_vehicle_count"],
                "tentative_ids": [
                    obj["hypothesis_track_id"]
                    for obj in row["tracked_vehicle_detections"]
                ],
                "vehicle_detections": row["tracked_vehicle_detections"],
            }
        else:
            comparisons[str(frame)] = {"status":"frame_not_in_report"}

    a, b = compare_frames
    shared_ids = []
    if a in by_frame and b in by_frame:
        ids_a = {
            t["hypothesis_track_id"] for t in
            by_frame[a]["tracked_vehicle_detections"]
        }
        ids_b = {
            t["hypothesis_track_id"] for t in
            by_frame[b]["tracked_vehicle_detections"]
        }
        shared_ids = sorted(ids_a & ids_b)

    return {
        "schema_version": OUTPUT_SCHEMA,
        "world_frame_count": len(frames),
        "input_detection_box_count": readiness.get("raw_detection_count"),
        "parsed_pixel_xyxy_box_count": readiness.get("valid_pixel_xyxy_count"),
        "four_wheel_vehicle_prediction_count": len(all_entries),
        "number_of_tentative_tracks": len(summaries),
        "tentative_adjacent_links": sum(
            obj["association_state"] == "adjacent_frame_match_tentative"
            for obj in all_entries
        ),
        "ambiguous_new_track_count": sum(
            obj["association_state"] == "unresolved_ambiguous_match_new_track"
            for obj in all_entries
        ),
        "class_change_on_tentative_link_count": sum(
            obj["label_changed_from_prior"] for obj in all_entries
        ),
        "center_region_heuristic": (
            "center x in [0.35,0.65], normalized center y >=0.20. "
            "An image center region is NOT the verified ego driving lane."
        ),
        "frame_comparisons": comparisons,
        "same_tentative_ids_in_comparison_frames": shared_ids,
        "track_summaries": summaries,
        "frames": frames,
        "source_isolation": {
            "CARLA_metadata_or_actor_truth_read": False,
            "ego_speed_or_relative_distance_used": False,
            "VLM_or_Qwen_used": False,
            "model_inference_performed": False,
        },
        "limitations": [
            "Track IDs are IMAGE-SPACE association hypotheses, not confirmed actor identities.",
            "A car->truck class change is a model-label change, not proof that the same car survived.",
            "Skipped frames are never bridged; ambiguity creates a fresh track.",
            "Only car/truck/bus are associated; motorcycle is deliberately excluded.",
            "Central image position is not a calibrated lane association.",
            "Apparent bounding-box size change is NOT calibrated depth, range, TTC or physical closing speed.",
            "Failure to detect a vehicle in a frame is not proof it disappeared from the road.",
            "Results support visual review only; no collision-risk judgement or vehicle control.",
        ],
    }


def render_overlay(image_path: str, detections: list[dict[str,Any]],
                   expected_width: int, expected_height: int) -> str:
    """Return a self-contained JPEG thumbnail of the actual RGB with annotated boxes."""
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"Cannot annotate missing RGB image: {path}")
    with Image.open(path) as picture:
        if picture.size != (expected_width, expected_height):
            raise ValueError(f"RGB size mismatch for {path}: {picture.size}")
        image = picture.convert("RGB")

    painter = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    for item in detections:
        box = item["box_pixel_xyxy"]
        x1,y1,x2,y2 = [int(round(x)) for x in box]
        changed = item["label_changed_from_prior"]
        color = "#E3774F" if changed else "#21B7A9"
        painter.rectangle((x1,y1,x2,y2), outline=color, width=3)
        label = f'{item["hypothesis_track_id"]} {item["class_predicted"]}'
        if changed:
            label += " *"
        bbox = painter.textbbox((0,0),label,font=font)
        label_w = bbox[2]-bbox[0]
        label_h = bbox[3]-bbox[1]
        ly = max(0,y1-label_h-9)
        painter.rectangle(
            (x1,ly,min(expected_width-1,x1+label_w+9),ly+label_h+7),
            fill=color
        )
        painter.text((x1+4,ly+3),label,font=font,fill="#073D43")

    image.thumbnail((960,720),Image.Resampling.LANCZOS)
    memory = io.BytesIO()
    image.save(memory,format="JPEG",quality=89,optimize=True)
    return "data:image/jpeg;base64,"+base64.b64encode(memory.getvalue()).decode("ascii")


def csv_report(result: dict[str, Any]) -> str:
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=CSV_COLUMNS, lineterminator="\n")
    writer.writeheader()
    for frame in result["frames"]:
        for item in frame["tracked_vehicle_detections"]:
            x1,y1,x2,y2 = item["box_pixel_xyxy"]
            writer.writerow({
                "world_frame":item["world_frame"],
                "hypothesis_track_id":item["hypothesis_track_id"],
                "detection_index":item["detection_index"],
                "class_predicted":item["class_predicted"],
                "confidence":item["confidence"],
                "x1":x1,"y1":y1,"x2":x2,"y2":y2,
                "normalized_center_x":item["normalized_center_x"],
                "normalized_center_y":item["normalized_center_y"],
                "bbox_area_fraction":item["bbox_area_fraction"],
                "association_state":item["association_state"],
                "match_iou_from_prior_frame":item["match_iou_from_prior_frame"],
                "match_center_shift_normalized":item["match_center_shift_normalized"],
                "prior_class":item["prior_class"],
                "label_changed_from_prior":item["label_changed_from_prior"],
                "center_region_heuristic":item["center_region_heuristic"],
            })
    return out.getvalue()


def html_report(result: dict[str,Any], frames_to_render: tuple[int,...]) -> str:
    lookup = {r["world_frame"]:r for r in result["frames"]}
    images = []
    for frame in frames_to_render:
        if frame not in lookup:
            continue
        record = lookup[frame]
        image_uri = render_overlay(
            record["image_path"], record["tracked_vehicle_detections"],
            record["image_width"],record["image_height"]
        )
        note = (
            f'Original rule: {record["original_yolo_rule_risk"]}; '
            f'four-wheel predictions: {record["vehicle_prediction_count"]}; '
            f'center-region candidates: {record["center_region_vehicle_count"]}'
        )
        images.append(
            f'<section class="card"><h2>Frame {frame}</h2>'
            f'<img alt="YOLO saved detection overlays at frame {frame}" '
            f'src="{image_uri}"><p class="muted">{escape(note)}</p></section>'
        )
    track_rows = []
    for t in result["track_summaries"]:
        track_rows.append(
            f'<tr><td>{escape(t["hypothesis_track_id"])}</td>'
            f'<td>{t["first_world_frame"]}–{t["last_world_frame"]}</td>'
            f'<td>{t["frames_observed"]}</td>'
            f'<td>{escape(" → ".join(t["predicted_classes_in_order"]))}</td>'
            f'<td>{t["class_changes_in_adjacent_matches"]}</td>'
            f'<td>{escape(str(t["last_to_first_bbox_area_fraction_ratio"]))}</td></tr>'
        )
    caveats = "".join("<li>"+escape(x)+"</li>" for x in result["limitations"])
    return """<!DOCTYPE html><html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>YOLO temporal box association | CARLA research</title>
<style>
:root{font-family:Segoe UI,Arial,sans-serif;color:#203C42;background:#F5F6F1}
body{max-width:1180px;margin:0 auto;padding:30px 22px 75px}
*{box-sizing:border-box}h1{font-size:34px}h2{font-size:19px;margin-top:0}
p,li{line-height:1.55}.minor{letter-spacing:.15em;font-size:12px;
text-transform:uppercase;color:#3D766C;font-weight:bold}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:16px}
.card{border:1px solid #DCE5E0;border-radius:15px;padding:20px;
background:white;margin:15px 0}.card img{max-width:100%;width:100%;
border-radius:10px}.stats{display:flex;flex-wrap:wrap;gap:12px}
.stat{background:#E6EEEA;border-radius:13px;padding:16px;min-width:145px;flex:1}
.stat strong{display:block;font-size:26px;margin-top:6px}
.muted{color:#526A6D;font-size:13px}
table{border-collapse:collapse;width:100%;font-size:13px}
td,th{padding:9px;border-bottom:1px solid #E1EBE8;text-align:left}
th{background:#E8F0EB}.scroll{overflow-x:auto}
.warn{border-left:3px solid #D48A54;padding:13px;background:#FAEEE6}
</style></head><body>
<p class="minor">CARLA research / Phase 4.3 / vision-only audit</p>
<h1>Conservative YOLO box association</h1>
<p>Vehicle track IDs are tentative image-space associations. Orange labels indicate a
class-label change on a tentative adjacent-frame match; teal labels do not establish identity.</p>
<div class="stats">
<div class="stat">RGB frames<strong>""" + str(result["world_frame_count"]) + """</strong></div>
<div class="stat">Vehicle predictions<strong>""" + str(result["four_wheel_vehicle_prediction_count"]) + """</strong></div>
<div class="stat">Tentative tracks<strong>""" + str(result["number_of_tentative_tracks"]) + """</strong></div>
<div class="stat">Adjacent links<strong>""" + str(result["tentative_adjacent_links"]) + """</strong></div>
</div><div class="grid">""" + "".join(images) + """</div>
<section class="card"><h2>Track hypotheses, not actor IDs</h2>
<p class="muted">Shared IDs between selected frames: """ + escape(
    ", ".join(result["same_tentative_ids_in_comparison_frames"]) or "(none)"
) + """</p><div class="scroll"><table><thead>
<tr><th>ID</th><th>World frames</th><th>Observations</th><th>Class labels</th>
<th>Class changes</th><th>Area ratio*</th></tr></thead><tbody>
""" + "".join(track_rows) + """</tbody></table></div>
<p class="muted">* Last/first bounding-box pixel area fraction. It is NOT
real-world distance or TTC.</p></section>
<section class="card"><h2>Interpretation limits</h2><p class="warn">
Do not use for safety decisions, real vehicle control or claims of confirmed tracking.</p>
<ul>""" + caveats + """</ul></section></body></html>"""


def save_new(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as fp:
        fp.write(content)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--readiness-report", required=True, type=Path)
    ap.add_argument("--compare-frames", nargs=2, type=int, default=[2358,2373])
    ap.add_argument("--ambiguity-margin", type=float, default=.06)
    args = ap.parse_args()
    source = args.readiness_report.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Phase 4.2 report not found: {source}")

    result = track_frames(
        read_json(source),
        compare_frames=tuple(args.compare_frames),
        ambiguity_margin=args.ambiguity_margin,
    )
    result["source_readiness_report"] = str(source)
    result["created_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    frame_ids = [r["world_frame"] for r in result["frames"]]
    middle_frame = frame_ids[len(frame_ids)//2]
    to_render = tuple(dict.fromkeys([
        args.compare_frames[0], middle_frame, args.compare_frames[1]
    ]))

    # Render ALL report bodies before creating files to avoid partial artifacts
    # when a source RGB file is missing or corrupt.
    json_body = json.dumps(result, ensure_ascii=False, indent=2)
    csv_body = csv_report(result)
    html_body = html_report(result, to_render)
    output_dir = Path(__file__).resolve().parent.parent / "evidence" / "yolo_vehicle_box_tracks"
    stem = f"vehicle_tracks_{datetime.now():%Y%m%d_%H%M%S_%f}"
    outputs = (
        (output_dir / (stem + ".json"), json_body),
        (output_dir / (stem + ".csv"), csv_body),
        (output_dir / (stem + ".html"), html_body),
    )
    for destination, content in outputs:
        save_new(destination, content)

    print("=== Phase 4.3: vision-only YOLO box association ===")
    print("Frames:", result["world_frame_count"])
    print("Parsed original pixel XYXY boxes:", result["parsed_pixel_xyxy_box_count"])
    print("Car/truck/bus predictions:", result["four_wheel_vehicle_prediction_count"])
    print("Tentative track hypotheses:", result["number_of_tentative_tracks"])
    print("Tentative adjacent-frame links:", result["tentative_adjacent_links"])
    print("Ambiguous matches not linked:", result["ambiguous_new_track_count"])
    print("Label changes across tentative links:", result["class_change_on_tentative_link_count"])
    for frame, data in result["frame_comparisons"].items():
        print(f"Frame {frame}: original risk={data.get('original_yolo_rule_risk')}, "
              f"vehicle boxes={data.get('vehicle_detection_count')}, "
              f"center heuristic={data.get('center_region_vehicle_count')}, "
              f"IDs={data.get('tentative_ids')}")
    print("IDs shared between comparison frames:", result["same_tentative_ids_in_comparison_frames"])
    print("IMPORTANT: matching is a hypothesis; no CARLA truth, range, TTC or danger label.")
    for destination, _ in outputs:
        print(destination.suffix[1:].upper() + ":", destination)
    print("PASS: vision-only temporal research report saved; no model re-run")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
