r"""
Phase 4.4 — offline YOLO track-continuity visual audit.

Read the EXISTING Phase 4.3 vehicle_tracks_*.json and only the matching RGB
images required to draw bounding boxes. Do not read CARLA metadata, speed,
ground truth, VLM or Qwen. Do not run any detector, tracker or simulator.

Output (all new):
  evidence/yolo_track_continuity/continuity_*.json
  evidence/yolo_track_continuity/continuity_*.csv
  evidence/yolo_track_continuity/continuity_*.html

The report shows track-ID start/end events and ALL saved RGB frames with
their original YOLO box hypotheses drawn over them. A Txxx ID is a
tentative image-space association, NOT a verified physical car identity.

From project root:
  & $PY -m scripts.yolo_track_continuity `
     --tracks-report ".\evidence\yolo_vehicle_box_tracks\vehicle_tracks_20261116_181725_280146.json" `
     --compare-frames 2358 2373
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
import re
import sys
from typing import Any

from scripts.yolo_vehicle_box_tracks import render_overlay

INPUT_SCHEMA = "yolo_vehicle_box_tracks.v1"
OUTPUT_SCHEMA = "yolo_track_continuity.v1"
VALID_CLASSES = {"car", "bus", "truck"}
VALID_STATES = {
    "new_unlinked_track",
    "unresolved_ambiguous_match_new_track",
    "adjacent_frame_match_tentative",
}
CSV_FIELDS = (
    "world_frame", "track_id", "predicted_class", "association_state",
    "class_changed_from_previous", "previous_class", "bbox_pixel_xyxy",
    "normalized_center_x", "normalized_center_y", "bbox_area_fraction",
    "center_region_heuristic", "matched_iou_from_previous",
    "image_path",
)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as f:
        result = json.load(f)
    if not isinstance(result, dict):
        raise ValueError("Expected top-level JSON object.")
    return result


def numeric(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError(f"{name} must be numeric.")
    num = float(value)
    if not math.isfinite(num):
        raise ValueError(f"{name} is not finite.")
    return num


def verify_report(original: dict[str, Any]) -> list[dict[str, Any]]:
    if original.get("schema_version") != INPUT_SCHEMA:
        raise ValueError(f"Expected {INPUT_SCHEMA}.")
    provenance = original.get("source_isolation")
    if not isinstance(provenance, dict):
        raise ValueError("Missing source-isolation record.")
    for field in (
        "CARLA_metadata_or_actor_truth_read",
        "ego_speed_or_relative_distance_used",
        "VLM_or_Qwen_used",
        "model_inference_performed",
    ):
        if provenance.get(field) is not False:
            raise ValueError(f"Cannot verify visual-only provenance: {field}")
    frames = original.get("frames")
    if not isinstance(frames, list) or not 1 <= len(frames) <= 200:
        raise ValueError("Expected 1 to 200 saved frames.")
    if original.get("world_frame_count") != len(frames):
        raise ValueError("Frame count in report is inconsistent.")
    expected_total = 0
    normalized: list[dict[str, Any]] = []
    seen_frame: set[int] = set()
    last_frame = -1
    track_appearances: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)

    for entry in frames:
        if not isinstance(entry, dict):
            raise ValueError("Invalid frame entry.")
        frame = entry.get("world_frame")
        if isinstance(frame, bool) or not isinstance(frame, int) or frame <= last_frame:
            raise ValueError("Frames must be sorted, nonnegative, and unique.")
        last_frame = frame
        seen_frame.add(frame)
        image_path = entry.get("image_path")
        if not isinstance(image_path, str) or not image_path.strip():
            raise ValueError(f"Missing image_path in frame {frame}.")
        width = entry.get("image_width")
        height = entry.get("image_height")
        if (
            isinstance(width, bool) or isinstance(height, bool) or
            not isinstance(width, int) or not isinstance(height, int)
            or not 16 <= width <= 8192 or not 16 <= height <= 8192
        ):
            raise ValueError(f"Invalid RGB size for frame {frame}.")
        detections = entry.get("tracked_vehicle_detections")
        if not isinstance(detections, list):
            raise ValueError(f"No tracked_vehicle_detections at frame {frame}.")
        if entry.get("vehicle_prediction_count") != len(detections):
            raise ValueError(f"Vehicle count mismatch in frame {frame}.")
        prior_ids = set()
        normalized_detections = []
        for detection in detections:
            if not isinstance(detection, dict):
                raise ValueError(f"Malformed vehicle detection at frame {frame}.")
            tid = detection.get("hypothesis_track_id")
            if not isinstance(tid, str) or re.fullmatch(r"T\d{3,}", tid) is None:
                raise ValueError(f"Unexpected tentative track ID at frame {frame}.")
            if tid in prior_ids:
                raise ValueError(f"Duplicate track ID {tid} in frame {frame}.")
            prior_ids.add(tid)
            cls = detection.get("class_predicted")
            if cls not in VALID_CLASSES:
                raise ValueError(f"Track {tid} has an unsupported class.")
            state = detection.get("association_state")
            if state not in VALID_STATES:
                raise ValueError(f"Track {tid} has unrecognized association state.")
            box = detection.get("box_pixel_xyxy")
            if not isinstance(box, list) or len(box) != 4:
                raise ValueError(f"Track {tid} has no pixel XYXY box.")
            coords = [numeric(value, "box coordinate") for value in box]
            x1, y1, x2, y2 = coords
            if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
                raise ValueError(f"Track {tid} box outside the RGB image.")
            conf = detection.get("confidence")
            if conf is not None and not 0 <= numeric(conf, "confidence") <= 1:
                raise ValueError(f"Track {tid} confidence outside [0,1].")
            ncx = numeric(detection.get("normalized_center_x"), "center x")
            ncy = numeric(detection.get("normalized_center_y"), "center y")
            area = numeric(detection.get("bbox_area_fraction"), "bbox area fraction")
            if not (0 <= ncx <= 1 and 0 <= ncy <= 1 and 0 < area <= 1):
                raise ValueError(f"Track {tid} invalid geometric metrics.")
            if abs(ncx - (x1+x2)/(2*width)) > 1e-4:
                raise ValueError(f"Track {tid} center x does not match bbox.")
            if abs(ncy - (y1+y2)/(2*height)) > 1e-4:
                raise ValueError(f"Track {tid} center y does not match bbox.")
            if abs(area - (x2-x1)*(y2-y1)/(width*height)) > 1e-4:
                raise ValueError(f"Track {tid} area fraction does not match bbox.")
            if not isinstance(detection.get("center_region_heuristic"), bool):
                raise ValueError("Missing center-region boolean heuristic.")
            if not isinstance(detection.get("label_changed_from_prior"), bool):
                raise ValueError("Missing class-change boolean.")
            record = {
                **detection,
                "box_pixel_xyxy": coords,
                "image_path": image_path,
                "world_frame": frame,
            }
            normalized_detections.append(record)
            track_appearances[tid].append((frame, record))
            expected_total += 1
        normalized.append({
            "world_frame": frame,
            "image_path": image_path,
            "image_width": width,
            "image_height": height,
            "original_yolo_rule_risk": entry.get("original_yolo_rule_risk"),
            "tracked_vehicle_detections": normalized_detections,
        })

    # Check that each repeated hypothesis ID has a legitimate adjacent
    # continuation and that no non-consecutive reappearance silently bridges
    # a gap. One-track objects observed once should be marked new.
    for tid, observations in track_appearances.items():
        for i, (frame, obj) in enumerate(observations):
            if i == 0:
                if obj["association_state"] == "adjacent_frame_match_tentative":
                    raise ValueError(f"{tid} started as a match, but has no earlier record.")
                if obj["prior_class"] is not None or obj["label_changed_from_prior"]:
                    raise ValueError(f"{tid} has a previous-class claim on its first frame.")
            else:
                prev_frame, prior = observations[i-1]
                if frame != prev_frame + 1:
                    raise ValueError(f"{tid} unexpectedly bridges unobserved world frames.")
                if obj["association_state"] != "adjacent_frame_match_tentative":
                    raise ValueError(f"{tid} reused an ID without an adjacent-frame link.")
                if obj.get("prior_class") != prior["class_predicted"]:
                    raise ValueError(f"{tid} previous class does not match the actual prior box.")
                if obj["label_changed_from_prior"] != (
                    obj["class_predicted"] != prior["class_predicted"]
                ):
                    raise ValueError(f"{tid} class-change flag inconsistent.")
    if original.get("four_wheel_vehicle_prediction_count") != expected_total:
        raise ValueError("Track report total vehicle count disagrees with detections.")
    if original.get("number_of_tentative_tracks") != len(track_appearances):
        raise ValueError("Track report tentative ID count disagrees with detections.")
    return normalized


def continuity_audit(
    original: dict[str, Any],
    compare_frames: tuple[int, int] = (2358, 2373),
) -> dict[str, Any]:
    rows = verify_report(original)
    first_id, last_id = compare_frames
    if first_id >= last_id:
        raise ValueError("Comparison frames must be ascending.")
    by_frame = {r["world_frame"]: r for r in rows}
    observed: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        for item in row["tracked_vehicle_detections"]:
            observed[item["hypothesis_track_id"]].append(item)

    histories = []
    for tid in sorted(observed, key=lambda v: int(v[1:])):
        items = observed[tid]
        begin, end = items[0], items[-1]
        histories.append({
            "hypothesis_track_id": tid,
            "first_world_frame": begin["world_frame"],
            "last_world_frame": end["world_frame"],
            "observed_frames": len(items),
            "first_predicted_class": begin["class_predicted"],
            "last_predicted_class": end["class_predicted"],
            "classes_in_order": [x["class_predicted"] for x in items],
            "class_change_events": [
                {"world_frame": x["world_frame"],
                 "prior_class": x.get("prior_class"),
                 "new_class": x["class_predicted"]}
                for x in items if x["label_changed_from_prior"]
            ],
            "first_center": [
                begin["normalized_center_x"], begin["normalized_center_y"]
            ],
            "last_center": [
                end["normalized_center_x"], end["normalized_center_y"]
            ],
            "first_area_fraction": begin["bbox_area_fraction"],
            "last_area_fraction": end["bbox_area_fraction"],
            "end_to_start_area_ratio_unverified": round(
                end["bbox_area_fraction"]/begin["bbox_area_fraction"], 4
            ),
            "state": "tentative_association_hypothesis_not_actor_identity",
        })

    starts = defaultdict(list)
    ends = defaultdict(list)
    for history in histories:
        starts[history["first_world_frame"]].append(history["hypothesis_track_id"])
        ends[history["last_world_frame"]].append(history["hypothesis_track_id"])

    frame_events = []
    for row in rows:
        frame = row["world_frame"]
        tracked = row["tracked_vehicle_detections"]
        frame_events.append({
            "world_frame": frame,
            "original_yolo_rule_risk": row["original_yolo_rule_risk"],
            "number_of_vehicle_box_hypotheses": len(tracked),
            "present_ids": [item["hypothesis_track_id"] for item in tracked],
            "new_track_ids": sorted(starts[frame]),
            "track_ids_last_seen_at_frame": sorted(ends[frame]),
            "class_changes": [
                {"track_id": d["hypothesis_track_id"],
                 "prior_class": d["prior_class"],
                 "new_class": d["class_predicted"]}
                for d in tracked if d["label_changed_from_prior"]
            ],
        })

    comparison: dict[str, Any] = {}
    for frame in (first_id, last_id):
        row = by_frame.get(frame)
        if row is None:
            comparison[str(frame)] = {"status": "frame_absent"}
        else:
            comparison[str(frame)] = {
                "original_yolo_rule_risk": row["original_yolo_rule_risk"],
                "track_ids": [
                    d["hypothesis_track_id"]
                    for d in row["tracked_vehicle_detections"]
                ],
                "boxes": [
                    {
                        "track_id": d["hypothesis_track_id"],
                        "class_predicted": d["class_predicted"],
                        "box_pixel_xyxy": d["box_pixel_xyxy"],
                        "center_region_heuristic": d["center_region_heuristic"],
                    }
                    for d in row["tracked_vehicle_detections"]
                ],
            }

    start_track_ids = set(comparison[str(first_id)].get("track_ids", []))
    end_track_ids = set(comparison[str(last_id)].get("track_ids", []))
    focus_ids = sorted(start_track_ids | end_track_ids)
    focus_histories = [x for x in histories if x["hypothesis_track_id"] in focus_ids]
    return {
        "schema_version": OUTPUT_SCHEMA,
        "frame_count": len(rows),
        "tentative_track_count": len(histories),
        "vehicle_box_count": sum(
            len(x["tracked_vehicle_detections"]) for x in rows
        ),
        "original_rule_risk_counts": dict(
            sorted(Counter(str(x["original_yolo_rule_risk"]) for x in rows).items())
        ),
        "selected_frames": comparison,
        "shared_ids_in_selected_frames": sorted(start_track_ids & end_track_ids),
        "focus_track_histories": focus_histories,
        "all_track_histories": histories,
        "frame_events": frame_events,
        "frames_for_visual_overlays": rows,
        "source_isolation": {
            "CARLA_truth_used": False,
            "speed_and_distance_used": False,
            "new_model_inference": False,
        },
        "limitations": [
            "Track IDs are tentative links between YOLO pixel boxes, not confirmed CARLA actor IDs.",
            "No shared ID across two frames does not establish an undetected vehicle; "
            "it means the current association did not preserve a continuous hypothesis.",
            "A class label change does not prove that car and truck describe the same vehicle.",
            "No tracked box or center-region flag establishes the ego driving lane.",
            "Box area is not calibrated physical distance or TTC.",
            "Original risk labels remain original YOLO rule results, not ground-truth hazard labels.",
            "All visuals are for human review; no risk reclassification or vehicle control.",
        ],
    }


def html_report(report: dict[str, Any]) -> str:
    histories = report["all_track_histories"]
    events = report["frame_events"]
    focus = set(x["hypothesis_track_id"] for x in report["focus_track_histories"])
    all_frames = [item["world_frame"] for item in report["frames_for_visual_overlays"]]

    head = """<!DOCTYPE html><html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>YOLO Track Continuity | CARLA Research</title>
<style>
:root{font-family:Segoe UI,Arial,sans-serif;background:#f7f6f1;color:#243d42}
body{max-width:1340px;margin:0 auto;padding:26px 22px 75px}
h1{font-size:34px;margin:8px 0}h2{font-size:20px}
.top{padding-bottom:18px;border-bottom:1px solid #c8d7d0}
.kicker{letter-spacing:.14em;text-transform:uppercase;color:#42776f;
font-size:12px;font-weight:bold}
.note{line-height:1.62;color:#506862}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:18px}
.card{border:1px solid #dce5df;background:white;border-radius:16px;padding:17px}
.card img{width:100%;height:auto;display:block;border-radius:9px}
figcaption{font-size:12px;color:#607875;margin-top:8px}
.frames{margin-top:25px}.break{color:#bc613f;font-weight:700}
.stats{display:flex;gap:12px;flex-wrap:wrap;margin:19px 0}
.stat{background:#e9f0ea;border-radius:12px;padding:14px;flex:1;min-width:140px}
.stat strong{display:block;font-size:26px}
.matrix{overflow-x:auto}.matrix table{width:100%;border-collapse:collapse;
font-size:12px;font-variant-numeric:tabular-nums}
.matrix th,.matrix td{padding:7px 5px;border-bottom:1px solid #e4ede7;
text-align:center;min-width:35px}
.matrix th:first-child,.matrix td:first-child{text-align:left;min-width:82px}
.present{background:#bfe3d6}.focus{background:#e7c49e}
table{width:100%;border-collapse:collapse;font-size:13px}
td,th{padding:10px 9px;border-bottom:1px solid #e2e9e5;text-align:left}
th{background:#edf2ef}.scroll{overflow-x:auto}
li{margin:7px 0;line-height:1.52}
</style></head><body>
<div class="top"><p class="kicker">CARLA research · Phase 4.4 · vision-only</p>
<h1>Where did the tentative vehicle tracks break?</h1>
<p class="note">This reuses existing YOLO boxes. T001, T004, etc. are unverified image-space
association IDs, NOT physically identified cars. Every saved RGB frame is displayed
with predicted vehicle boxes to support manual review; no CARLA truth is read.</p></div>
"""
    first, last = all_frames[0], all_frames[-1]
    focused = ", ".join(sorted(focus)) if focus else "(none)"
    status = ", ".join(report["shared_ids_in_selected_frames"]) or "(none)"
    stats = (
        '<div class="stats">'
        f'<div class="stat">Frames<strong>{report["frame_count"]}</strong></div>'
        f'<div class="stat">Tentative tracks<strong>{report["tentative_track_count"]}</strong></div>'
        f'<div class="stat">Vehicle boxes<strong>{report["vehicle_box_count"]}</strong></div>'
        f'<div class="stat">Shared comparison IDs<strong>{escape(status)}</strong></div>'
        '</div>'
        f'<p class="note">Focused tentative IDs: <strong>{escape(focused)}</strong>. '
        'An absent shared ID does not prove a collision, misclassification, or physical '
        'vehicle disappearance.</p>'
    )

    summary_rows = []
    for track in histories:
        tid = track["hypothesis_track_id"]
        title = escape(" → ".join(track["classes_in_order"]))
        changes = "; ".join(
            f'{e["world_frame"]}: {e["prior_class"]}→{e["new_class"]}'
            for e in track["class_change_events"]
        ) or "—"
        summary_rows.append(
            f'<tr><td><strong>{escape(tid)}</strong></td>'
            f'<td>{track["first_world_frame"]}</td>'
            f'<td>{track["last_world_frame"]}</td>'
            f'<td>{track["observed_frames"]}</td>'
            f'<td>{escape(track["first_predicted_class"])} → '
            f'{escape(track["last_predicted_class"])}</td>'
            f'<td>{escape(changes)}</td></tr>'
        )
    table = (
        '<section class="card"><h2>Track start / last seen</h2>'
        '<div class="scroll"><table><thead><tr>'
        '<th>Hypothesis</th><th>First seen</th><th>Last seen</th>'
        '<th>Observations</th><th>Class labels</th><th>Class-change events</th>'
        '</tr></thead><tbody>'
        + "".join(summary_rows) + '</tbody></table></div></section>'
    )
    matrix_rows = []
    for track in histories:
        tid = track["hypothesis_track_id"]
        observed = set(range(track["first_world_frame"], track["last_world_frame"]+1))
        cells = []
        for frame in all_frames:
            if frame in observed:
                css = "focus" if tid in focus else "present"
                cells.append(f'<td class="{css}" title="{escape(tid)} in frame {frame}">●</td>')
            else:
                cells.append("<td>·</td>")
        matrix_rows.append(
            f'<tr><td>{escape(tid)}</td>' + "".join(cells) + '</tr>'
        )
    matrix = (
        '<section class="card"><h2>Track continuity matrix</h2>'
        '<p class="note">Orange = track ID present in either comparison frame; '
        'green = other tentative track. Columns show saved world frames.</p>'
        '<div class="matrix"><table><thead><tr><th>Track</th>'
        + "".join(f"<th>{f}</th>" for f in all_frames)
        + "</tr></thead><tbody>" + "".join(matrix_rows)
        + "</tbody></table></div></section>"
    )
    events_rows = []
    for record in events:
        events_rows.append(
            f'<tr><td>{record["world_frame"]}</td>'
            f'<td>{escape(str(record["original_yolo_rule_risk"]))}</td>'
            f'<td>{record["number_of_vehicle_box_hypotheses"]}</td>'
            f'<td>{escape(", ".join(record["present_ids"]))}</td>'
            f'<td>{escape(", ".join(record["new_track_ids"]) or "—")}</td>'
            f'<td>{escape(", ".join(record["track_ids_last_seen_at_frame"]) or "—")}</td></tr>'
        )
    event_table = (
        '<section class="card"><h2>Frame-by-frame start and end events</h2>'
        '<p class="note">“Last seen” is a boundary of a tentative association, '
        'not proof a real vehicle left the image.</p>'
        '<div class="scroll"><table><thead><tr><th>Frame</th><th>YOLO risk</th>'
        '<th>Vehicle boxes</th><th>Present IDs</th>'
        '<th>New IDs</th><th>Last seen IDs</th>'
        '</tr></thead><tbody>'
        + "".join(events_rows) + '</tbody></table></div></section>'
    )

    # Reuse the existing, verified Phase 4.3 overlay renderer: no image editing
    # or new model inference; only the content of saved RGB frames is used.
    cards = []
    for frame in report["frames_for_visual_overlays"]:
        frame_id = frame["world_frame"]
        uri = render_overlay(
            frame["image_path"],
            frame["tracked_vehicle_detections"],
            frame["image_width"], frame["image_height"],
        )
        ids = ", ".join(
            item["hypothesis_track_id"]
            for item in frame["tracked_vehicle_detections"]
        ) or "(no car/truck/bus boxes)"
        classes = ", ".join(
            item["class_predicted"]
            for item in frame["tracked_vehicle_detections"]
        ) or "(none)"
        new_ids = next(
            event["new_track_ids"] for event in events
            if event["world_frame"] == frame_id
        )
        starts_text = ", ".join(new_ids) or "—"
        cards.append(
            '<div class="card">'
            f'<h2>Frame {frame_id} · risk {escape(str(frame["original_yolo_rule_risk"]))}</h2>'
            f'<img src="{uri}" alt="Saved YOLO vehicle boxes, frame {frame_id}">'
            f'<figcaption>IDs: {escape(ids)} · classes: {escape(classes)}'
            f' · newly started: {escape(starts_text)}</figcaption></div>'
        )
    images = (
        '<section class="frames"><h2>All saved frames / image-space overlays</h2>'
        '<p class="note">Review whether a box follows the RED lead car rather than '
        'a roadside scooter, truck, or unrelated vehicle. The script cannot establish '
        'a CARLA actor identity.</p><div class="grid">'
        + "".join(cards) + "</div></section>"
    )
    notes = (
        '<section class="card"><h2>Interpretation limits</h2><ul>'
        + "".join(f"<li>{escape(note)}</li>" for note in report["limitations"])
        + "</ul></section>"
    )
    return head + stats + table + matrix + event_table + images + notes + "</body></html>"


def csv_report(report: dict[str, Any]) -> str:
    data = io.StringIO()
    writer = csv.DictWriter(data, fieldnames=CSV_FIELDS, lineterminator="\n")
    writer.writeheader()
    for frame in report["frames_for_visual_overlays"]:
        for item in frame["tracked_vehicle_detections"]:
            writer.writerow({
                "world_frame": item["world_frame"],
                "track_id": item["hypothesis_track_id"],
                "predicted_class": item["class_predicted"],
                "association_state": item["association_state"],
                "class_changed_from_previous": item["label_changed_from_prior"],
                "previous_class": item.get("prior_class"),
                "bbox_pixel_xyxy": json.dumps(item["box_pixel_xyxy"]),
                "normalized_center_x": item["normalized_center_x"],
                "normalized_center_y": item["normalized_center_y"],
                "bbox_area_fraction": item["bbox_area_fraction"],
                "center_region_heuristic": item["center_region_heuristic"],
                "matched_iou_from_previous": item.get("match_iou_from_prior_frame"),
                "image_path": frame["image_path"],
            })
    return data.getvalue()


def new_file(destination: Path, text: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as stream:
        stream.write(text)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracks-report", required=True, type=Path)
    parser.add_argument("--compare-frames", nargs=2, type=int, default=[2358, 2373])
    args = parser.parse_args()

    source = args.tracks_report.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"No Phase 4.3 track report: {source}")
    result = continuity_audit(
        load_json(source), compare_frames=tuple(args.compare_frames)
    )
    result["source_tracks_report"] = str(source)
    result["created_at"] = datetime.now().astimezone().isoformat(timespec="seconds")

    # Build outputs BEFORE creating files, so a bad image is reported early.
    bodies = [
        (".json", json.dumps(result, ensure_ascii=False, indent=2)),
        (".csv", csv_report(result)),
        (".html", html_report(result)),
    ]
    project = Path(__file__).resolve().parent.parent
    stem = f"continuity_{datetime.now():%Y%m%d_%H%M%S_%f}"
    folder = project / "evidence" / "yolo_track_continuity"
    paths = [(folder / (stem + ext), body) for ext, body in bodies]
    for path, body in paths:
        new_file(path, body)

    print("=== Phase 4.4: YOLO tentative-track continuity audit ===")
    print("Frames:", result["frame_count"])
    print("Tracks:", result["tentative_track_count"])
    print("Vehicle boxes:", result["vehicle_box_count"])
    print("Shared IDs in selected frames:", result["shared_ids_in_selected_frames"])
    print("Track histories:")
    for row in result["all_track_histories"]:
        print(
            f"  {row['hypothesis_track_id']}: {row['first_world_frame']} -> "
            f"{row['last_world_frame']}, observations={row['observed_frames']}, "
            f"first_class={row['first_predicted_class']}, "
            f"last_class={row['last_predicted_class']}"
        )
        if row["class_change_events"]:
            print("    Class changes:", row["class_change_events"])
    print("Track start/end events:")
    for row in result["frame_events"]:
        if row["new_track_ids"] or row["track_ids_last_seen_at_frame"]:
            print(
                f"  Frame {row['world_frame']}: started={row['new_track_ids']} "
                f"last_seen={row['track_ids_last_seen_at_frame']}"
            )
    print("IMPORTANT: this does not verify physical vehicle identities or collision risk.")
    for path, _ in paths:
        print(path.suffix.lstrip(".").upper() + ":", path)
    print("PASS: 25-frame tracking hypotheses visualized with no new model inference")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
