r"""
Phase 4.0: lead-braking timeline audit (existing CARLA + YOLO records only).

Reads existing capture metadata.json and its original yolo_risk_*.json.
Joins rows by CARLA world frame. It DOES NOT load CARLA, run YOLO/VLM/Qwen,
download anything, or overwrite any prior evidence. No risk ground-truth
labels, calibrated bumper gap, TTC, or collision probability are assumed.

Outputs a new JSON and CSV in evidence/lead_brake_timeline/.

Example (PowerShell from project root):
  & $PY -m scripts.lead_brake_timeline `
      --capture-dir ".\evidence\carla_lead_brake\20261116_140208_590540" `
      --compare-frames 2358 2373

If the simulator's longitudinal actor-origin separation key is not among
the recognized names, the script prints matching numeric field candidates
and still saves the per-frame risk/detection timeline. Use --separation-key
EXACT_NAME on a later run only after confirming the metadata's exact key.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime
import json
import math
from pathlib import Path
import sys
from typing import Any

from scripts.scene_fusion_preview import (
    load, find_records, latest_yolo, original_risk, detection_info,
)

# Names are accepted only when their semantics indicate longitudinal origin
# separation. An arbitrary "distance" field might be Euclidean or bumper gap
# and MUST NOT be treated as equivalent.
SEPARATION_ALIASES = (
    "lead_relative_longitudinal_m",
    "lead_relative_longitudinal_distance_m",
    "lead_vehicle_relative_longitudinal_m",
    "lead_vehicle_relative_longitudinal_distance_m",
    "lead_longitudinal_origin_separation_m",
    "lead_origin_longitudinal_separation_m",
    "longitudinal_origin_separation_m",
    "lead_vehicle_longitudinal_m",
    "lead_longitudinal_distance_m",
    "lead_vehicle_relative_x_m",
    "lead_relative_x_m",
)
EGO_SPEED_ALIASES = ("ego_speed_kmh", "ego_vehicle_speed_kmh")
LEAD_SPEED_ALIASES = (
    "lead_speed_kmh", "lead_vehicle_speed_kmh",
    "front_vehicle_speed_kmh",
)
COLUMNS = (
    "world_frame", "ego_speed_kmh", "lead_speed_kmh",
    "origin_longitudinal_separation_m", "original_yolo_rule_risk",
    "yolo_detection_count", "yolo_detected_classes",
    "yolo_frame_status",
)


def numeric(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def extract_world_frame(item: dict[str, Any]) -> int:
    for key in ("world_frame", "image_frame", "frame", "frame_id"):
        x = item.get(key)
        if isinstance(x, int) and not isinstance(x, bool) and x >= 0:
            return x
        if isinstance(x, str) and x.isdecimal():
            return int(x)
    path = item.get("image_file") or item.get("image_path")
    if isinstance(path, str):
        import re
        match = re.search(r"frame_(\d+)\.(png|jpe?g)$", path, re.I)
        if match:
            return int(match.group(1))
    raise ValueError("A capture metadata record has no identifiable world frame.")


def flatten_numeric_fields(item: dict[str, Any], prefix: str = "",
                           max_depth: int = 4) -> dict[str, float]:
    """Inspect a bounded set of nested metadata, no guessed path semantics."""
    found: dict[str, float] = {}
    def scan(value: Any, path: str, depth: int) -> None:
        if depth > max_depth:
            return
        n = numeric(value)
        if n is not None:
            found[path] = n
        elif isinstance(value, dict):
            for key, child in value.items():
                if isinstance(key, str):
                    scan(child, f"{path}.{key}" if path else key, depth + 1)
    scan(item, prefix, 0)
    return found


def pick_numeric(
    fields: dict[str, float], aliases: tuple[str, ...],
    exact_key: str | None = None,
) -> tuple[float | None, str | None]:
    """Choose only exact field names or exact suffix names, never fuzzy matches."""
    if exact_key is not None:
        if exact_key in fields:
            return fields[exact_key], exact_key
        return None, None
    for alias in aliases:
        matches = sorted(
            key for key in fields
            if key == alias or key.endswith("." + alias)
        )
        if len(matches) == 1:
            return fields[matches[0]], matches[0]
        if len(matches) > 1:
            # If multiple sources share the same field suffix, do not guess.
            return None, None
    return None, None


def build_timeline(
    metadata: dict[str, Any], yolo_doc: Any,
    *,
    separation_key: str | None = None,
    compare_frames: tuple[int, int] = (2358, 2373),
) -> dict[str, Any]:
    frames = metadata.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError("metadata.json is missing a nonempty frames list.")
    rows: list[dict[str, Any]] = []
    seen: set[int] = set()
    audit_fields: set[str] = set()
    chosen_sep: set[str] = set()

    for entry in frames:
        if not isinstance(entry, dict):
            raise ValueError("Encountered malformed CARLA metadata frame.")
        frame_id = extract_world_frame(entry)
        if frame_id in seen:
            raise ValueError(f"Repeated metadata world_frame: {frame_id}")
        seen.add(frame_id)

        fields = flatten_numeric_fields(entry)
        for key in fields:
            if any(word in key.lower()
                   for word in ("lead", "front", "ego", "speed", "gap",
                                "longitudinal", "separation", "distance")):
                audit_fields.add(key)

        speed, _ = pick_numeric(fields, EGO_SPEED_ALIASES)
        lead_speed, _ = pick_numeric(fields, LEAD_SPEED_ALIASES)
        separation, actual_sep_field = pick_numeric(
            fields, SEPARATION_ALIASES, separation_key
        )
        if actual_sep_field:
            chosen_sep.add(actual_sep_field)

        matches = list(find_records(yolo_doc, frame_id))
        risk = None
        detections = None
        classes = None
        match_status = "not_found"
        if matches:
            match_status = "multiple_matches_check_structure" if len(matches) > 1 else "matched"
            record = max(
                matches,
                key=lambda obj: len(json.dumps(obj, ensure_ascii=False)),
            )
            risk = original_risk(record)
            det = detection_info(record)
            detections = det["count"]
            if det["objects"] is not None:
                classes = [
                    str(obj.get("class") or "unknown")
                    for obj in det["objects"]
                ]
        rows.append({
            "world_frame": frame_id,
            "ego_speed_kmh": speed,
            "lead_speed_kmh": lead_speed,
            "origin_longitudinal_separation_m": separation,
            "original_yolo_rule_risk": risk,
            "yolo_detection_count": detections,
            "yolo_detected_classes": classes,
            "yolo_frame_status": match_status,
        })

    rows.sort(key=lambda r: r["world_frame"])
    by_frame = {r["world_frame"]: r for r in rows}
    first_frame, second_frame = compare_frames
    comparison = None
    if first_frame in by_frame and second_frame in by_frame:
        earlier = by_frame[first_frame]
        later = by_frame[second_frame]
        sep_a = earlier["origin_longitudinal_separation_m"]
        sep_b = later["origin_longitudinal_separation_m"]
        delta = None if sep_a is None or sep_b is None else round(sep_b - sep_a, 5)
        comparison = {
            "earlier": earlier,
            "later": later,
            "risk_changed": earlier["original_yolo_rule_risk"] != later["original_yolo_rule_risk"],
            "actor_origin_separation_change_m": delta,
            "interpretation": (
                "Model classification changed between sampled images. "
                "An actor-origin longitudinal difference is NOT a bumper gap or "
                "an independent safety or false-negative label."
            ),
        }

    sep_rows = [
        r for r in rows
        if r["origin_longitudinal_separation_m"] is not None
    ]
    trend = None
    if len(sep_rows) >= 2:
        trend = {
            "measured_frame_count": len(sep_rows),
            "first_measured_world_frame": sep_rows[0]["world_frame"],
            "last_measured_world_frame": sep_rows[-1]["world_frame"],
            "first_origin_longitudinal_separation_m":
                sep_rows[0]["origin_longitudinal_separation_m"],
            "last_origin_longitudinal_separation_m":
                sep_rows[-1]["origin_longitudinal_separation_m"],
            "raw_end_minus_start_m": round(
                sep_rows[-1]["origin_longitudinal_separation_m"]
                - sep_rows[0]["origin_longitudinal_separation_m"], 5
            ),
            "consecutive_decreases": sum(
                b["origin_longitudinal_separation_m"] <
                a["origin_longitudinal_separation_m"]
                for a, b in zip(sep_rows, sep_rows[1:])
            ),
            "consecutive_increases": sum(
                b["origin_longitudinal_separation_m"] >
                a["origin_longitudinal_separation_m"]
                for a, b in zip(sep_rows, sep_rows[1:])
            ),
            "important": (
                "These changes are evaluated only between saved frames. "
                "Do not interpret as continuous-time measurements or TTC."
            ),
        }

    return {
        "schema_version": "carla_lead_brake_timeline.v1",
        "source_role": (
            "CARLA metadata = independent simulation trajectory for "
            "post-hoc auditing; never passed to the vision/LLM models."
        ),
        "joined_frame_count": len(rows),
        "yolo_frame_matches": sum(
            r["yolo_frame_status"] != "not_found" for r in rows
        ),
        "original_risk_distribution": dict(Counter(
            str(r["original_yolo_rule_risk"]) for r in rows
        )),
        "separation_field_matched": sorted(chosen_sep),
        "numeric_field_candidates_in_metadata": sorted(audit_fields),
        "separation_trend": trend,
        "selected_frame_comparison": comparison,
        "rows": rows,
        "limitations": [
            "YOLO low/medium/high are original rule outputs, NOT truth labels.",
            "Actors' longitudinal origin separation is NOT a calibrated bumper gap.",
            "No TTC, collision probability, or autonomous-control command is calculated.",
            "Image and metadata are joined by world frame, not visual object identity.",
            "A label decrease with reduced distance is a review flag, not proof of unsafe behavior.",
            "No VLM or Qwen inference is conducted; historical reports remain unchanged.",
        ],
    }


def save_new(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as file:
        file.write(text)


def csv_text(rows: list[dict[str, Any]]) -> str:
    import io
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=COLUMNS, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({
            **row,
            "yolo_detected_classes": (
                json.dumps(row["yolo_detected_classes"], ensure_ascii=False)
                if row["yolo_detected_classes"] is not None else ""
            ),
        })
    return buf.getvalue()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", required=True, type=Path)
    parser.add_argument("--yolo-report", default=None, type=Path)
    parser.add_argument("--compare-frames", nargs=2, type=int,
                        default=[2358, 2373])
    parser.add_argument("--separation-key", type=str, default=None,
                        help="Exact metadata field key/path, if needed.")
    args = parser.parse_args()

    capture_dir = args.capture_dir.expanduser().resolve()
    metadata_path = capture_dir / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing CARLA metadata: {metadata_path}")
    yolo_path = (
        args.yolo_report.expanduser().resolve()
        if args.yolo_report else latest_yolo(capture_dir)
    )
    if not yolo_path.is_file():
        raise FileNotFoundError(f"Missing YOLO report: {yolo_path}")

    result = build_timeline(
        load(metadata_path), load(yolo_path),
        separation_key=args.separation_key,
        compare_frames=tuple(args.compare_frames),
    )
    result["source_metadata_path"] = str(metadata_path)
    result["source_yolo_path"] = str(yolo_path)
    result["created_at"] = datetime.now().astimezone().isoformat(timespec="seconds")

    project = Path(__file__).resolve().parent.parent
    out_dir = project / "evidence" / "lead_brake_timeline"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    report_path = out_dir / f"timeline_{stamp}.json"
    csv_path = out_dir / f"timeline_{stamp}.csv"
    save_new(
        report_path,
        json.dumps(result, ensure_ascii=False, indent=2)
    )
    save_new(csv_path, csv_text(result["rows"]))

    print("=== Phase 4.0: lead-braking offline temporal audit ===")
    print("Capture frames:", result["joined_frame_count"])
    print("Matched original YOLO records:", result["yolo_frame_matches"])
    print("Original risk distribution:", result["original_risk_distribution"])
    for row in result["rows"]:
        print(
            f"Frame {row['world_frame']} | risk={row['original_yolo_rule_risk']} "
            f"| detections={row['yolo_detection_count']} "
            f"| ego_kmh={row['ego_speed_kmh']} "
            f"| lead_kmh={row['lead_speed_kmh']} "
            f"| origin_separation_m={row['origin_longitudinal_separation_m']}"
        )
    if result["selected_frame_comparison"] is not None:
        comp = result["selected_frame_comparison"]
        print(
            f"Compare {comp['earlier']['world_frame']} -> {comp['later']['world_frame']}: "
            f"risk={comp['earlier']['original_yolo_rule_risk']} -> "
            f"{comp['later']['original_yolo_rule_risk']}; "
            f"origin_sep_delta_m={comp['actor_origin_separation_change_m']}"
        )
    else:
        print("Comparison frame IDs absent from saved capture; no comparison fabricated.")

    if not result["separation_field_matched"]:
        print("NOTE: No recognized actor-origin longitudinal separation field.")
        print("Candidate numeric keys in metadata:")
        for key in result["numeric_field_candidates_in_metadata"]:
            print("  ", key)
        print("Use --separation-key EXACT_METADATA_KEY only after confirming units/semantics.")

    print("JSON:", report_path)
    print("CSV:", csv_path)
    print("PASS: existing trajectory and YOLO outputs aligned; no model rerun")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
