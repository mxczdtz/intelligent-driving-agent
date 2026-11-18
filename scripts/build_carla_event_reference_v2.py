"""Build an evaluation-only, frame-aligned CARLA event reference table.

Input:  existing CARLA metadata.json and YOLO yolo_risk_*.json for three scenes.
Output: evidence/evaluation/carla_events_<timestamp>.csv and .json

Important research boundary:
  CARLA event truth (brake command / lateral lane band / signal state) is NOT
  a reference label for collision danger. Never feed the output to an LLM.
  This script computes event-conditioned baseline distributions, not risk
  detection accuracy or dangerous-event miss rate.

Run from the project root:
    python -m scripts.build_carla_event_reference

No CARLA process, GPU, model weights or third-party Python packages required.
Original input reports are never modified.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any

SCENES = {
    "carla_lead_brake": "lead_vehicle_braking",
    "carla_pedestrian": "pedestrian_lane_crossing",
    "carla_traffic_light": "traffic_light_green_yellow_red",
}
RISK_LEVELS = {"unknown", "low", "medium", "high"}
CSV_COLUMNS = (
    "scene", "capture", "world_frame", "phase",
    "reference_event_type", "reference_event_value",
    "yolo_risk", "yolo_detection_count", "yolo_inference_ms",
    "ego_speed_kmh", "lead_speed_kmh", "lead_brake_command",
    "longitudinal_origin_separation_m", "closing_speed_mps",
    "walker_inside_lane_lateral_band", "walker_relative_longitudinal_m",
    "walker_relative_lateral_m", "walker_planar_origin_distance_m",
    "traffic_light_state", "stop_waypoint_longitudinal_distance_m",
)


def load_object(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        obj = json.load(handle)
    if not isinstance(obj, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return obj


def number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        n = float(value)
    except (TypeError, ValueError):
        return None
    return n if math.isfinite(n) else None


def boolean(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if type(value) is int and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        s = value.strip().lower()
        if s in ("true", "false", "1", "0"):
            return s in ("true", "1")
    return None


def brake_command_event(value: Any) -> str:
    """Decode CARLA vehicle brake-control intensity for evaluation ONLY.

    CARLA's normalized brake command is numeric in [0, 1]. In JSON it may be
    written as 0.0 / 1.0 (floats), rather than False / True or 0 / 1.
    Any finite positive intensity is an active command. Invalid or out-of-range
    values remain 'unavailable' instead of being silently relabeled.
    """
    if isinstance(value, bool):
        return "active" if value else "inactive"
    if isinstance(value, str) and value.strip().lower() in ("true", "false"):
        return "active" if value.strip().lower() == "true" else "inactive"
    strength = number(value)
    if strength is None or strength < 0.0 or strength > 1.0:
        return "unavailable"
    return "active" if strength > 1e-6 else "inactive"


def risk_level(value: Any) -> str:
    if isinstance(value, dict):
        for key in ("risk_level", "level", "severity", "grade"):
            if key in value:
                value = value[key]
                break
        else:
            return "unparsed"
    if isinstance(value, str) and value.strip().lower() in RISK_LEVELS:
        return value.strip().lower()
    return "unparsed"


def signal_state(value: Any) -> str:
    if not isinstance(value, str):
        return "unavailable"
    normalized = value.strip().lower()
    # Strings may come from CARLA TrafficLightState enums.
    if normalized.startswith("trafficlightstate."):
        normalized = normalized.rsplit(".", 1)[-1]
    return normalized if normalized in ("green", "yellow", "red") else "unavailable"


def indexed_frames(data: dict[str, Any], path: Path) -> dict[int, dict[str, Any]]:
    frames = data.get("frames")
    if not isinstance(frames, list):
        raise ValueError(f"Missing frames list: {path}")
    declared = data.get("matched_frame_count", data.get("frame_count"))
    if type(declared) is not int or declared != len(frames):
        raise ValueError(
            f"Frame count inconsistency in {path}: declared={declared!r}, "
            f"actual={len(frames)}"
        )
    out: dict[int, dict[str, Any]] = {}
    for index, frame in enumerate(frames):
        if not isinstance(frame, dict):
            raise ValueError(f"Frame index {index} not an object: {path}")
        fid = frame.get("world_frame")
        if type(fid) is not int or fid < 0:
            raise ValueError(f"Invalid world_frame {fid!r} in {path}")
        if fid in out:
            raise ValueError(f"Duplicate world_frame={fid} in {path}")
        out[fid] = frame
    return out


def canonical_image(value: Any) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    return value.replace("\\", "/").rsplit("/", 1)[-1].lower()


def selected_yolo_report(folder: Path) -> tuple[Path, dict[int, dict[str, Any]], int]:
    paths = sorted(folder.glob("yolo_risk_*.json"))
    if not paths:
        raise FileNotFoundError(f"No YOLO reports in {folder}")
    chosen = paths[0]  # deterministic: earliest timestamp-encoded filename
    primary = indexed_frames(load_object(chosen), chosen)
    for alternate in paths[1:]:
        other = indexed_frames(load_object(alternate), alternate)
        if set(primary) != set(other):
            raise ValueError(
                f"YOLO reports for one capture have different frame IDs: "
                f"{chosen.name} vs {alternate.name}; select baseline manually."
            )
        for fid in primary:
            for key in ("image_file", "detections", "risk"):
                left = json.dumps(primary[fid].get(key), ensure_ascii=False, sort_keys=True)
                right = json.dumps(other[fid].get(key), ensure_ascii=False, sort_keys=True)
                if left != right:
                    raise ValueError(
                        f"YOLO reports disagree on {key} at frame {fid}: "
                        f"{chosen.name} vs {alternate.name}; "
                        f"do not silently select one."
                    )
        # Inference time is purposely not compared: separate repeated passes
        # often have different cold-start/warm timings on identical predictions.
    return chosen, primary, len(paths) - 1


def reference_event(scene: str, frame: dict[str, Any]) -> tuple[str, str]:
    """Only simulator-reported observable events; never severity inference."""
    if scene == "carla_lead_brake":
        return "lead_brake_command", brake_command_event(
            frame.get("lead_brake_command")
        )
    if scene == "carla_pedestrian":
        inside = boolean(frame.get("walker_inside_lane_lateral_band"))
        return "walker_lateral_lane_band", (
            "inside" if inside is True else "outside" if inside is False else "unavailable"
        )
    if scene == "carla_traffic_light":
        return "traffic_light_state", signal_state(frame.get("traffic_light_state"))
    raise ValueError(f"Unrecognized scene: {scene}")


def safe_scalar(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    return str(value)[:240]


def make_row(scene: str, capture: str, frame: dict[str, Any],
             yolo: dict[str, Any]) -> dict[str, Any]:
    event_kind, event_value = reference_event(scene, frame)
    detections = yolo.get("detections")
    return {
        "scene": scene,
        "capture": capture,
        "world_frame": frame["world_frame"],
        "phase": safe_scalar(frame.get("phase")),
        "reference_event_type": event_kind,
        "reference_event_value": event_value,
        "yolo_risk": risk_level(yolo.get("risk")),
        "yolo_detection_count": len(detections) if isinstance(detections, list) else None,
        "yolo_inference_ms": number(yolo.get("inference_ms")),
        "ego_speed_kmh": number(frame.get("ego_speed_kmh")),
        "lead_speed_kmh": number(frame.get("lead_speed_kmh")),
        "lead_brake_command": safe_scalar(frame.get("lead_brake_command")),
        "longitudinal_origin_separation_m": number(frame.get("longitudinal_origin_separation_m")),
        "closing_speed_mps": number(frame.get("closing_speed_mps")),
        "walker_inside_lane_lateral_band": boolean(frame.get("walker_inside_lane_lateral_band")),
        "walker_relative_longitudinal_m": number(frame.get("walker_relative_longitudinal_m")),
        "walker_relative_lateral_m": number(frame.get("walker_relative_lateral_m")),
        "walker_planar_origin_distance_m": number(frame.get("walker_planar_origin_distance_m")),
        "traffic_light_state": signal_state(frame.get("traffic_light_state")) if scene == "carla_traffic_light" else None,
        "stop_waypoint_longitudinal_distance_m": number(frame.get("stop_waypoint_longitudinal_distance_m")),
    }


def percentile_nearest_rank(data: list[float], percentage: float) -> float | None:
    if not data:
        return None
    values = sorted(data)
    return values[max(0, math.ceil(percentage / 100.0 * len(values)) - 1)]


def latency_summary(values: list[float]) -> str:
    if not values:
        return "n=0"
    return (
        f"n={len(values)}, median={statistics.median(values):.2f} ms, "
        f"p95={percentile_nearest_rank(values,95):.2f} ms, "
        f"max={max(values):.2f} ms"
    )


def distribution(labels: list[str]) -> str:
    counts = Counter(labels)
    return ", ".join(f"{v}={counts[v]}" for v in ("low", "medium", "high", "unknown", "unparsed") if counts[v]) or "(empty)"


def build(project: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    records: list[dict[str, Any]] = []
    scenes: list[dict[str, Any]] = []
    duplicate_reports_ignored = 0
    for folder_name, expected_scenario in SCENES.items():
        parent = project / "evidence" / folder_name
        capture_folders = sorted(
            path for path in parent.iterdir()
            if path.is_dir() and (path / "metadata.json").is_file()
        ) if parent.is_dir() else []
        if not capture_folders:
            raise FileNotFoundError(f"No capture with metadata.json: {parent}")
        for capture in capture_folders:
            meta_file = capture / "metadata.json"
            metadata = load_object(meta_file)
            if metadata.get("scenario") != expected_scenario:
                raise ValueError(
                    f"Unexpected metadata scenario {metadata.get('scenario')!r} in "
                    f"{meta_file}; expected {expected_scenario!r}"
                )
            truth = indexed_frames(metadata, meta_file)
            yolo_file, baseline, ignored = selected_yolo_report(capture)
            duplicate_reports_ignored += ignored
            if set(truth) != set(baseline):
                missing_from_yolo = sorted(set(truth) - set(baseline))
                missing_from_truth = sorted(set(baseline) - set(truth))
                raise ValueError(
                    f"Frame alignment mismatch in {capture.name}; "
                    f"missing YOLO IDs={missing_from_yolo[:12]}, "
                    f"missing CARLA IDs={missing_from_truth[:12]}"
                )

            scene_rows: list[dict[str, Any]] = []
            for fid in sorted(truth):
                a, b = truth[fid], baseline[fid]
                a_image, b_image = canonical_image(a.get("image_file")), canonical_image(b.get("image_file"))
                if a_image is not None and b_image is not None and a_image != b_image:
                    raise ValueError(
                        f"Image filename mismatch at frame {fid}: "
                        f"CARLA={a_image} vs YOLO={b_image}"
                    )
                a_camera, b_camera = a.get("image_frame"), b.get("image_frame")
                if a_camera is not None and b_camera is not None and a_camera != b_camera:
                    raise ValueError(
                        f"Camera image_frame mismatch at frame {fid}: "
                        f"CARLA={a_camera!r} vs YOLO={b_camera!r}"
                    )
                row = make_row(folder_name, capture.name, a, b)
                scene_rows.append(row)
                records.append(row)

            events = Counter(row["reference_event_value"] for row in scene_rows)
            latencies = [row["yolo_inference_ms"] for row in scene_rows
                         if row["yolo_inference_ms"] is not None and row["yolo_inference_ms"] >= 0]
            warm = [row["yolo_inference_ms"] for row in scene_rows[1:]
                    if row["yolo_inference_ms"] is not None and row["yolo_inference_ms"] >= 0]
            conditional = {}
            for event_value in sorted(events):
                predictions = [row["yolo_risk"] for row in scene_rows
                               if row["reference_event_value"] == event_value]
                conditional[event_value] = dict(sorted(Counter(predictions).items()))
            counts = Counter(row["yolo_risk"] for row in scene_rows)
            summary = {
                "scene_folder": folder_name,
                "scenario": expected_scenario,
                "capture": capture.name,
                "frame_count": len(scene_rows),
                "reference_event_counts": dict(sorted(events.items())),
                "yolo_risk_counts": dict(sorted(counts.items())),
                "event_conditioned_yolo_risk": conditional,
                "yolo_report_selected": str(yolo_file),
                "additional_same_capture_reports_ignored": ignored,
                "yolo_latency_all_frames_ms": {
                    "count": len(latencies),
                    "median": statistics.median(latencies) if latencies else None,
                    "p95": percentile_nearest_rank(latencies, 95),
                    "maximum": max(latencies) if latencies else None,
                },
                "yolo_latency_excluding_first_frame_ms": {
                    "count": len(warm),
                    "median": statistics.median(warm) if warm else None,
                    "p95": percentile_nearest_rank(warm, 95),
                    "maximum": max(warm) if warm else None,
                },
                "ground_truth_metadata": str(meta_file),
                "event_truth_is_collision_danger_label": False,
            }
            if folder_name == "carla_traffic_light":
                expected_counts = (metadata.get("scenario_validation") or {}).get("frames_per_state")
                if isinstance(expected_counts, dict):
                    if {k: int(v) for k, v in expected_counts.items()} != {
                        k: events.get(k, 0) for k in expected_counts
                    }:
                        raise ValueError(
                            "Recorded per-frame traffic light truth does not match "
                            f"scenario_validation.frames_per_state in {meta_file}: {dict(events)}"
                        )
            scenes.append(summary)

            print(f"\n=== {folder_name} / {capture.name} ===")
            print(f"Frames aligned: {len(scene_rows)}")
            print(f"YOLO report: {yolo_file.name}; duplicate analyses ignored: {ignored}")
            print("CARLA event reference: " + ", ".join(f"{k}={n}" for k, n in sorted(events.items())))
            print("YOLO risk predictions: " + distribution([r["yolo_risk"] for r in scene_rows]))
            print("YOLO inference (all): " + latency_summary(latencies))
            print("YOLO inference (excluding first captured frame): " + latency_summary(warm))
            print("YOLO risk by CARLA event:")
            for event, counts_for_event in sorted(conditional.items()):
                print(f"  {event}: " + ", ".join(f"{k}={v}" for k, v in counts_for_event.items()))
            if folder_name == "carla_lead_brake":
                values = [
                    number(frame.get("lead_brake_command"))
                    for frame in truth.values()
                ]
                parsed_values = [v for v in values if v is not None]
                print(
                    "CARLA lead brake command (raw normalized intensity): "
                    + (
                        f"min={min(parsed_values):.4f}, max={max(parsed_values):.4f}"
                        if parsed_values else "no numeric observations"
                    )
                )
            if events.get("unavailable"):
                unparseable = [
                    frame.get("lead_brake_command")
                    for frame in truth.values()
                    if reference_event(folder_name, frame)[1] == "unavailable"
                ] if folder_name == "carla_lead_brake" else []
                print(
                    "CAUTION: unavailable event observations="
                    f"{events['unavailable']}; examples={unparseable[:4]}"
                )
    return records, scenes, duplicate_reports_ignored


def write_outputs(project: Path, records: list[dict[str, Any]],
                  scenes: list[dict[str, Any]], duplicates: int,
                  output_dir: Path | None) -> tuple[Path, Path]:
    dest = output_dir or project / "evidence" / "evaluation"
    dest.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    prefix = f"carla_events_{stamp}"
    csv_file = dest / f"{prefix}.csv"
    json_file = dest / f"{prefix}.json"
    # Exclusive creation preserves previous evaluation runs.
    with csv_file.open("x", encoding="utf-8-sig", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=CSV_COLUMNS, extrasaction="raise")
        writer.writeheader()
        writer.writerows(records)
    report = {
        "schema_version": "carla_event_evaluation.v1",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "evaluation_only_never_use_as_model_input": True,
        "warning": (
            "Event reference labels are CARLA facts, not independent risk-severity labels. "
            "No collision-risk accuracy, dangerous-event miss rate or TTC computed here. "
            "Only original YOLO rule predictions appear in this dataset."
        ),
        "project_root": str(project),
        "unique_aligned_frame_count": len(records),
        "duplicated_same_capture_yolo_reports_ignored": duplicates,
        "scene_summaries": scenes,
        "records": records,
    }
    try:
        with json_file.open("x", encoding="utf-8") as out:
            json.dump(report, out, ensure_ascii=False, indent=2, allow_nan=False)
    except Exception:
        csv_file.unlink(missing_ok=True)
        raise
    return csv_file, json_file


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root", type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="Project containing evidence/ (default: one level above scripts/).",
    )
    parser.add_argument("--output-dir", type=Path, help="Optional output folder.")
    args = parser.parse_args()
    project = args.project_root.resolve()
    print("=== CARLA EVENT REFERENCE: evaluation-only, NO GPU ===")
    records, scene_summaries, duplicates = build(project)
    csv_file, json_file = write_outputs(
        project, records, scene_summaries, duplicates,
        args.output_dir.resolve() if args.output_dir else None,
    )
    print("\n=== FINAL DATASET SUMMARY ===")
    print("Unique aligned frames:", len(records))
    print("Same-capture duplicate YOLO reports ignored:", duplicates)
    print("Collision-danger ground truth labels: NOT ASSIGNED")
    print("Risk accuracy / dangerous miss rate: NOT CALCULATED")
    print("CSV:", csv_file)
    print("JSON:", json_file)
    print("PASS: CARLA event truth aligned with existing YOLO baseline")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
