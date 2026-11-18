"""Evaluate CARLA event-aligned YOLO data against explicit *risk proxies*.

Inputs:  evidence/evaluation/carla_events_*.json, from build_carla_event_reference_v2.
Outputs: evidence/evaluation/proxy_assessment_*.json and .csv.

Research honesty:
- A CARLA event is an observed simulator fact; the risk proxy is a human-defined
  heuristic, NOT an independently validated danger ground truth.
- This code computes proxy agreement and potential under-alerts, NOT accuracy
  against confirmed danger labels or a dangerous-event miss rate.
- It reads YOLO outputs and CARLA metadata-derived fields already in the dataset;
  NO model inference, graphics, simulator or GPU is needed.
- These evaluator-only outputs MUST NEVER be fed into VLM/Qwen risk inference.

From the project root:
    python -m scripts.evaluate_carla_risk_proxies

Optional:
    python -m scripts.evaluate_carla_risk_proxies --input PATH
    python -m scripts.evaluate_carla_risk_proxies --self-test

Default, illustrative heuristic thresholds:
- Lead: origin-distance / positive closing speed as CENTER-POINT TTC proxy.
  HIGH if closing > 0.5 m/s and proxy TTC <= 2.0 s.
  MEDIUM if closing > 0.5 m/s and proxy TTC <= 4.0 s.
  LOW if lead is NOT commanded to brake, abs(closing) <= 0.25 m/s
      and origin distance >= 15 m; UNKNOWN otherwise.
  NOTE: center-point TTC is NOT bumper-to-bumper TTC; unknown vehicle
  extents, braking performance and path invalidate safety guarantees.
- Pedestrian: MEDIUM caution proxy only if walker is inside the lane
  lateral band, ego speed >= 3 km/h, pedestrian ahead and within 25 m.
  All other cases UNKNOWN. A lateral lane band does not prove a collision
  trajectory; HIGH is never assigned from these fields.
- Light: no collision-danger proxy from light color alone: UNKNOWN for all.
  The official CARLA signal state provides a separate action context only.
  The fact that a red light is ahead does not prove a red-light violation.

Thresholds can be changed using flags, and the chosen values are saved to JSON.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime
import json
import math
from pathlib import Path
import sys
from typing import Any

SCENE_IDS = (
    "carla_lead_brake",
    "carla_pedestrian",
    "carla_traffic_light",
)
LABELS = ("low", "medium", "high", "unknown")
CSV_FIELDS = (
    "scene", "capture", "world_frame", "carla_event",
    "yolo_risk", "proxy_risk", "proxy_reason",
    "yolo_vs_proxy", "center_origin_ttc_proxy_s",
    "action_context", "ego_speed_kmh",
    "lead_origin_separation_m", "closing_speed_mps",
    "walker_lane_band", "walker_relative_longitudinal_m",
    "traffic_light_state", "stop_waypoint_longitudinal_distance_m",
    "yolo_inference_ms",
)


def finite(value: Any) -> float | None:
    if isinstance(value, bool) or value is None or value == "":
        return None
    try:
        result = float(value)
    except (ValueError, TypeError):
        return None
    return result if math.isfinite(result) else None


def enum_value(value: Any, allowed: tuple[str, ...]) -> str:
    if isinstance(value, str) and value.strip().lower() in allowed:
        return value.strip().lower()
    return "unavailable"


def gt_strict(val: float | None, bound: float) -> bool:
    return val is not None and val > bound


def non_negative(val: float | None) -> bool:
    return val is not None and val >= 0


def proxy_row(item: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    scene = item.get("scene")
    event = str(item.get("reference_event_value") or "unavailable").lower()
    label = enum_value(item.get("yolo_risk"), LABELS)
    if label == "unavailable":
        raise ValueError(f"Unrecognized yolo_risk: {item.get('yolo_risk')!r}")
    try:
        fid = item["world_frame"]
        if type(fid) is not int or fid < 0:
            raise ValueError
    except (KeyError, ValueError):
        raise ValueError("Missing/invalid world_frame") from None

    ego = finite(item.get("ego_speed_kmh"))
    sep = finite(item.get("longitudinal_origin_separation_m"))
    closing = finite(item.get("closing_speed_mps"))
    forward = finite(item.get("walker_relative_longitudinal_m"))
    light = enum_value(item.get("traffic_light_state"), ("green", "yellow", "red"))
    distance_stop = finite(item.get("stop_waypoint_longitudinal_distance_m"))
    ttc = sep / closing if sep is not None and sep > 0 and closing is not None and closing > 0 else None

    severity = "unknown"
    reason = ""
    action = "none"
    if scene == "carla_lead_brake":
        if event not in ("active", "inactive", "unavailable"):
            raise ValueError(f"Invalid lead_brake event at frame {fid}: {event!r}")
        if sep is None or sep <= 0 or closing is None:
            reason = "Missing/invalid longitudinal origin separation or closing speed."
        elif closing > args.min_closing_mps and ttc is not None and ttc <= args.high_ttc_s:
            severity = "high"
            reason = "Positive closing speed and center-point TTC proxy <= high threshold."
        elif closing > args.min_closing_mps and ttc is not None and ttc <= args.medium_ttc_s:
            severity = "medium"
            reason = "Positive closing speed and center-point TTC proxy <= medium threshold."
        elif (event == "inactive" and abs(closing) <= args.steady_closing_mps
              and sep >= args.steady_min_gap_m):
            severity = "low"
            reason = "No lead-brake command; near-zero relative closing and minimum origin gap."
        else:
            reason = "Available kinematics do not satisfy a defined proxy category."
        action = "lead_braking_command_active" if event == "active" else (
            "lead_braking_command_inactive" if event == "inactive" else "unknown"
        )
    elif scene == "carla_pedestrian":
        if event not in ("inside", "outside", "unavailable"):
            raise ValueError(f"Invalid walker lane-band event at frame {fid}: {event!r}")
        if (event == "inside" and ego is not None and ego >= args.walker_min_ego_kmh
                and forward is not None and 0 < forward <= args.walker_max_forward_m):
            severity = "medium"
            reason = (
                "Walker in lateral lane band and ahead of a moving ego vehicle "
                "within attention distance; trajectory/collision not verified."
            )
        else:
            reason = (
                "No defensible severity proxy from lane-band occupancy alone "
                "outside the defined moving-ego forward exposure window."
            )
        action = "pedestrian_lane_band_" + event
    elif scene == "carla_traffic_light":
        if event not in ("green", "yellow", "red", "unavailable"):
            raise ValueError(f"Invalid signal state at frame {fid}: {event!r}")
        if light != event and event != "unavailable":
            raise ValueError(
                f"Traffic state inconsistency at world_frame={fid}: "
                f"event={event}, field={light}"
            )
        reason = (
            "Signal color and stop-line distance alone do not establish collision "
            "risk or a traffic-rule violation."
        )
        action = {
            "green": "proceed_only_if_clear",
            "yellow": "prepare_to_stop_if_safe",
            "red": "stop_before_stop_line",
        }.get(event, "unknown")
    else:
        raise ValueError(f"Unknown scene: {scene!r}")

    if severity == "unknown":
        relationship = "reference_proxy_abstains"
    elif label == "unknown":
        relationship = "yolo_abstains"
    elif severity == label:
        relationship = "exact_proxy_match"
    elif severity in ("medium", "high") and label == "low":
        relationship = "potential_under_alert"
    else:
        relationship = "proxy_disagreement"

    return {
        "scene": scene,
        "capture": item.get("capture"),
        "world_frame": fid,
        "carla_event": event,
        "yolo_risk": label,
        "proxy_risk": severity,
        "proxy_reason": reason,
        "yolo_vs_proxy": relationship,
        "center_origin_ttc_proxy_s": round(ttc, 5) if ttc is not None else None,
        "action_context": action,
        "ego_speed_kmh": ego,
        "lead_origin_separation_m": sep,
        "closing_speed_mps": closing,
        "walker_lane_band": event if scene == "carla_pedestrian" else None,
        "walker_relative_longitudinal_m": forward if scene == "carla_pedestrian" else None,
        "traffic_light_state": light if scene == "carla_traffic_light" else None,
        "stop_waypoint_longitudinal_distance_m": (
            distance_stop if scene == "carla_traffic_light" else None
        ),
        "yolo_inference_ms": finite(item.get("yolo_inference_ms")),
    }


def ratios(rows: list[dict[str, Any]]) -> dict[str, Any]:
    evaluable = [row for row in rows if row["proxy_risk"] in ("low", "medium", "high")]
    caution = [row for row in evaluable if row["proxy_risk"] in ("medium", "high")]
    exact = sum(row["yolo_vs_proxy"] == "exact_proxy_match" for row in evaluable)
    alerted = sum(row["yolo_vs_proxy"] == "potential_under_alert" for row in caution)
    abstained_caution = sum(row["yolo_vs_proxy"] == "yolo_abstains" for row in caution)
    predicted_known = [row for row in evaluable if row["yolo_risk"] != "unknown"]
    return {
        "n_all": len(rows),
        "n_proxy_evaluable": len(evaluable),
        "n_proxy_unknown": len(rows) - len(evaluable),
        "n_yolo_unknown_among_proxy_evaluable": sum(
            row["yolo_risk"] == "unknown" for row in evaluable
        ),
        "strict_agreement_on_proxy_evaluable": (
            exact / len(evaluable) if evaluable else None
        ),
        "selective_agreement_excluding_yolo_unknown": (
            exact / len(predicted_known) if predicted_known else None
        ),
        "caution_proxy_frames": len(caution),
        "caution_proxy_with_yolo_low": alerted,
        "potential_under_alert_rate_among_caution_proxies": (
            alerted / len(caution) if caution else None
        ),
        "caution_proxy_with_yolo_unknown": abstained_caution,
        "yolo_risk_distribution": dict(sorted(Counter(
            row["yolo_risk"] for row in rows
        ).items())),
        "proxy_risk_distribution": dict(sorted(Counter(
            row["proxy_risk"] for row in rows
        ).items())),
        "comparison_counts": dict(sorted(Counter(
            row["yolo_vs_proxy"] for row in rows
        ).items())),
        "not_true_danger_ground_truth": True,
    }


def choose_input(project: Path, supplied: Path | None) -> Path:
    if supplied is not None:
        path = supplied if supplied.is_absolute() else project / supplied
        if not path.is_file():
            raise FileNotFoundError(path)
        return path.resolve()
    base = project / "evidence" / "evaluation"
    paths = sorted(base.glob("carla_events_*.json")) if base.is_dir() else []
    if not paths:
        raise FileNotFoundError(
            f"No carla_events_*.json in {base}; generate it first."
        )
    return max(paths, key=lambda p: p.stat().st_mtime_ns).resolve()


def load_input(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as fh:
        contents = json.load(fh)
    if not isinstance(contents, dict) or contents.get("schema_version") != "carla_event_evaluation.v1":
        raise ValueError(f"Expected carla_event_evaluation.v1 JSON: {path}")
    if contents.get("evaluation_only_never_use_as_model_input") is not True:
        raise ValueError("Input is not marked as evaluation-only.")
    data = contents.get("records")
    if not isinstance(data, list) or not all(isinstance(x, dict) for x in data):
        raise ValueError("Invalid or missing records list.")
    expected = contents.get("unique_aligned_frame_count")
    if type(expected) is not int or expected != len(data):
        raise ValueError(f"Record count mismatch: declared={expected}, actual={len(data)}")
    seen: set[tuple[str, str, int]] = set()
    counts = Counter()
    for row in data:
        scene, capture, frame = row.get("scene"), row.get("capture"), row.get("world_frame")
        if scene not in SCENE_IDS or not isinstance(capture, str) or not capture or type(frame) is not int:
            raise ValueError(f"Invalid frame key in {path}: {scene}, {capture}, {frame}")
        key = (scene, capture, frame)
        if key in seen:
            raise ValueError(f"Duplicate frame row: {key}")
        seen.add(key)
        counts[scene] += 1
    for scene in SCENE_IDS:
        if counts[scene] == 0:
            raise ValueError(f"No frames from required scene: {scene}")
    return data


def save_output(project: Path, source: Path, args: argparse.Namespace,
                records: list[dict[str, Any]],
                summaries: dict[str, Any]) -> tuple[Path, Path]:
    output_dir = project / "evidence" / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"proxy_assessment_{datetime.now():%Y%m%d_%H%M%S_%f}"
    csv_out = output_dir / f"{stem}.csv"
    json_out = output_dir / f"{stem}.json"
    with csv_out.open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="raise")
        writer.writeheader()
        writer.writerows(records)
    report = {
        "schema_version": "carla_risk_proxy_assessment.v1",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "evaluation_only_never_use_as_model_input": True,
        "source_event_dataset": str(source),
        "warnings": [
            "CARLA event truth is independent of YOLO predictions, but the severity proxy is human-defined and NOT a validated danger label.",
            "Potential under-alert rate is not the confirmed dangerous-event miss rate.",
            "Center-origin TTC proxy is not bumper-to-bumper TTC and is not calibrated to vehicle geometry.",
            "Pedestrian lane-band occupancy does not confirm an impending collision.",
            "A red signal ahead does not mean the ego vehicle violated the signal; no severity proxy is assigned for lights.",
            "The 103 frames belong to three correlated sequences, not 103 independent scenarios.",
        ],
        "heuristic_parameters": {
            "high_ttc_s": args.high_ttc_s,
            "medium_ttc_s": args.medium_ttc_s,
            "min_closing_mps": args.min_closing_mps,
            "steady_closing_mps": args.steady_closing_mps,
            "steady_min_gap_m": args.steady_min_gap_m,
            "walker_min_ego_kmh": args.walker_min_ego_kmh,
            "walker_max_forward_m": args.walker_max_forward_m,
        },
        "metrics": summaries,
        "rows": records,
    }
    try:
        with json_out.open("x", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
    except Exception:
        csv_out.unlink(missing_ok=True)
        raise
    return csv_out, json_out


def self_test() -> None:
    args = argparse.Namespace(
        high_ttc_s=2.0, medium_ttc_s=4.0, min_closing_mps=0.5,
        steady_closing_mps=0.25, steady_min_gap_m=15.0,
        walker_min_ego_kmh=3.0, walker_max_forward_m=25.0,
    )
    base = {
        "world_frame":1, "capture":"sample", "yolo_risk":"low",
        "ego_speed_kmh":10.8,
    }
    def lead(sep: float, closing: float, phase: str) -> dict[str, Any]:
        return proxy_row({
            **base, "scene":"carla_lead_brake",
            "reference_event_value":phase,
            "longitudinal_origin_separation_m":sep,
            "closing_speed_mps":closing,
        }, args)
    assert lead(14.0, 7.0, "active")["proxy_risk"] == "high"
    assert lead(18.0, 6.0, "active")["proxy_risk"] == "medium"
    assert lead(18.0, 0.1, "inactive")["proxy_risk"] == "low"
    assert lead(18.0, 0.1, "active")["proxy_risk"] == "unknown"
    assert lead(18.0, 0.0, "inactive")["center_origin_ttc_proxy_s"] is None
    assert lead(14.0, 7.0, "active")["yolo_vs_proxy"] == "potential_under_alert"
    ped = proxy_row({
        **base, "scene":"carla_pedestrian",
        "reference_event_value":"inside",
        "walker_relative_longitudinal_m":18.0,
    }, args)
    assert ped["proxy_risk"] == "medium"
    assert ped["yolo_vs_proxy"] == "potential_under_alert"
    ped_o = proxy_row({
        **base, "scene":"carla_pedestrian",
        "reference_event_value":"outside",
        "walker_relative_longitudinal_m":18.0,
    }, args)
    assert ped_o["proxy_risk"] == "unknown"
    red = proxy_row({
        **base, "scene":"carla_traffic_light",
        "reference_event_value":"red", "traffic_light_state":"red",
        "stop_waypoint_longitudinal_distance_m":11.2,
    }, args)
    assert red["proxy_risk"] == "unknown" and red["action_context"] == "stop_before_stop_line"
    stats = ratios([lead(14.0, 7.0, "active"), lead(18.0, 0.1, "inactive"), red])
    assert stats["n_proxy_evaluable"] == 2
    assert stats["caution_proxy_with_yolo_low"] == 1
    assert stats["potential_under_alert_rate_among_caution_proxies"] == 1.0
    print("SELF-TEST PASS: boundary cases, abstention, evidence-only and score denominators")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path,
                        default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--input", type=Path, help="Input carla_events_*.json.")
    parser.add_argument("--high-ttc-s", type=float, default=2.0)
    parser.add_argument("--medium-ttc-s", type=float, default=4.0)
    parser.add_argument("--min-closing-mps", type=float, default=0.5)
    parser.add_argument("--steady-closing-mps", type=float, default=0.25)
    parser.add_argument("--steady-min-gap-m", type=float, default=15.0)
    parser.add_argument("--walker-min-ego-kmh", type=float, default=3.0)
    parser.add_argument("--walker-max-forward-m", type=float, default=25.0)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return 0
    vals = [
        args.high_ttc_s, args.medium_ttc_s, args.min_closing_mps,
        args.steady_closing_mps, args.steady_min_gap_m,
        args.walker_min_ego_kmh, args.walker_max_forward_m,
    ]
    if any(not math.isfinite(n) or n <= 0 for n in vals):
        parser.error("All heuristic thresholds must be finite positive numbers.")
    if args.high_ttc_s >= args.medium_ttc_s:
        parser.error("--high-ttc-s must be less than --medium-ttc-s.")

    project = args.project_root.resolve()
    source = choose_input(project, args.input)
    dataset = load_input(source)
    records = [proxy_row(row, args) for row in dataset]
    summaries = {"all_scenes": ratios(records), "by_scene": {}}
    for scene in SCENE_IDS:
        sub = [row for row in records if row["scene"] == scene]
        summaries["by_scene"][scene] = ratios(sub)

    csv_file, json_file = save_output(project, source, args, records, summaries)
    print("=== CARLA RISK PROXY EVALUATION (NO MODEL INFERENCE) ===")
    print("Input:", source)
    print("Thresholds: TTC high <= %.2fs; TTC medium <= %.2fs (center-origin only)." %
          (args.high_ttc_s, args.medium_ttc_s))
    for scene in SCENE_IDS:
        summary = summaries["by_scene"][scene]
        print("\n" + scene)
        print("  Frames:", summary["n_all"])
        print("  Risk proxy labels:", summary["proxy_risk_distribution"])
        print("  Valid proxy coverage: %s/%s" %
              (summary["n_proxy_evaluable"], summary["n_all"]))
        print("  Exact agreement (including YOLO abstention as mismatch):",
              summary["strict_agreement_on_proxy_evaluable"])
        print(
            "  Potential under-alert (YOLO LOW on MEDIUM/HIGH proxy): %s/%s" %
            (summary["caution_proxy_with_yolo_low"], summary["caution_proxy_frames"])
        )
        print(
            "  YOLO UNKNOWN on caution proxy:",
            summary["caution_proxy_with_yolo_unknown"]
        )
    print("\n=== SUMMARY ===")
    print("Distinct aligned frames:", len(records))
    print("Proxy-evaluable frames:", summaries["all_scenes"]["n_proxy_evaluable"])
    print("Provisional heuristic proxy: YES; true danger ground truth: NO")
    print("Confirmed risk accuracy / dangerous miss rate: NOT AVAILABLE")
    print("CSV:", csv_file)
    print("JSON:", json_file)
    print("PASS: reproducible, bounded proxy assessment produced")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
