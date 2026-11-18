r"""
Phase 4.1: offline lead-braking time-window diagnostic.
Input: a Phase 4.0 timeline_*.json already made from CARLA metadata + original YOLO.

Separately compares: (1) ego-speed minus lead-speed magnitudes,
(2) longitudinal ACTOR-ORIGIN separation changes between saved frames,
(3) the ORIGINAL rule risk. An audit flag is NOT a collision-risk label,
TTC, a false negative or real vehicle action. Simulator truth is for
post-hoc evaluation ONLY and is not passed to YOLO, VLM, or Qwen.

No inference, downloads, simulator, external assets, or new dependencies.
Outputs new JSON, CSV, HTML under evidence/lead_brake_diagnostics/.
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
import sys
from typing import Any

INPUT_SCHEMA = "carla_lead_brake_timeline.v1"
OUTPUT_SCHEMA = "lead_brake_diagnostics.v1"
RISKS = {None, "unknown", "low", "medium", "high"}
CSV_KEYS = (
    "world_frame", "ego_speed_kmh", "lead_speed_kmh",
    "ego_minus_lead_speed_kmh", "origin_longitudinal_separation_m",
    "origin_sep_change_from_prior_saved_frame_m", "original_yolo_rule_risk",
    "yolo_detection_count", "yolo_detected_classes",
    "review_context", "low_with_closing_context_flag",
)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as f:
        result = json.load(f)
    if not isinstance(result, dict):
        raise ValueError("Expected JSON object.")
    return result


def number(value: Any, field: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError(f"{field} should be numeric or null.")
    val = float(value)
    if not math.isfinite(val):
        raise ValueError(f"{field} should be finite.")
    return val


def validate_timeline(report: dict[str, Any]) -> list[dict[str, Any]]:
    if report.get("schema_version") != INPUT_SCHEMA:
        raise ValueError(f"Expected {INPUT_SCHEMA} timeline.")
    rows = report.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Timeline has no frames.")
    if report.get("joined_frame_count") != len(rows):
        raise ValueError("Timeline frame count mismatch.")
    prev = -1
    cleaned = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Timeline frame is not an object.")
        frame = row.get("world_frame")
        if isinstance(frame, bool) or not isinstance(frame, int) or frame <= prev:
            raise ValueError("World frames must be strictly increasing integers.")
        prev = frame
        risk = row.get("original_yolo_rule_risk")
        if risk not in RISKS:
            raise ValueError(f"Unexpected original YOLO risk: {risk!r}")
        ego = number(row.get("ego_speed_kmh"), "ego_speed_kmh")
        lead = number(row.get("lead_speed_kmh"), "lead_speed_kmh")
        if any(s is not None and not 0 <= s <= 300 for s in (ego, lead)):
            raise ValueError("Speed outside [0,300] km/h.")
        sep = number(
            row.get("origin_longitudinal_separation_m"),
            "origin_longitudinal_separation_m",
        )
        count = row.get("yolo_detection_count")
        if count is not None and (
            isinstance(count, bool) or not isinstance(count, int) or count < 0
        ):
            raise ValueError("Invalid YOLO detection count.")
        names = row.get("yolo_detected_classes")
        if names is not None and (
            not isinstance(names, list) or not all(isinstance(n, str) for n in names)
        ):
            raise ValueError("Invalid YOLO class names.")
        if count is not None and names is not None and len(names) != count:
            raise ValueError("YOLO class count mismatch.")
        cleaned.append({
            "world_frame": frame,
            "ego_speed_kmh": ego,
            "lead_speed_kmh": lead,
            "origin_longitudinal_separation_m": sep,
            "original_yolo_rule_risk": risk,
            "yolo_detection_count": count,
            "yolo_detected_classes": names,
        })
    return cleaned


def analyze(timeline: dict[str, Any], *, speed_diff_kmh: float = 2.0,
            sep_drop_m: float = 0.01) -> dict[str, Any]:
    """A review flag requires speed difference + shrinking actor-origin separation."""
    sd = number(speed_diff_kmh, "speed_diff_kmh")
    threshold = number(sep_drop_m, "sep_drop_m")
    if sd is None or not 0 < sd <= 100:
        raise ValueError("speed_diff_kmh must be in (0,100].")
    if threshold is None or not 0 <= threshold <= 5:
        raise ValueError("sep_drop_m must be in [0,5].")
    rows = validate_timeline(timeline)
    output: list[dict[str, Any]] = []
    previous = None
    for row in rows:
        ego, lead = row["ego_speed_kmh"], row["lead_speed_kmh"]
        sep = row["origin_longitudinal_separation_m"]
        diff = round(ego - lead, 6) if ego is not None and lead is not None else None
        prior = previous["origin_longitudinal_separation_m"] if previous else None
        delta = round(sep - prior, 6) if sep is not None and prior is not None else None
        if diff is None or delta is None:
            context = "missing_audit_telemetry"
        elif diff >= sd and delta < 0 and -delta >= threshold:
            context = "closing_context"
        else:
            context = "criteria_not_both_met"
        flag = context == "closing_context" and row["original_yolo_rule_risk"] == "low"
        output.append({
            "world_frame": row["world_frame"],
            "ego_speed_kmh": ego,
            "lead_speed_kmh": lead,
            "ego_minus_lead_speed_kmh": diff,
            "origin_longitudinal_separation_m": sep,
            "origin_sep_change_from_prior_saved_frame_m": delta,
            "original_yolo_rule_risk": row["original_yolo_rule_risk"],
            "yolo_detection_count": row["yolo_detection_count"],
            "yolo_detected_classes": row["yolo_detected_classes"],
            "review_context": context,
            "low_with_closing_context_flag": flag,
        })
        previous = row

    flags = [r for r in output if r["low_with_closing_context_flag"]]
    context_rows = [r for r in output if r["review_context"] == "closing_context"]
    return {
        "schema_version": OUTPUT_SCHEMA,
        "world_frame_count": len(output),
        "first_world_frame": output[0]["world_frame"],
        "last_world_frame": output[-1]["world_frame"],
        "thresholds_for_review_only": {
            "ego_minus_lead_speed_at_least_kmh": sd,
            "origin_sep_drop_at_least_m": threshold,
        },
        "original_rule_risk_counts": dict(sorted(Counter(
            str(r["original_yolo_rule_risk"]) for r in output
        ).items())),
        "closing_context_frame_count": len(context_rows),
        "low_with_closing_context_count": len(flags),
        "flagged_world_frames": [r["world_frame"] for r in flags],
        "low_fraction_among_context_frames_not_false_negative_rate": (
            round(len(flags) / len(context_rows), 4) if context_rows else None
        ),
        "rows": output,
        "carla_truth_role": "independent_posthoc_audit_only",
        "is_safety_ground_truth": False,
        "limitations": [
            "Flags are diagnostic review cases, not safety labels or false negatives.",
            "CARLA actor speeds and longitudinal origin separation are audit-only; "
            "they have not been provided to YOLO, VLM or Qwen.",
            "Scalar speed difference is not a calibrated relative longitudinal velocity.",
            "Actor-origin longitudinal separation is not the bumper-to-bumper clearance.",
            "Saved frames do not provide a validated TTC, collision probability or safe stopping distance.",
            "Unflagged frames are not automatically safe. Thresholds are experimental.",
            "No simulator, model, or vehicle controller was run.",
        ],
    }


def svg_plot(rows: list[dict[str, Any]], key: str, label: str, unit: str) -> str:
    valid = [r for r in rows if r[key] is not None]
    if not valid:
        return "<p>No measurement available.</p>"
    width, height = 860, 255
    left, right, top, bottom = 78, 30, 20, 47
    minframe, maxframe = rows[0]["world_frame"], rows[-1]["world_frame"]
    minimum = min(float(r[key]) for r in valid)
    maximum = max(float(r[key]) for r in valid)
    pad = max((maximum - minimum) * 0.08, .1)
    minimum -= pad
    maximum += pad
    x = lambda frame: left + (frame-minframe)/(maxframe-minframe or 1)*(width-left-right)
    y = lambda value: top+(maximum-value)/(maximum-minimum)*(height-top-bottom)
    elements = [f'<svg viewBox="0 0 {width} {height}" role="img" '
                f'aria-label="{escape(label)}" xmlns="http://www.w3.org/2000/svg">']
    for i in range(5):
        v = minimum + (maximum-minimum)*i/4
        yy = y(v)
        elements.append(f'<line x1="{left}" y1="{yy:.1f}" x2="{width-right}" '
                        f'y2="{yy:.1f}" stroke="#dbe7e2" stroke-width="1"/>')
        elements.append(f'<text x="{left-10}" y="{yy+4:.1f}" text-anchor="end" '
                        f'font-size="13" fill="#526861">{v:.1f}</text>')
    elements.append(f'<text x="{left}" y="{height-13}" font-size="13">{minframe}</text>')
    elements.append(f'<text x="{width-right}" y="{height-13}" font-size="13" '
                    f'text-anchor="end">{maxframe}</text>')
    points = " ".join(
        f'{x(r["world_frame"]):.2f},{y(float(r[key])):.2f}' for r in valid
    )
    elements.append(f'<polyline points="{points}" fill="none" stroke="#27776d" '
                    'stroke-width="3" stroke-linejoin="round"/>')
    for row in valid:
        marker = "#be6645" if row["low_with_closing_context_flag"] else "#27776d"
        frame, value = row["world_frame"], float(row[key])
        elements.append(f'<circle cx="{x(frame):.2f}" cy="{y(value):.2f}" '
                        f'r="{"5" if row["low_with_closing_context_flag"] else "3"}" '
                        f'fill="{marker}"><title>Frame {frame}: {value:.3f} {escape(unit)}'
                        '</title></circle>')
    elements.append("</svg>")
    return "".join(elements)


def html_report(data: dict[str, Any]) -> str:
    rows = data["rows"]
    graph_sep = svg_plot(rows, "origin_longitudinal_separation_m",
                         "Origin longitudinal separation", "m")
    graph_lead = svg_plot(rows, "lead_speed_kmh", "Lead vehicle speed", "km/h")
    table_rows = []
    fmt = lambda val: "—" if val is None else f"{val:.2f}"
    for row in rows:
        flag = bool(row["low_with_closing_context_flag"])
        table_rows.append(
            f'<tr class="{"flag" if flag else ""}"><td>{row["world_frame"]}</td>'
            f'<td>{fmt(row["ego_speed_kmh"])}</td>'
            f'<td>{fmt(row["lead_speed_kmh"])}</td>'
            f'<td>{fmt(row["origin_longitudinal_separation_m"])}</td>'
            f'<td>{escape(str(row["original_yolo_rule_risk"]))}</td>'
            f'<td>{escape(str(row["yolo_detection_count"]))}</td>'
            f'<td>{"Review" if flag else "—"}</td></tr>'
        )
    notes = "".join(f"<li>{escape(note)}</li>" for note in data["limitations"])
    return """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport"
content="width=device-width,initial-scale=1">
<title>Lead Brake / Temporal Evidence Review</title>
<style>
:root{font-family:Segoe UI,Arial,sans-serif;background:#f6f5f0;color:#233d42}
*{box-sizing:border-box}body{margin:auto;max-width:1100px;padding:26px 20px 75px}
small{color:#5a706a}.cap{letter-spacing:.14em;text-transform:uppercase;font-size:12px;
font-weight:700;color:#327468}h1{font-size:34px;margin:8px 0 13px}
.card{background:white;border-radius:16px;border:1px solid #dde6e1;padding:22px;margin:20px 0}
.stats{display:flex;flex-wrap:wrap;gap:10px}.stat{flex:1;min-width:170px;background:#e9f0ec;
border-radius:12px;padding:15px}.stat strong{display:block;font-size:27px;margin-top:5px}
svg{width:100%;height:auto}h2{font-size:20px;margin:4px 0 12px}
table{width:100%;border-collapse:collapse;font-size:13px;font-variant-numeric:tabular-nums}
th,td{padding:9px;text-align:right;border-bottom:1px solid #e6ece9}
th:first-child,td:first-child{text-align:left}th{background:#e9f0ec}
tr.flag{background:#fcf0e7}.scroll{overflow-x:auto}
p,li{line-height:1.55}.muted{font-size:13px;color:#526e68}
</style></head><body>
<p class="cap">CARLA multimodal research · Phase 4.1</p>
<h1>Lead-braking temporal review</h1>
<p>Offline audit using existing simulator trajectory and YOLO outputs.
Orange marks indicate review criteria, NOT safety ground truth, false negatives or TTC.</p>
<div class="stats">
<div class="stat">Saved frames<strong>""" + str(data["world_frame_count"]) + """</strong></div>
<div class="stat">Closing-context frames<strong>""" + str(data["closing_context_frame_count"]) + """</strong></div>
<div class="stat">Low + closing-context flags<strong>""" + str(data["low_with_closing_context_count"]) + """</strong></div>
</div>
<section class="card"><h2>Actor-origin longitudinal separation</h2>
<p class="muted">Meters, NOT bumper clearance. Orange markers are audit flags.</p>
""" + graph_sep + """</section>
<section class="card"><h2>Lead vehicle speed</h2>
<p class="muted">km/h. Speed magnitudes; no acceleration or TTC is estimated.</p>
""" + graph_lead + """</section>
<section class="card"><h2>Frame-by-frame evidence</h2>
<p class="muted">Flag criterion: ego-minus-lead speed ≥ """ + str(
    data["thresholds_for_review_only"]["ego_minus_lead_speed_at_least_kmh"]
) + """ km/h, consecutive actor-origin separation drop ≥ """ + str(
    data["thresholds_for_review_only"]["origin_sep_drop_at_least_m"]
) + """ m, original rule risk = low.</p><div class="scroll"><table>
<thead><tr><th>Frame</th><th>Ego km/h</th><th>Lead km/h</th>
<th>Origin sep m</th><th>YOLO risk</th><th>Detections</th><th>Audit</th>
</tr></thead><tbody>""" + "".join(table_rows) + """</tbody></table></div></section>
<section class="card"><h2>Essential limitations</h2><ul>""" + notes + """</ul></section>
</body></html>"""


def csv_report(rows: list[dict[str, Any]]) -> str:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=CSV_KEYS, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        item = dict(row)
        if item["yolo_detected_classes"] is not None:
            item["yolo_detected_classes"] = json.dumps(
                item["yolo_detected_classes"], ensure_ascii=False
            )
        writer.writerow(item)
    return buf.getvalue()


def create_file(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as f:
        f.write(content)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeline-report", required=True, type=Path)
    parser.add_argument("--speed-diff-kmh", type=float, default=2.0)
    parser.add_argument("--sep-drop-m", type=float, default=0.01)
    args = parser.parse_args()
    input_path = args.timeline_report.expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Timeline file not found: {input_path}")
    result = analyze(
        load_json(input_path),
        speed_diff_kmh=args.speed_diff_kmh,
        sep_drop_m=args.sep_drop_m,
    )
    result["source_timeline_report"] = str(input_path)
    result["created_at"] = datetime.now().astimezone().isoformat(timespec="seconds")

    directory = Path(__file__).resolve().parent.parent / "evidence" / "lead_brake_diagnostics"
    stem = f"diagnostics_{datetime.now():%Y%m%d_%H%M%S_%f}"
    outputs = [
        (directory / (stem + ".json"), json.dumps(result, ensure_ascii=False, indent=2)),
        (directory / (stem + ".csv"), csv_report(result["rows"])),
        (directory / (stem + ".html"), html_report(result)),
    ]
    for path, content in outputs:
        create_file(path, content)

    print("=== Phase 4.1: lead-braking offline diagnostic ===")
    print("Frames:", result["world_frame_count"])
    print("Original risk counts:", result["original_rule_risk_counts"])
    print("Closing-context frames:", result["closing_context_frame_count"])
    print("Low-risk review flags:", result["low_with_closing_context_count"])
    print("Flagged frames:", result["flagged_world_frames"])
    print("IMPORTANT: a review flag is not a safety ground truth / false negative.")
    for path, _ in outputs:
        print(path.suffix.upper().strip(".") + ":", path)
    print("PASS: offline reports written; no model inference or simulator rerun")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
