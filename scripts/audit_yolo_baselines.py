"""Read-only audit of existing CARLA YOLO baseline reports.

Run from project root:
    python -m scripts.audit_yolo_baselines

What this does:
* Verify each report's frame counts and world-frame uniqueness.
* Compare multiple YOLO reports from the same capture (do not double count).
* Summarize risk labels if the existing risk field is recognizable.
* Summarize per-frame YOLO inference_ms, NOT end-to-end latency.

What this does NOT do:
* Rerun models or access the GPU.
* Invent ground-truth danger labels or calculate risk accuracy.
* Alter any existing reports or simulator files.
"""
from __future__ import annotations

from collections import Counter
import argparse
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any

SCENE_DIRS = (
    "carla_lead_brake",
    "carla_pedestrian",
    "carla_traffic_light",
)
LEVELS = {"unknown", "low", "medium", "high"}


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return data


def risk_level(frame: dict[str, Any]) -> str | None:
    """Extract only explicit recognized labels; never infer risk from detections."""
    value: Any = frame.get("risk")
    if isinstance(value, dict):
        for field in ("risk_level", "level", "severity", "grade"):
            if field in value:
                value = value[field]
                break
        else:
            return None
    if isinstance(value, str) and value.strip().lower() in LEVELS:
        return value.strip().lower()
    return None


def valid_ms(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number >= 0 else None


def summarize_ms(values: list[float]) -> str:
    if not values:
        return "not available (numeric values not found)"
    ordered = sorted(values)
    p95 = ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]
    return (
        f"n={len(ordered)}, median={statistics.median(ordered):.2f} ms, "
        f"p95(nearest-rank)={p95:.2f} ms, "
        f"min={ordered[0]:.2f} ms, max={ordered[-1]:.2f} ms"
    )


def stable_repr(item: Any) -> str:
    return json.dumps(item, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def check_report(path: Path) -> dict[str, Any]:
    data = read_json(path)
    frames = data.get("frames")
    if not isinstance(frames, list):
        raise ValueError(f"Missing 'frames' list: {path}")
    if not all(isinstance(row, dict) for row in frames):
        raise ValueError(f"Non-object frame in: {path}")
    ids = [row.get("world_frame") for row in frames]
    if not all(isinstance(i, int) and not isinstance(i, bool) and i >= 0 for i in ids):
        raise ValueError(f"Invalid world_frame in: {path}")
    duplicates = [fid for fid, n in Counter(ids).items() if n > 1]
    declared = data.get("frame_count")
    mismatched = (
        not isinstance(declared, int)
        or isinstance(declared, bool)
        or declared != len(frames)
    )
    labels = [risk_level(f) for f in frames]
    latencies = [valid_ms(f.get("inference_ms")) for f in frames]
    numeric = [n for n in latencies if n is not None]
    ordered_ids = sorted(set(ids))
    gaps = [
        (a, b, b - a - 1)
        for a, b in zip(ordered_ids, ordered_ids[1:])
        if b - a > 1
    ]
    return {
        "path": path,
        "data": data,
        "frames": frames,
        "by_frame": {f["world_frame"]: f for f in frames},
        "declared": declared,
        "mismatched_count": mismatched,
        "duplicates": duplicates,
        "ordered_ids": ordered_ids,
        "gaps": gaps,
        "label_counts": Counter(label if label else "UNPARSED" for label in labels),
        "latencies": numeric,
        "missing_latency": len(frames) - len(numeric),
    }


def compare_reports(a: dict[str, Any], b: dict[str, Any]) -> None:
    first = a["by_frame"]
    second = b["by_frame"]
    matched = sorted(first.keys() & second.keys())
    only_first = sorted(first.keys() - second.keys())
    only_second = sorted(second.keys() - first.keys())
    same_images = sum(
        first[k].get("image_file") == second[k].get("image_file")
        for k in matched
    )
    detection_diff = [
        k for k in matched
        if stable_repr(first[k].get("detections")) != stable_repr(second[k].get("detections"))
    ]
    risk_diff = [
        k for k in matched
        if stable_repr(first[k].get("risk")) != stable_repr(second[k].get("risk"))
    ]
    latency_diff = sum(
        stable_repr(first[k].get("inference_ms"))
        != stable_repr(second[k].get("inference_ms"))
        for k in matched
    )
    print(f"  Compare: {a['path'].name} <-> {b['path'].name}")
    print(
        f"    Matched world frames={len(matched)}, "
        f"only A={len(only_first)}, only B={len(only_second)}"
    )
    print(
        f"    Matching image_file strings={same_images}/{len(matched)}; "
        f"detections differ={len(detection_diff)}; "
        f"risk fields differ={len(risk_diff)}; "
        f"inference_ms differ={latency_diff}"
    )
    if only_first or only_second:
        print(f"    Exclusive frame IDs (first 8): A={only_first[:8]}, B={only_second[:8]}")
    if detection_diff or risk_diff:
        print(
            "    Differing world frames (first 8): "
            f"detections={detection_diff[:8]}, risk={risk_diff[:8]}"
        )
    if len(matched) == len(first) == len(second):
        print(
            "    Frame sets: IDENTICAL. Treat these as repeated analyses "
            "of one capture, not independent frames."
        )
    else:
        print("    Frame sets: DIFFERENT. Review before selecting a formal baseline.")


def audit(root: Path) -> int:
    evidence = root / "evidence"
    all_capture_frames: set[tuple[str, str, int]] = set()
    print("=== CARLA YOLO BASELINE REPORT AUDIT (read-only) ===")
    print("Project:", root)
    print("No models loaded; no ground-truth danger labels inferred.")
    issues = 0
    for scene in SCENE_DIRS:
        print(f"\n========== {scene} ==========")
        folder = evidence / scene
        paths = sorted(folder.rglob("yolo_risk_*.json")) if folder.is_dir() else []
        if not paths:
            print("  NO YOLO REPORTS FOUND")
            issues += 1
            continue
        reports: list[dict[str, Any]] = []
        for path in paths:
            try:
                report = check_report(path)
            except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
                print(f"  ERROR: {path.name}: {type(exc).__name__}: {exc}")
                issues += 1
                continue
            reports.append(report)
            frames = report["frames"]
            all_capture_frames.update(
                (scene, str(path.parent.resolve()), fid)
                for fid in report["ordered_ids"]
            )
            frame_first = report["ordered_ids"][0] if report["ordered_ids"] else "none"
            frame_last = report["ordered_ids"][-1] if report["ordered_ids"] else "none"
            print(f"\n  Report: {path.name}")
            print(
                f"    Frames: declared={report['declared']}, "
                f"actual={len(frames)}, unique={len(report['ordered_ids'])}; "
                f"world-frame range={frame_first}..{frame_last}"
            )
            if report["mismatched_count"]:
                print("    WARNING: declared frame_count differs or is invalid")
                issues += 1
            if report["duplicates"]:
                print(f"    WARNING: duplicate world_frame IDs: {report['duplicates'][:8]}")
                issues += 1
            if report["gaps"]:
                skipped = sum(n for _, _, n in report["gaps"])
                print(
                    f"    World-frame ID gaps: {len(report['gaps'])} gaps, "
                    f"{skipped} unrepresented frame numbers"
                )
                print(f"    Gap examples (from, to, between): {report['gaps'][:6]}")
                print("    NOTE: gaps can arise from intentional frame sampling.")
            else:
                print("    World-frame IDs: consecutive")
            print(
                "    Risk labels: "
                + ", ".join(
                    f"{label}={count}"
                    for label, count in sorted(report["label_counts"].items())
                )
            )
            if report["label_counts"].get("UNPARSED"):
                first_unparsed = next(
                    (row.get("risk") for row in frames if risk_level(row) is None),
                    None,
                )
                if isinstance(first_unparsed, dict):
                    shape = "dict keys=" + repr(list(first_unparsed.keys()))
                else:
                    shape = f"type={type(first_unparsed).__name__}, sample={str(first_unparsed)[:160]}"
                print(f"    Unparsed risk shape: {shape}")
            print("    YOLO inference_ms: " + summarize_ms(report["latencies"]))
            if report["missing_latency"]:
                print(f"    Missing/non-numeric inference_ms: {report['missing_latency']}")
            if report["data"].get("timing_note"):
                print("    Timing note: " + str(report["data"]["timing_note"])[:220])
        for n, a in enumerate(reports):
            for b in reports[n + 1:]:
                if a["path"].parent == b["path"].parent:
                    print("\n  -- Two reports from the SAME capture directory --")
                    compare_reports(a, b)
                else:
                    print(
                        "  NOTE: reports belong to different capture directories; "
                        "world-frame IDs alone should not be used to deduplicate."
                    )
    print("\n========== AUDIT SUMMARY ==========")
    print("Distinct (scene, capture folder, world_frame) records:", len(all_capture_frames))
    print("Data validation issues:", issues)
    print("IMPORTANT: This is a baseline audit, NOT a danger-detection accuracy test.")
    print("Separate CARLA truth-derived reference labels are required for accuracy/miss rate.")
    return 1 if issues else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="Project directory containing evidence/ (default: parent of scripts/).",
    )
    args = parser.parse_args()
    root = args.project_root.expanduser().resolve()
    if not (root / "evidence").is_dir():
        print(f"ERROR: no evidence directory at {root}", file=sys.stderr)
        return 2
    return audit(root)


if __name__ == "__main__":
    raise SystemExit(main())
