"""Bridge existing offline YOLO/VLM fusion to the Qwen risk-input contract.

This is a no-GPU preflight: it does not run CARLA, YOLO, VLM, Qwen, or control vehicles.
It never edits original source reports or previously generated results.
Run: python -m scripts.scene_pipeline_bridge [--alignment-report PATH]
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

from app.scene_fusion import build_truth_audit, fuse_visual_evidence, load_json
from app.scene_llm_risk import build_model_input


def select_alignment(project: Path, specified: str | None) -> Path:
    if specified:
        path = Path(specified).expanduser()
        if not path.is_absolute():
            path = project / path
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Alignment report does not exist: {path}")
        return path

    evidence = project / "evidence"
    if not evidence.is_dir():
        raise FileNotFoundError("evidence/ directory is missing.")
    candidates = [p for p in evidence.rglob("aligned_frame_*.json") if p.is_file()]
    if not candidates:
        raise FileNotFoundError(
            "No aligned_frame_*.json found under evidence/. "
            "Specify an existing alignment report with --alignment-report."
        )
    return max(candidates, key=lambda p: p.stat().st_mtime_ns).resolve()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--alignment-report",
        help="Existing aligned_frame_*.json (relative to project root or absolute). "
             "If omitted, use the most recently modified matching file in evidence/.",
    )
    args = parser.parse_args()
    project = Path(__file__).resolve().parent.parent
    source = select_alignment(project, args.alignment_report)

    # Reuse the two existing modules, without duplicating their parsing logic.
    aligned = load_json(source)
    visual = fuse_visual_evidence(aligned)
    audit = build_truth_audit(aligned, visual)
    report = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "input_alignment_report": str(source),
        **visual,
        "audit_only": audit,
    }

    # Verify the EXACT hand-off that the existing Qwen module will consume.
    # build_model_input() deliberately selects only visual evidence fields.
    model_payload, evidence_ids = build_model_input(report, ego_speed_kmh=None)

    if report["schema_version"] != "scene_fusion_evidence.v1":
        raise ValueError("Unexpected fusion schema_version.")
    if report["provenance"]["simulator_truth_used_in_fusion"] is not False:
        raise ValueError("Fusion provenance does not affirm truth isolation.")
    if report["audit_only"]["used_for_visual_fusion"] is not False:
        raise ValueError("Truth audit claims it was used for visual fusion.")
    if model_payload["world_frame"] != report["world_frame"]:
        raise ValueError("World frame changed across fusion/LLM hand-off.")
    forbidden_keys = {
        "audit_only", "carla_truth_for_human_audit_only", "simulator_truth",
        "risk", "original_yolo_rule_baseline",
    }
    if forbidden_keys.intersection(model_payload):
        raise ValueError("Forbidden evaluation/baseline keys in Qwen input.")

    output_dir = project / "evidence" / "scene_fusion"
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output = output_dir / f"fusion_frame_{report['world_frame']:08d}_{stamp}.json"
    # Exclusive creation: never overwrite any existing evidence.
    with output.open("x", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("=== Scene fusion -> Qwen input contract (NO MODEL INFERENCE) ===")
    print("Alignment:", source)
    print("World frame:", report["world_frame"])
    print("YOLO detections:", len(model_payload["YOLO_predictions"]))
    print("VLM claims:", len(model_payload["VLM_claims_unverified"]))
    print("Available evidence IDs:", len(evidence_ids))
    print("CARLA truth kept separate: True")
    print("Original YOLO risk excluded from Qwen input: True")
    print("Fusion report:", output)
    print("PASS: alignment -> fusion -> Qwen model-input contract")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
