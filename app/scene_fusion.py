"""
Phase 3.3 — structured evidence fusion (offline, no model inference).

Input: an existing evidence-alignment JSON produced by scene_fusion_preview.py.
Output: a versioned JSON with attributed YOLO and VLM claims, cross-checks,
uncertainties and a SEPARATE CARLA truth audit.

This module does not change the original YOLO risk baseline, does not call a
VLM/LLM, does not label a frame as dangerous, and does not control a vehicle.
Only the visual branches contribute to fuse_visual_evidence(); CARLA truth is
read exclusively by build_truth_audit() after the visual report is complete.

Run from project root:
    python -m app.scene_fusion --alignment-report "evidence/.../aligned_frame_00003488_....json"

Only Python standard library is required.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import json
import math
from pathlib import Path
import re
import sys
from typing import Any

SCHEMA_VERSION = "scene_fusion_evidence.v1"
RISK_LEVELS = {"unknown", "low", "medium", "high"}
WORD_COUNTS = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3,
    "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "both": 2,
}
COLOR_WORDS = ("red", "yellow", "green", "blue", "white", "black", "gray", "grey")
ALLOWED_CLAIMS = {
    "pedestrian_count", "pedestrian_sidewalk_location",
    "car_color", "traffic_light_colors", "vehicle_motion_language",
}


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as fp:
        data = json.load(fp)
    if not isinstance(data, dict):
        raise ValueError(f"Expected top-level JSON object: {path}")
    return data


def validated_detections(info: Any) -> list[dict[str, Any]]:
    if not isinstance(info, dict):
        raise ValueError("Missing yolo_detection_info object.")
    objects = info.get("objects")
    if not isinstance(objects, list):
        raise ValueError(
            "Alignment report has no usable YOLO objects list. "
            "Regenerate preview from the original report, rather than guessing classes."
        )
    expected = info.get("count")
    if not isinstance(expected, int) or isinstance(expected, bool) or expected < 0:
        raise ValueError("Invalid YOLO detection count.")
    if expected != len(objects):
        raise ValueError(
            f"YOLO count mismatch: count={expected}, objects={len(objects)}. "
            "Alignment summary may be incomplete; do not silently truncate it."
        )

    result: list[dict[str, Any]] = []
    for index, obj in enumerate(objects):
        if not isinstance(obj, dict):
            raise ValueError(f"YOLO detection #{index} is not an object.")
        name = obj.get("class")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"Missing class name in YOLO detection #{index}.")
        confidence = obj.get("confidence")
        if confidence is not None:
            try:
                confidence = float(confidence)
            except (ValueError, TypeError) as exc:
                raise ValueError(f"Invalid confidence at YOLO #{index}.") from exc
            if not math.isfinite(confidence) or not 0 <= confidence <= 1:
                raise ValueError(f"YOLO confidence outside [0,1] at #{index}.")

        bbox = obj.get("bbox")
        # A null in the alignment summary is NOT proof that the original YOLO
        # report had no bounding box; it may be a key-extraction limitation.
        if bbox is not None:
            if not isinstance(bbox, (tuple, list)) or len(bbox) != 4:
                raise ValueError(f"Unsupported bbox in alignment at #{index}.")
            try:
                bbox = [float(x) for x in bbox]
            except (ValueError, TypeError) as exc:
                raise ValueError(f"Invalid bbox at #{index}.") from exc
            if not all(math.isfinite(x) for x in bbox):
                raise ValueError(f"Non-finite bbox at #{index}.")
            if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                raise ValueError(f"Invalid xyxy bounding box at #{index}.")
        result.append({
            "class": name.strip().lower(),
            "confidence": confidence,
            "bbox_xyxy_from_alignment": bbox,
            "provenance": "YOLO original report via alignment summary",
        })
    return result


def sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def parse_number(text: str) -> int | None:
    value = text.lower()
    if value.isdecimal():
        return int(value)
    return WORD_COUNTS.get(value)


def extract_vlm_claims(description: str) -> list[dict[str, Any]]:
    """Rule-based extraction of what the VLM *said*, not visual verification."""
    claims: list[dict[str, Any]] = []

    def add(kind: str, value: Any, sentence: str) -> None:
        if kind not in ALLOWED_CLAIMS:
            raise AssertionError(kind)
        claims.append({
            "type": kind,
            "value": value,
            "source": "VLM generated text",
            "quoted_context": sentence,
            "verification": "unverified",
        })

    for sentence in sentences(description):
        lower = sentence.lower()

        # Count only explicitly specified persons; do not infer from "people".
        pedestrian = re.search(
            r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten|both)"
            r"\s+(?:\w+\s+){0,2}pedestrians?\b",
            lower,
        )
        if pedestrian:
            count = parse_number(pedestrian.group(1))
            if count is not None:
                add("pedestrian_count", count, sentence)
        elif re.search(r"\b(?:a|an)\s+pedestrian\b", lower):
            add("pedestrian_count", 1, sentence)

        if "pedestrian" in lower and re.search(r"\b(?:sidewalk|pavement)\b", lower):
            add("pedestrian_sidewalk_location", True, sentence)

        color_car = re.search(
            r"\b(" + "|".join(COLOR_WORDS) + r")\s+car\b",
            lower,
        )
        if color_car:
            add("car_color", color_car.group(1), sentence)

        if re.search(r"\btraffic\s+lights?\b", lower):
            # Colors are only attributed to a traffic light if mentioned in the
            # same sentence; still not treated as confirmed signal states.
            colors = [c for c in ("red", "yellow", "green")
                      if re.search(r"\b" + c + r"\b", lower)]
            if colors:
                add("traffic_light_colors", colors, sentence)

        if re.search(r"\b(?:car|vehicle)\b", lower) and re.search(
            r"\b(?:driving away|moving away|driving towards|moving towards)\b",
            lower,
        ):
            add("vehicle_motion_language", "motion_direction_asserted", sentence)

    return claims


def fuse_visual_evidence(aligned: dict[str, Any]) -> dict[str, Any]:
    """PURE fusion: deliberately never reads carla_truth_for_human_audit_only."""
    frame = aligned.get("world_frame")
    if not isinstance(frame, int) or isinstance(frame, bool) or frame < 0:
        raise ValueError("Missing or invalid aligned world_frame.")
    info = aligned.get("yolo_detection_info")
    detections = validated_detections(info)
    text = aligned.get("vlm_unverified_description")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Missing VLM description.")
    level = aligned.get("yolo_original_rule_risk")
    if level not in RISK_LEVELS:
        raise ValueError(f"Unknown original risk: {level!r}")

    claims = extract_vlm_claims(text)
    counts = dict(sorted(Counter(obj["class"] for obj in detections).items()))
    check_items: list[dict[str, Any]] = []
    uncertainties: list[str] = []

    vlm_person_counts = [c["value"] for c in claims if c["type"] == "pedestrian_count"]
    if vlm_person_counts:
        latest = vlm_person_counts[0]
        status = "same_count_not_identity_verified" if latest == counts.get("person", 0) \
            else "count_disagreement_needs_image_review"
        check_items.append({
            "comparison": "pedestrian_count",
            "yolo_count": counts.get("person", 0),
            "vlm_text_count": latest,
            "status": status,
        })
        if status.startswith("count_disagreement"):
            uncertainties.append("YOLO and VLM pedestrian counts differ; image review required.")

    if any(c["type"] == "traffic_light_colors" for c in claims):
        check_items.append({
            "comparison": "traffic_lights",
            "yolo_count": counts.get("traffic light", 0),
            "vlm_claimed_colors": next(
                c["value"] for c in claims if c["type"] == "traffic_light_colors"
            ),
            "status": "unverified_by_yolo" if counts.get("traffic light", 0) == 0
            else "light_color_still_unverified",
        })
        uncertainties.append(
            "Traffic-light colors are VLM text claims, not validated signal states."
        )

    if any(c["type"] == "vehicle_motion_language" for c in claims):
        uncertainties.append(
            "Direction of motion cannot be established from this single RGB frame alone."
        )
    if any(c["type"] == "pedestrian_sidewalk_location" for c in claims):
        uncertainties.append(
            "Sidewalk/lane location claimed by VLM has not been geometrically verified."
        )
    if any(obj["bbox_xyxy_from_alignment"] is None for obj in detections):
        uncertainties.append(
            "Some bounding boxes are unavailable in the alignment SUMMARY; "
            "original YOLO report may still contain coordinates."
        )

    uncertainties.extend([
        "Count agreement does not prove detection correctness or object identity.",
        "No calibrated spatial proximity, relative speed, or TTC is available here.",
        "No independent danger label is available; original low risk is not a safety guarantee.",
    ])

    return {
        "schema_version": SCHEMA_VERSION,
        "world_frame": frame,
        "image": aligned.get("image"),
        "provenance": {
            "yolo": aligned.get("sources", {}).get("yolo"),
            "vlm": aligned.get("sources", {}).get("vlm"),
            "simulator_truth_used_in_fusion": False,
        },
        "yolo_visual_evidence": {
            "class_counts": counts,
            "detections": detections,
        },
        "vlm_text_evidence": {
            "description_unverified": text,
            "claims_unverified": claims,
        },
        "cross_model_checks": check_items,
        "uncertainties": uncertainties,
        "risk": {
            "original_yolo_rule_baseline": level,
            "new_fused_risk": None,
            "reason": "Evidence alignment only; no new risk inference or safety validation.",
            "vehicle_control": None,
        },
    }


def build_truth_audit(aligned: dict[str, Any],
                      visual: dict[str, Any]) -> dict[str, Any]:
    """Post-hoc audit; its output must NEVER be fed into the visual fusion."""
    raw = aligned.get("carla_truth_for_human_audit_only")
    if not isinstance(raw, dict):
        raw = {}
    notes: list[str] = [
        "Simulator truth is for human evaluation ONLY, not a vision-model input.",
        "CARLA target actor may not correspond to every YOLO/VLM person mention.",
    ]
    sidewalk = any(
        c["type"] == "pedestrian_sidewalk_location"
        for c in visual["vlm_text_evidence"]["claims_unverified"]
    )
    if sidewalk and raw.get("walker_inside_lane_lateral_band") is True:
        notes.append(
            "Potential semantic mismatch: VLM mentions pedestrians on the sidewalk, "
            "while the tracked CARLA walker is within the lane lateral band. "
            "Verify the actual PNG and actor association before deciding accuracy."
        )
    return {
        "simulator_truth": raw,
        "posthoc_review_notes": notes,
        "used_for_visual_fusion": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alignment-report", required=True, type=Path,
                        help="The previously generated aligned_frame_*.json")
    parser.add_argument("--output", type=Path, default=None,
                        help="Optional output path (default evidence/scene_fusion)")
    args = parser.parse_args()
    input_path = args.alignment_report.resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    aligned = load_json(input_path)
    visual = fuse_visual_evidence(aligned)
    audit = build_truth_audit(aligned, visual)
    result = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "input_alignment_report": str(input_path),
        **visual,
        "audit_only": audit,
    }
    root = Path(__file__).resolve().parent.parent
    out = args.output or (
        root / "evidence" / "scene_fusion" /
        f"fusion_frame_{visual['world_frame']:08d}_{datetime.now():%Y%m%d_%H%M%S}.json"
    )
    out = out.resolve()
    # Do not overwrite either input file or an existing output report.
    if out == input_path or out.exists():
        raise FileExistsError(f"Refusing to overwrite existing report: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2),
                   encoding="utf-8")

    print("=== Phase 3.3: YOLO + VLM structured evidence (offline) ===")
    print("Frame:", visual["world_frame"])
    print("YOLO class counts:", json.dumps(
        visual["yolo_visual_evidence"]["class_counts"], ensure_ascii=False))
    print("Original YOLO rule risk:", visual["risk"]["original_yolo_rule_baseline"])
    print("VLM claims extracted:", len(visual["vlm_text_evidence"]["claims_unverified"]))
    for check in visual["cross_model_checks"]:
        print("Cross-check:", json.dumps(check, ensure_ascii=False))
    print("Uncertainty flags:", len(visual["uncertainties"]))
    for note in audit["posthoc_review_notes"]:
        print("Audit-only:", note)
    print("New fused risk: NOT COMPUTED")
    print("Report:", out)
    print("PASS: structured evidence created without re-running any model")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
