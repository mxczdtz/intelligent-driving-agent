r"""
Phase 3.6: Build a citation-ready visual evidence packet from Phase 3.3.

This is a new *offline* input-structuring step, NOT a risk predictor.
It reads an existing scene_fusion evidence report and outputs:
  - evidence IDs for each YOLO detection and extracted VLM claim
  - source-linked cross-model comparisons, with uncertainty qualifiers
  - independently cited limitations
  - a bounded, claim-by-claim packet for a subsequent LLM experiment

CARLA audit truth and original rule risk are intentionally NEVER read.
Neither model runs; no network, package installation or vehicle control.

PowerShell example, project root:
  & $PY -m app.scene_grounded_packet `
      --fusion-report ".\evidence\scene_fusion\fusion_frame_00003488_20261116_162158.json" `
      --ego-speed-kmh 10.795

Output: evidence/scene_grounded_packet/grounded_frame_*.json
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import json
import math
from pathlib import Path
import sys
from typing import Any

SCHEMA_VERSION = "scene_grounded_packet.v1"
INPUT_SCHEMA_VERSION = "scene_fusion_evidence.v1"
VLM_KINDS = {
    "pedestrian_count", "pedestrian_sidewalk_location",
    "car_color", "traffic_light_colors", "vehicle_motion_language",
}
KNOWN_COMPARISONS = {"pedestrian_count", "traffic_lights"}
KNOWN_STATUSES = {
    "same_count_not_identity_verified",
    "count_disagreement_needs_image_review",
    "unverified_by_yolo",
    "light_color_still_unverified",
}


def json_object(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as f:
        content = json.load(f)
    if not isinstance(content, dict):
        raise ValueError("Expected a JSON object.")
    return content


def clean_float(value: Any, name: str, min_: float, max_: float) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} cannot be bool.")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric.") from exc
    if not math.isfinite(result) or not min_ <= result <= max_:
        raise ValueError(f"{name} out of allowed range [{min_}, {max_}].")
    return result


def bounded_text(value: Any, name: str, length: int = 180) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string.")
    if len(value) > length:
        raise ValueError(f"{name} is longer than {length}.")
    return value.strip()


def build_packet(fusion: dict[str, Any], ego_speed_kmh: float | None = None) -> dict[str, Any]:
    """Only whitelisted visual fields are accessed. No CARLA audit truth or risk."""
    if fusion.get("schema_version") != INPUT_SCHEMA_VERSION:
        raise ValueError(f"Expected {INPUT_SCHEMA_VERSION} fusion report.")
    frame = fusion.get("world_frame")
    if isinstance(frame, bool) or not isinstance(frame, int) or frame < 0:
        raise ValueError("Invalid world_frame.")

    yolo = fusion.get("yolo_visual_evidence")
    vlm = fusion.get("vlm_text_evidence")
    checks = fusion.get("cross_model_checks")
    if not isinstance(yolo, dict) or not isinstance(vlm, dict) or not isinstance(checks, list):
        raise ValueError("Fusion report missing YOLO/VLM/cross-check sections.")

    detections = yolo.get("detections")
    if not isinstance(detections, list) or len(detections) > 100:
        raise ValueError("Invalid YOLO detection list.")
    claims = vlm.get("claims_unverified")
    if not isinstance(claims, list) or len(claims) > 100:
        raise ValueError("Invalid VLM claim list.")
    if len(checks) > 30:
        raise ValueError("Too many cross-check records.")

    evidence: list[dict[str, Any]] = []
    yolo_ids: dict[str, list[str]] = {}
    for i, item in enumerate(detections):
        if not isinstance(item, dict):
            raise ValueError(f"YOLO detection #{i} is invalid.")
        label = bounded_text(item.get("class"), f"YOLO label #{i}", 60).lower()
        confidence = item.get("confidence")
        if confidence is not None:
            confidence = clean_float(confidence, f"confidence #{i}", 0, 1)
        bbox = item.get("bbox_xyxy_from_alignment")
        if bbox is not None:
            if not isinstance(bbox, list) or len(bbox) != 4:
                raise ValueError(f"Invalid bbox #{i}; expected xyxy.")
            bbox = [
                clean_float(value, f"bbox value #{i}", -1e8, 1e8)
                for value in bbox
            ]
            if bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:
                raise ValueError(f"Invalid bbox ordering #{i}.")

        eid = f"yolo:{i}"
        yolo_ids.setdefault(label, []).append(eid)
        evidence.append({
            "id": eid,
            "kind": "detector_output",
            "source": "YOLO",
            "observation": {"predicted_class": label,
                            "confidence": confidence,
                            "bbox_xyxy_from_alignment": bbox},
            "epistemic_status": "unverified_prediction",
        })

    derived_counts = dict(sorted(Counter(
        e["observation"]["predicted_class"] for e in evidence
    ).items()))
    expected_counts = yolo.get("class_counts")
    if not isinstance(expected_counts, dict) or expected_counts != derived_counts:
        raise ValueError("YOLO class_counts do not match detections; refusing to guess.")

    vlm_ids: dict[str, list[str]] = {}
    for i, claim in enumerate(claims):
        if not isinstance(claim, dict):
            raise ValueError(f"VLM claim #{i} is invalid.")
        kind = claim.get("type")
        if kind not in VLM_KINDS:
            raise ValueError(f"Unexpected VLM claim type: {kind!r}")
        if claim.get("verification") != "unverified":
            raise ValueError("VLM output must remain explicitly unverified.")
        value = claim.get("value")
        if isinstance(value, bool):
            pass
        elif isinstance(value, int) and 0 <= value <= 100:
            pass
        elif isinstance(value, str):
            value = bounded_text(value, f"VLM claim #{i}", 100)
        elif isinstance(value, list) and len(value) <= 5 and all(
            isinstance(x, str) and 1 <= len(x) <= 30 for x in value
        ):
            value = list(value)
        else:
            raise ValueError(f"Unexpected VLM value for claim #{i}.")
        vid = f"vlm:{i}"
        vlm_ids.setdefault(kind, []).append(vid)
        evidence.append({
            "id": vid,
            "kind": "language_claim",
            "source": "VLM",
            "observation": {"claim_type": kind, "claim_value": value},
            "epistemic_status": "unverified_generated_text",
        })

    by_id = {item["id"]: item for item in evidence}
    candidates: list[dict[str, Any]] = []
    forbidden_assertions: list[dict[str, Any]] = []
    warnings: list[str] = []

    for check in checks:
        if not isinstance(check, dict):
            raise ValueError("Cross-check must be an object.")
        comparison = check.get("comparison")
        status = check.get("status")
        if comparison not in KNOWN_COMPARISONS or status not in KNOWN_STATUSES:
            raise ValueError(f"Unsupported comparison/status: {comparison}/{status}")
        cross_id = f"cross:{comparison}"
        if cross_id in by_id:
            raise ValueError("Duplicate cross-check comparison; refusing to overwrite.")

        if comparison == "pedestrian_count":
            yolo_persons = yolo_ids.get("person", [])
            vlm_count = vlm_ids.get("pedestrian_count", [])
            if not vlm_count:
                raise ValueError("Pedestrian comparison has no VLM count claim.")
            observed = by_id[vlm_count[0]]["observation"]["claim_value"]
            if isinstance(observed, bool) or not isinstance(observed, int):
                raise ValueError("Expected integer pedestrian count claim.")
            expected_status = (
                "same_count_not_identity_verified"
                if len(yolo_persons) == observed
                else "count_disagreement_needs_image_review"
            )
            if status != expected_status:
                raise ValueError("Pedestrian comparison disagrees with underlying counts.")
            dependencies = [*yolo_persons, vlm_count[0]]
            observation = {
                "yolo_person_count": len(yolo_persons),
                "vlm_claimed_person_count": observed,
                "comparison_status": status,
            }
            qualifiers = [
                "Only the reported counts are compared; neither count is verified ground truth.",
                "Matching counts do not establish matching identities, lanes, or distances.",
            ]
            if status == "same_count_not_identity_verified":
                candidates.append({
                    "claim_id": "candidate:pedestrian_count_agreement",
                    "text": (
                        f"YOLO 预测 {len(yolo_persons)} 个 person，"
                        f"VLM 文本提及 {observed} 名行人；两者仅在报告数量上相同。"
                    ),
                    "support_refs": [cross_id, *dependencies],
                    "status": "model_report_agreement_not_scene_truth",
                })
                forbidden_assertions.append({
                    "text": "YOLO 与 VLM 的行人数量已证实不一致",
                    "reason": "The saved pedestrian-count comparison indicates agreement.",
                })
            else:
                candidates.append({
                    "claim_id": "candidate:pedestrian_count_difference",
                    "text": (
                        f"YOLO 预测 {len(yolo_persons)} 个 person，"
                        f"VLM 文本提及 {observed} 名行人；报告数量存在差异，需人工核验。"
                    ),
                    "support_refs": [cross_id, *dependencies],
                    "status": "model_report_count_difference_unverified",
                })

        else:  # traffic_lights
            lights = yolo_ids.get("traffic light", [])
            light_claims = vlm_ids.get("traffic_light_colors", [])
            if not light_claims:
                raise ValueError("Traffic-light comparison requires a VLM color claim.")
            expected_status = (
                "unverified_by_yolo" if not lights else "light_color_still_unverified"
            )
            if status != expected_status:
                raise ValueError("Traffic-light status conflicts with detector records.")
            dependencies = [*lights, light_claims[0]]
            observation = {
                "yolo_traffic_light_count": len(lights),
                "vlm_claimed_colors_unverified": (
                    by_id[light_claims[0]]["observation"]["claim_value"]
                ),
                "comparison_status": status,
            }
            qualifiers = [
                "YOLO classification cannot verify traffic-light color.",
                "Failure to detect a light does not establish its absence.",
                "VLM-generated colors are unverified against the saved image.",
            ]
            candidates.append({
                "claim_id": "candidate:traffic_light_uncertainty",
                "text": (
                    f"YOLO 预测 {len(lights)} 个 traffic light；"
                    "VLM 提到交通灯颜色，但颜色与信号灯关联尚未核验。"
                ),
                "support_refs": [cross_id, *dependencies],
                "status": "unverified_visual_color_claim",
            })
            forbidden_assertions.append({
                "text": "仅凭 VLM 的颜色描述确认当前真实交通灯状态",
                "reason": "No verified signal color is provided.",
            })

        evidence.append({
            "id": cross_id,
            "kind": "cross_model_comparison",
            "source": "deterministic_comparison_of_model_outputs",
            "observation": observation,
            "dependencies": dependencies,
            "qualifiers": qualifiers,
            "epistemic_status": "comparison_not_independent_truth",
        })
        by_id[cross_id] = evidence[-1]

    if ego_speed_kmh is not None:
        speed = clean_float(ego_speed_kmh, "ego_speed_kmh", 0, 300)
        evidence.append({
            "id": "ego:speed",
            "kind": "telemetry",
            "source": "separately_supplied_previously_logged_ego_speed",
            "observation": {"speed_kmh": round(speed, 3)},
            "epistemic_status": "logged_value_provided_by_operator",
        })

    # Build explicit negative capability evidence from known input limitations.
    evidence.append({
        "id": "limit:single_frame",
        "kind": "measurement_limitation",
        "source": "visual_input_scope",
        "observation": "One RGB frame cannot establish motion direction or temporal overlap.",
        "epistemic_status": "input_scope_limitation",
    })
    evidence.append({
        "id": "limit:unknown_geometry",
        "kind": "measurement_limitation",
        "source": "absence_of_verified_3d_geometry_in_visual_evidence",
        "observation": "No verified lane relation, bumper gap, relative speed or TTC is supplied.",
        "epistemic_status": "input_scope_limitation",
    })

    warnings.extend([
        "Do not treat the VLM's pedestrian-sidewalk claim as verified lane geometry.",
        "Do not infer traffic-light state from COCO YOLO traffic-light category.",
        "Do not call a low baseline risk safe without independent risk ground truth.",
        "Candidates state facts about model outputs, not facts about the road scene.",
    ])

    packet = {
        "schema_version": SCHEMA_VERSION,
        "world_frame": frame,
        "evidence": evidence,
        "candidate_statements_for_human_review": candidates,
        "statements_not_justified_by_visual_evidence": forbidden_assertions,
        "warnings": warnings,
        "followup_llm_contract": {
            "require_each_factor_to_have_its_own_source_refs": True,
            "allowable_ref_ids": [item["id"] for item in evidence],
            "no_unsupported_model_conflict_as_fact": True,
            "new_risk_level": None,
            "no_vehicle_control": True,
        },
        "source_isolation": {
            "CARLA_audit_truth_used": False,
            "original_YOLO_rule_risk_used": False,
            "Qwen_generated_claims_used": False,
        },
    }
    all_ids = set(packet["followup_llm_contract"]["allowable_ref_ids"])
    if len(all_ids) != len(evidence):
        raise ValueError("Evidence ID collision.")
    for candidate in candidates:
        if not set(candidate["support_refs"]) <= all_ids:
            raise ValueError("Candidate has invalid evidence references.")
    return packet


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fusion-report", type=Path, required=True)
    parser.add_argument("--ego-speed-kmh", type=float, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    fusion_path = args.fusion_report.expanduser().resolve()
    if not fusion_path.is_file():
        raise FileNotFoundError(f"Fusion report not found: {fusion_path}")
    data = json_object(fusion_path)
    packet = build_packet(data, args.ego_speed_kmh)

    project = Path(__file__).resolve().parent.parent
    destination = args.output or (
        project / "evidence" / "scene_grounded_packet" /
        f"grounded_frame_{packet['world_frame']:08d}_{datetime.now():%Y%m%d_%H%M%S_%f}.json"
    )
    destination = destination.expanduser().resolve()
    if destination == fusion_path or destination.exists():
        raise FileExistsError("Refusing to overwrite an existing report.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source_fusion_report": str(fusion_path),
        **packet,
    }
    destination.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("=== Phase 3.6: citation-ready evidence packet ===")
    print("Frame:", packet["world_frame"])
    print("Evidence records:", len(packet["evidence"]))
    print("Candidate statements:", len(packet["candidate_statements_for_human_review"]))
    for item in packet["candidate_statements_for_human_review"]:
        print("Candidate:", item["text"])
        print("  Support refs:", json.dumps(item["support_refs"], ensure_ascii=False))
        print("  Status:", item["status"])
    print("CARLA audit truth used: False")
    print("New risk assessment: NOT COMPUTED")
    print("Report:", destination)
    print("PASS: provenance packet saved; no models re-run")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, TypeError, KeyError, OSError, json.JSONDecodeError) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
