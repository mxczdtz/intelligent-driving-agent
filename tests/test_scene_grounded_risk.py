"""Tests for Phase 3.7 grounded risk selection. No Qwen or GPU required."""
from __future__ import annotations

import copy
import json
import unittest

from app.scene_grounded_packet import build_packet
from app.scene_grounded_risk import (
    assemble_decision, make_messages, packet_index,
    parse_or_error, parse_selection, repair_messages,
)


def fusion_fixture():
    """Consistent Phase 3.3 frame 3488, with the exact claim ordering used."""
    return {
        "schema_version": "scene_fusion_evidence.v1",
        "world_frame": 3488,
        "yolo_visual_evidence": {
            "class_counts": {"car": 1, "person": 2},
            "detections": [
                {"class": "person", "confidence": .776938,
                 "bbox_xyxy_from_alignment": None},
                {"class": "car", "confidence": .42803,
                 "bbox_xyxy_from_alignment": None},
                {"class": "person", "confidence": .384485,
                 "bbox_xyxy_from_alignment": None},
            ],
        },
        "vlm_text_evidence": {
            "description_unverified": "DO_NOT_SEND_RAW_VLM_PROSE_TO_QWEN",
            "claims_unverified": [
                {"type": "car_color", "value": "red", "verification": "unverified"},
                {"type": "vehicle_motion_language",
                 "value": "motion_direction_asserted", "verification": "unverified"},
                {"type": "pedestrian_count", "value": 2, "verification": "unverified"},
                {"type": "pedestrian_sidewalk_location", "value": True,
                 "verification": "unverified"},
                {"type": "traffic_light_colors", "value": ["red", "yellow"],
                 "verification": "unverified"},
            ],
        },
        "cross_model_checks": [
            {"comparison": "pedestrian_count",
             "status": "same_count_not_identity_verified"},
            {"comparison": "traffic_lights",
             "status": "unverified_by_yolo"},
        ],
        "uncertainties": [],
        "risk": {"original_yolo_rule_baseline": "low"},
        "audit_only": {"simulator_truth": {
            "private_marker": "CARLA_GROUND_TRUTH_NEVER_IN_PROMPT"}},
    }


def packet_fixture():
    return build_packet(fusion_fixture(), 10.795)


def selection(level="unknown", candidate=None, limits=None):
    return {
        "risk_level": level,
        "selected_claim_ids": (
            ["candidate:pedestrian_count_agreement"]
            if candidate is None else candidate
        ),
        "limitation_ids": (
            ["limit:single_frame", "limit:unknown_geometry"]
            if limits is None else limits
        ),
    }


class GroundedRiskTests(unittest.TestCase):
    def get_context(self):
        return packet_index(packet_fixture())

    def test_packet_contains_expected_candidates(self):
        payload, candidates, limits, evidence = self.get_context()
        self.assertEqual(payload["world_frame"], 3488)
        self.assertEqual(payload["ego_speed_kmh_logged"], 10.795)
        self.assertEqual(len(candidates), 2)
        self.assertEqual(len(limits), 2)
        self.assertEqual(candidates["candidate:pedestrian_count_agreement"]["support_refs"],
                         ["cross:pedestrian_count", "yolo:0", "yolo:2", "vlm:2"])
        self.assertIn("cross:pedestrian_count", evidence)

    def test_no_truth_or_prior_qwen_or_original_risk_in_prompt(self):
        packet = packet_fixture()
        packet["prior_qwen_decision"] = "DO_NOT_ECHO_PRIOR_QWEN"
        packet["carla_truth"] = "DO_NOT_ECHO_CARLA"
        payload, _, _, _ = packet_index(packet)
        prompt = json.dumps(make_messages(payload), ensure_ascii=False)
        self.assertNotIn("DO_NOT_ECHO_CARLA", prompt)
        self.assertNotIn("DO_NOT_ECHO_PRIOR_QWEN", prompt)
        self.assertNotIn("original_yolo_rule_baseline", prompt)
        self.assertNotIn("DO_NOT_SEND_RAW_VLM_PROSE_TO_QWEN", prompt)

    def test_valid_model_selection_yields_real_factor_source_refs(self):
        payload, candidates, limits, _ = self.get_context()
        picked = parse_selection(
            json.dumps(selection(), ensure_ascii=False), candidates, limits
        )
        decision, flags = assemble_decision(picked, candidates, limits)
        self.assertEqual(decision["risk_level"], "unknown")
        self.assertEqual(
            decision["factors"][0]["source_refs"],
            ["cross:pedestrian_count", "yolo:0", "yolo:2", "vlm:2"],
        )
        self.assertIn("仅在报告数量上相同", decision["factors"][0]["statement"])

    def test_invalid_candidate_id_rejected(self):
        _, candidates, limits, _ = self.get_context()
        bad = selection(candidate=["candidate:imaginary_high_risk"])
        with self.assertRaisesRegex(ValueError, "not found"):
            parse_selection(json.dumps(bad), candidates, limits)

    def test_invalid_limitation_id_rejected(self):
        _, candidates, limits, _ = self.get_context()
        bad = selection(limits=["limit:CARLA_actor_truth"])
        with self.assertRaisesRegex(ValueError, "not found"):
            parse_selection(json.dumps(bad), candidates, limits)

    def test_missing_fields_and_extra_factors_rejected(self):
        _, candidates, limits, _ = self.get_context()
        bad = selection()
        bad["new_factor"] = "There is a confirmed collision"
        with self.assertRaisesRegex(ValueError, "exactly"):
            parse_selection(json.dumps(bad), candidates, limits)
        bad.pop("new_factor")
        bad.pop("limitation_ids")
        with self.assertRaisesRegex(ValueError, "exactly"):
            parse_selection(json.dumps(bad), candidates, limits)

    def test_more_than_two_claim_ids_rejected(self):
        _, candidates, limits, _ = self.get_context()
        bad = selection(candidate=[
            "candidate:pedestrian_count_agreement",
            "candidate:traffic_light_uncertainty",
            "candidate:pedestrian_count_agreement",
        ])
        with self.assertRaisesRegex(ValueError, "0 to 2"):
            parse_selection(json.dumps(bad), candidates, limits)

    def test_duplicate_id_and_duplicate_json_key_rejected(self):
        _, candidates, limits, _ = self.get_context()
        bad = selection(candidate=[
            "candidate:pedestrian_count_agreement",
            "candidate:pedestrian_count_agreement",
        ])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            parse_selection(json.dumps(bad), candidates, limits)
        with self.assertRaisesRegex(ValueError, "Duplicate JSON key"):
            parse_selection(
                '{"risk_level":"unknown","risk_level":"high",'
                '"selected_claim_ids":[],"limitation_ids":["limit:single_frame"]}',
                candidates, limits,
            )

    def test_non_unknown_requires_a_candidate(self):
        _, candidates, limits, _ = self.get_context()
        bad = selection(level="medium", candidate=[])
        with self.assertRaisesRegex(ValueError, "must reference"):
            parse_selection(json.dumps(bad), candidates, limits)

    def test_high_and_low_safety_guard(self):
        _, candidates, limits, _ = self.get_context()
        for level in ("high", "low"):
            guarded, actions = assemble_decision(
                selection(level=level), candidates, limits
            )
            self.assertEqual(guarded["risk_level"], "unknown")
            self.assertTrue(any("downgraded" in a for a in actions))

    def test_medium_kept_but_unverified(self):
        _, candidates, limits, _ = self.get_context()
        decision, actions = assemble_decision(selection("medium"), candidates, limits)
        self.assertEqual(decision["risk_level"], "medium")
        self.assertIn("unverified", decision["decision_status"])
        self.assertTrue(any("uncalibrated" in flag for flag in actions))

    def test_invalid_model_output_fallback(self):
        _, candidates, limits, _ = self.get_context()
        parsed, error = parse_or_error(
            '{"risk_level":"unknown","selected_claim_ids":"oops",'
            '"limitation_ids":["limit:single_frame"]}', candidates, limits
        )
        self.assertIsNone(parsed)
        self.assertIn("selected_claim_ids", error)
        result, actions = assemble_decision(None, candidates, limits)
        self.assertEqual(result["risk_level"], "unknown")
        self.assertEqual(result["factors"], [])
        self.assertIn("fallback", result["decision_status"])

    def test_retry_feedback_does_not_modify_source_prompt(self):
        original, candidates, limits, _ = self.get_context()
        msgs = make_messages(original)
        revised = repair_messages(msgs, '{"bad":true}', "missing selected_claim_ids")
        self.assertEqual(len(revised), len(msgs) + 2)
        self.assertEqual(revised[:2], msgs)
        self.assertIn("selected_claim_ids", revised[-1]["content"])

    def test_cross_dependency_tampering_rejected(self):
        packet = packet_fixture()
        candidate = packet["candidate_statements_for_human_review"][0]
        candidate["support_refs"].remove("yolo:0")
        with self.assertRaisesRegex(ValueError, "missing comparison dependencies"):
            packet_index(packet)

    def test_source_isolation_must_be_false(self):
        packet = packet_fixture()
        packet["source_isolation"]["CARLA_audit_truth_used"] = True
        with self.assertRaisesRegex(ValueError, "Source-isolation"):
            packet_index(packet)

    def test_packet_cannot_invent_new_support_ref(self):
        packet = packet_fixture()
        packet["candidate_statements_for_human_review"][0]["support_refs"].append(
            "carla:walkers"
        )
        with self.assertRaisesRegex(ValueError, "unknown support refs"):
            packet_index(packet)

    def test_fenced_json_accepted(self):
        _, candidates, limits, _ = self.get_context()
        raw = "```json\n" + json.dumps(selection()) + "\n```"
        self.assertEqual(parse_selection(raw, candidates, limits)["risk_level"], "unknown")


if __name__ == "__main__":
    unittest.main()
