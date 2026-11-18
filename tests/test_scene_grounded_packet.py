"""Phase 3.6 citation-ready packet tests (standard library, no models)."""
from __future__ import annotations

import copy
import unittest

from app.scene_grounded_packet import build_packet


def phase33_fixture():
    return {
        "schema_version": "scene_fusion_evidence.v1",
        "world_frame": 3488,
        "image": "frame_00003488.png",
        "provenance": {"yolo": "saved_yolo.json", "vlm": "saved_vlm.json"},
        "yolo_visual_evidence": {
            "class_counts": {"car": 1, "person": 2},
            "detections": [
                {"class": "person", "confidence": 0.776938259601593,
                 "bbox_xyxy_from_alignment": None},
                {"class": "car", "confidence": 0.42803144454956055,
                 "bbox_xyxy_from_alignment": None},
                {"class": "person", "confidence": 0.3844858407974243,
                 "bbox_xyxy_from_alignment": None},
            ],
        },
        "vlm_text_evidence": {
            "description_unverified": (
                "The driving camera image shows a street with a red car "
                "driving away from the camera. There are two pedestrians "
                "on the sidewalk, and one red and one yellow traffic light."
            ),
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
            {
                "comparison": "pedestrian_count", "yolo_count": 2,
                "vlm_text_count": 2,
                "status": "same_count_not_identity_verified",
            },
            {
                "comparison": "traffic_lights", "yolo_count": 0,
                "vlm_claimed_colors": ["red", "yellow"],
                "status": "unverified_by_yolo",
            },
        ],
        "uncertainties": [
            "No metric distance or TTC",
            "Traffic-light colors unverified",
        ],
        "risk": {
            "original_yolo_rule_baseline": "low",
            "new_fused_risk": None,
        },
        "audit_only": {
            "simulator_truth": {
                "walker_inside_lane_lateral_band": True,
                "marker": "NEVER_INCLUDE_CARLA_GROUND_TRUTH",
            }
        },
    }


class EvidencePacketTests(unittest.TestCase):
    def test_real_scenario_count_agrees_but_not_truth(self):
        packet = build_packet(phase33_fixture(), 10.795)
        candidates = packet["candidate_statements_for_human_review"]
        self.assertEqual(len(candidates), 2)
        self.assertEqual(
            candidates[0]["status"],
            "model_report_agreement_not_scene_truth",
        )
        self.assertEqual(
            candidates[0]["support_refs"],
            ["cross:pedestrian_count", "yolo:0", "yolo:2", "vlm:2"],
        )
        self.assertIn("仅在报告数量上相同", candidates[0]["text"])
        self.assertEqual(packet["followup_llm_contract"]["new_risk_level"], None)

    def test_traffic_light_claim_not_verified(self):
        packet = build_packet(phase33_fixture())
        light = packet["candidate_statements_for_human_review"][1]
        self.assertEqual(light["status"], "unverified_visual_color_claim")
        self.assertIn("vlm:4", light["support_refs"])
        cross = next(e for e in packet["evidence"] if e["id"] == "cross:traffic_lights")
        self.assertEqual(cross["observation"]["yolo_traffic_light_count"], 0)
        self.assertIn("Failure to detect", cross["qualifiers"][1])

    def test_carla_truth_never_enters_packet(self):
        input_a = phase33_fixture()
        input_b = copy.deepcopy(input_a)
        input_b["audit_only"]["simulator_truth"] = {
            "danger_level": "high",
            "marker": "OTHER_TRUTH",
        }
        input_b["risk"]["original_yolo_rule_baseline"] = "high"
        result_a = build_packet(input_a, 10.795)
        result_b = build_packet(input_b, 10.795)
        self.assertEqual(result_a, result_b)
        self.assertNotIn("NEVER_INCLUDE_CARLA_GROUND_TRUTH", str(result_a))

    def test_not_all_vlm_claims_become_verified(self):
        packet = build_packet(phase33_fixture())
        vlms = [e for e in packet["evidence"] if e["source"] == "VLM"]
        self.assertEqual(len(vlms), 5)
        self.assertTrue(all(x["epistemic_status"] == "unverified_generated_text" for x in vlms))

    def test_invalid_count_agreement_is_rejected(self):
        doc = phase33_fixture()
        doc["vlm_text_evidence"]["claims_unverified"][2]["value"] = 3
        with self.assertRaisesRegex(ValueError, "comparison disagrees"):
            build_packet(doc)

    def test_actual_count_difference_is_qualified(self):
        doc = phase33_fixture()
        doc["vlm_text_evidence"]["claims_unverified"][2]["value"] = 3
        doc["cross_model_checks"][0]["status"] = "count_disagreement_needs_image_review"
        packet = build_packet(doc)
        stmt = packet["candidate_statements_for_human_review"][0]
        self.assertEqual(stmt["status"], "model_report_count_difference_unverified")
        self.assertIn("需人工核验", stmt["text"])

    def test_traffic_light_status_checked_against_detections(self):
        doc = phase33_fixture()
        doc["yolo_visual_evidence"]["detections"].append({
            "class": "traffic light", "confidence": 0.5,
            "bbox_xyxy_from_alignment": [10, 20, 50, 100],
        })
        doc["yolo_visual_evidence"]["class_counts"] = {
            "car": 1, "person": 2, "traffic light": 1,
        }
        with self.assertRaisesRegex(ValueError, "Traffic-light status conflicts"):
            build_packet(doc)
        doc["cross_model_checks"][1]["status"] = "light_color_still_unverified"
        result = build_packet(doc)
        cross = next(e for e in result["evidence"] if e["id"] == "cross:traffic_lights")
        self.assertEqual(cross["observation"]["yolo_traffic_light_count"], 1)

    def test_unsupported_class_counts_are_rejected(self):
        doc = phase33_fixture()
        doc["yolo_visual_evidence"]["class_counts"] = {"person": 2}
        with self.assertRaisesRegex(ValueError, "class_counts"):
            build_packet(doc)

    def test_no_duplicate_evidence_refs(self):
        pkt = build_packet(phase33_fixture(), 10.795)
        refs = [e["id"] for e in pkt["evidence"]]
        self.assertEqual(len(refs), len(set(refs)))
        self.assertIn("ego:speed", refs)
        for stmt in pkt["candidate_statements_for_human_review"]:
            self.assertTrue(set(stmt["support_refs"]) <= set(refs))

    def test_rejects_untrusted_verified_vlm_claim(self):
        doc = phase33_fixture()
        doc["vlm_text_evidence"]["claims_unverified"][0]["verification"] = "verified"
        with self.assertRaisesRegex(ValueError, "unverified"):
            build_packet(doc)

    def test_bboxes_not_fabricated_when_missing(self):
        pkt = build_packet(phase33_fixture())
        person = next(e for e in pkt["evidence"] if e["id"] == "yolo:0")
        self.assertIsNone(person["observation"]["bbox_xyxy_from_alignment"])
        self.assertEqual(person["epistemic_status"], "unverified_prediction")

    def test_invalid_speed_rejected(self):
        with self.assertRaises(ValueError):
            build_packet(phase33_fixture(), float("nan"))
        with self.assertRaises(ValueError):
            build_packet(phase33_fixture(), -1)


if __name__ == "__main__":
    unittest.main()
