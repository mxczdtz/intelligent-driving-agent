"""Tests for app.scene_fusion; no GPU, CARLA, or model weights required."""
import copy
import unittest

from app.scene_fusion import (
    build_truth_audit, extract_vlm_claims, fuse_visual_evidence,
)

TEXT = (
    "The driving camera image shows a street with a red car driving away from the camera. "
    "There are two pedestrians on the sidewalk, one walking towards the camera and "
    "the other walking away from the camera. "
    "There are traffic lights on the side of the road, one red and one yellow."
)


def example():
    return {
        "world_frame": 3488,
        "image": "test-frame.png",
        "sources": {"yolo": "saved-yolo.json", "vlm": "saved-vlm.json"},
        "yolo_original_rule_risk": "low",
        "yolo_detection_info": {
            "count": 3,
            "objects": [
                {"class": "person", "confidence": 0.776938259601593, "bbox": None},
                {"class": "car", "confidence": 0.42803144454956055, "bbox": None},
                {"class": "person", "confidence": 0.3844858407974243, "bbox": None},
            ],
        },
        "vlm_unverified_description": TEXT,
        "carla_truth_for_human_audit_only": {
            "walker_inside_lane_lateral_band": True,
            "walker_lane_center_offset_m": 0.04232860562396214,
        },
    }


class FusionTest(unittest.TestCase):
    def test_counts_original_risk_and_unverified_signal(self):
        fused = fuse_visual_evidence(example())
        self.assertEqual(fused["yolo_visual_evidence"]["class_counts"],
                         {"car": 1, "person": 2})
        self.assertEqual(fused["risk"]["original_yolo_rule_baseline"], "low")
        self.assertIsNone(fused["risk"]["new_fused_risk"])
        self.assertEqual(fused["cross_model_checks"][0]["status"],
                         "same_count_not_identity_verified")
        self.assertEqual(fused["cross_model_checks"][1]["status"],
                         "unverified_by_yolo")

    def test_truth_never_enters_visual_fusion(self):
        a = example()
        b = copy.deepcopy(a)
        b["carla_truth_for_human_audit_only"] = {
            "walker_inside_lane_lateral_band": False,
            "walker_lane_center_offset_m": -100.0,
        }
        self.assertEqual(fuse_visual_evidence(a), fuse_visual_evidence(b))
        audit = build_truth_audit(a, fuse_visual_evidence(a))
        self.assertTrue(any("Potential semantic mismatch" in note
                            for note in audit["posthoc_review_notes"]))

    def test_missing_count_does_not_create_agreement(self):
        x = example()
        x["vlm_unverified_description"] = "Pedestrians near the street."
        fused = fuse_visual_evidence(x)
        self.assertFalse(any(c["comparison"] == "pedestrian_count"
                             for c in fused["cross_model_checks"]))

    def test_cannot_silently_accept_incomplete_summary(self):
        x = example()
        x["yolo_detection_info"]["count"] = 4
        with self.assertRaises(ValueError):
            fuse_visual_evidence(x)

    def test_bad_confidence_is_rejected(self):
        x = example()
        x["yolo_detection_info"]["objects"][0]["confidence"] = 2.5
        with self.assertRaises(ValueError):
            fuse_visual_evidence(x)

    def test_language_claims_are_unverified(self):
        claims = extract_vlm_claims(TEXT)
        self.assertTrue(all(c["verification"] == "unverified" for c in claims))
        self.assertEqual(next(c["value"] for c in claims
                              if c["type"] == "pedestrian_count"), 2)


if __name__ == "__main__":
    unittest.main()
