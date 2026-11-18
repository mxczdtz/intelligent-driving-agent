"""Phase 3.5 evidence-reference audit tests. No model / GPU is required."""
import copy
import unittest

from app.scene_evidence_audit import audit_report, extract_sources


def sample_report():
    return {
        "schema_version": "scene_llm_risk.v1",
        "frame": 3488,
        "validation_status": "valid_after_retry",
        "input_to_llm": {
            "world_frame": 3488,
            "YOLO_predictions": [
                {"id": "yolo:0", "class": "person", "confidence": .7769},
                {"id": "yolo:1", "class": "car", "confidence": .4280},
                {"id": "yolo:2", "class": "person", "confidence": .3845},
            ],
            "VLM_claims_unverified": [
                {"id": "vlm:0", "type": "pedestrian_count", "value": 2}
            ],
            "available_evidence_refs": [
                "ego:speed", "vlm:0", "yolo:0", "yolo:1", "yolo:2"
            ],
            "cross_checks": [
                {"comparison": "pedestrian_count",
                 "status": "same_count_not_identity_verified"},
                {"comparison": "traffic_lights", "status": "unverified_by_yolo"},
            ],
            "has_calibrated_distance_or_TTC": False,
        },
        "final_research_decision": {
            "risk_level": "unknown",
            "key_factors": [
                "无可靠空间关联或时序信息",
                "缺乏独立危险标签",
                "YOLO检测结果与VLM声明不一致",
                "未提供可靠的位置和运动趋势",
            ],
            "evidence_refs": ["ego:speed"],
            "driving_suggestion": "仅供仿真研究和人工复核",
            "uncertainty": [
                "缺乏可靠的空间关联或时序信息",
                "缺乏独立危险标签",
                "YOLO检测结果与VLM声明不一致",
                "未提供可靠的位置和运动趋势",
            ]
        },
        "audit_only": {
            "simulator_truth": {"secret": "NOT_FOR_AUDIT_ALGORITHM"}
        },
    }


class EvidenceAuditTests(unittest.TestCase):
    def test_user_case_detects_source_gap(self):
        result = audit_report(sample_report())
        self.assertEqual(result["audit_status"], "needs_review")
        self.assertEqual(result["final_research_risk_unchanged"], "unknown")
        self.assertEqual(
            result["factor_audit"][2]["missing_source_types_in_global_refs"],
            ["vlm", "yolo"],
        )

    def test_unqualified_conflict_flag(self):
        result = audit_report(sample_report())
        self.assertIn(
            "unqualified_cross_model_conflict",
            result["factor_audit"][2]["flag_codes"],
        )

    def test_model_naming_detection(self):
        self.assertEqual(extract_sources("YOLO与VLM输出存在差异"), {"yolo", "vlm"})
        self.assertEqual(extract_sources("自车速度是10.8km/h"), {"ego"})
        self.assertEqual(extract_sources("没有TTC信息"), set())

    def test_both_sources_cited_still_not_semantically_verified(self):
        report = sample_report()
        report["final_research_decision"]["evidence_refs"] = [
            "yolo:0", "vlm:0"
        ]
        result = audit_report(report)
        self.assertNotIn(
            "explicit_source_reference_missing",
            result["factor_audit"][2]["flag_codes"],
        )
        self.assertIn(
            "unqualified_cross_model_conflict",
            result["factor_audit"][2]["flag_codes"],
        )

    def test_no_conflict_overstatement_if_qualified(self):
        report = sample_report()
        report["final_research_decision"]["evidence_refs"] = [
            "yolo:0", "vlm:0"
        ]
        report["final_research_decision"]["key_factors"][2] = \
            "YOLO与VLM声明可能存在不一致，仍需图像核验"
        result = audit_report(report)
        self.assertEqual(
            result["factor_audit"][2]["flag_codes"], []
        )
        self.assertEqual(result["audit_status"], "no_obvious_source_mismatch")

    def test_audit_does_not_depend_on_simulator_truth(self):
        report = sample_report()
        changed = copy.deepcopy(report)
        changed["audit_only"]["simulator_truth"] = {
            "private_different_truth": 9999,
        }
        self.assertEqual(audit_report(report), audit_report(changed))
        self.assertNotIn("NOT_FOR_AUDIT_ALGORITHM", str(audit_report(report)))

    def test_unknown_citations_flagged(self):
        report = sample_report()
        report["final_research_decision"]["evidence_refs"] = ["fake:99"]
        result = audit_report(report)
        self.assertTrue(any("Unknown evidence IDs" in item for item in result["issues"]))
        self.assertTrue(any("Unsupported evidence prefixes" in item for item in result["issues"]))

    def test_repeated_citations_flagged(self):
        report = sample_report()
        report["final_research_decision"]["evidence_refs"] = ["ego:speed", "ego:speed"]
        result = audit_report(report)
        self.assertTrue(any("Repeated evidence IDs" in item for item in result["issues"]))

    def test_fallback_decision_still_auditable(self):
        report = sample_report()
        report["validation_status"] = "fallback"
        report["final_research_decision"] = {
            "risk_level": "unknown",
            "key_factors": ["证据不足"],
            "evidence_refs": [],
            "driving_suggestion": "仅供人工复核",
            "uncertainty": ["推理输出无效"],
        }
        result = audit_report(report)
        self.assertEqual(result["llm_validation_status"], "fallback")
        self.assertEqual(result["audit_status"], "no_obvious_source_mismatch")

    def test_wrong_schema_rejected(self):
        report = sample_report()
        report["schema_version"] = "other_schema"
        with self.assertRaises(ValueError):
            audit_report(report)


if __name__ == "__main__":
    unittest.main()
