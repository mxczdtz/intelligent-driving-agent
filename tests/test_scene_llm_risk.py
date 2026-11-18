"""Unit tests for local-Qwen structured risk module; no model or GPU needed."""
from __future__ import annotations

import copy
import json
import unittest

from app.scene_llm_risk import (
    build_model_input, prompt_for, parse_model_json, validate_decision,
    safety_filter, fallback_unknown,
)

def fixture():
    return {
        "schema_version": "scene_fusion_evidence.v1",
        "world_frame": 3488,
        "yolo_visual_evidence": {
            "detections": [
                {"class": "person", "confidence": .7769,
                 "bbox_xyxy_from_alignment": None},
                {"class": "car", "confidence": .4280,
                 "bbox_xyxy_from_alignment": None},
                {"class": "person", "confidence": .3845,
                 "bbox_xyxy_from_alignment": None},
            ]
        },
        "vlm_text_evidence": {
            "description_unverified": "Untrusted VLM text; do not use raw.",
            "claims_unverified": [
                {"type": "pedestrian_count", "value": 2, "verification": "unverified",
                 "quoted_context": "The two pedestrians are on the sidewalk."},
                {"type": "pedestrian_sidewalk_location", "value": True,
                 "verification": "unverified"},
                {"type": "traffic_light_colors", "value": ["red", "yellow"],
                 "verification": "unverified"},
            ]
        },
        "cross_model_checks": [
            {"comparison": "pedestrian_count", "status": "same_count_not_identity_verified"},
            {"comparison": "traffic_lights", "status": "unverified_by_yolo"},
        ],
        "uncertainties": ["No metric distance or TTC."],
        "risk": {"original_yolo_rule_baseline": "low", "new_fused_risk": None},
        "audit_only": {
            "simulator_truth": {
                "walker_inside_lane_lateral_band": True,
                "walker_lane_center_offset_m": 0.042,
                "private_marker": "NEVER_SEND_THIS_TO_LLM",
            }
        },
    }

def sample_decision(level="medium"):
    return {
        "risk_level": level,
        "key_factors": ["YOLO检出了行人，但不清楚是否在行驶路径内"],
        "evidence_refs": ["yolo:0", "ego:speed"],
        "driving_suggestion": "谨慎驾驶，保持观察",
        "uncertainty": ["缺少几何距离与运动趋势"],
    }


class LocalRiskUnitTests(unittest.TestCase):
    def test_prompt_excludes_audit_truth_and_original_baseline(self):
        x = fixture()
        payload, ids = build_model_input(x, 10.795)
        prompt = json.dumps(prompt_for(payload), ensure_ascii=False)
        self.assertNotIn("NEVER_SEND_THIS_TO_LLM", prompt)
        self.assertNotIn("walker_inside_lane_lateral_band", prompt)
        self.assertNotIn("original_yolo_rule_baseline", prompt)
        self.assertIn("ego:speed", ids)
        self.assertIn("yolo:0", ids)

        x2 = copy.deepcopy(x)
        x2["audit_only"]["simulator_truth"]["private_marker"] = "DIFFERENT_TRUTH"
        x2["risk"]["original_yolo_rule_baseline"] = "high"
        self.assertEqual(build_model_input(x, 10.795),
                         build_model_input(x2, 10.795))

    def test_references_must_come_from_input(self):
        payload, ids = build_model_input(fixture(), 10.795)
        valid = validate_decision(sample_decision(), ids)
        self.assertEqual(valid["risk_level"], "medium")
        bad = sample_decision()
        bad["evidence_refs"] = ["carla:walker_ground_truth"]
        with self.assertRaises(ValueError):
            validate_decision(bad, ids)

    def test_strict_schema_rejects_extra_fields_and_wrong_types(self):
        _, ids = build_model_input(fixture(), 10.795)
        bad = sample_decision()
        bad["risk_score"] = 0.99
        with self.assertRaises(ValueError):
            validate_decision(bad, ids)
        bad = sample_decision()
        bad["uncertainty"] = "none"
        with self.assertRaises(ValueError):
            validate_decision(bad, ids)
        bad = sample_decision("critical")
        with self.assertRaises(ValueError):
            validate_decision(bad, ids)

    def test_json_parser_rejects_extra_commentary_and_duplicate_keys(self):
        example = sample_decision()
        result = parse_model_json(json.dumps(example, ensure_ascii=False))
        self.assertEqual(result["risk_level"], "medium")
        fenced = "```json\n" + json.dumps(example, ensure_ascii=False) + "\n```"
        self.assertEqual(parse_model_json(fenced)["risk_level"], "medium")
        with self.assertRaises(json.JSONDecodeError):
            parse_model_json("Here is JSON: " + json.dumps(example))
        with self.assertRaises(ValueError):
            parse_model_json('{"risk_level":"unknown","risk_level":"high"}')

    def test_high_downgraded_without_ttc_or_verified_geometry(self):
        payload, ids = build_model_input(fixture(), 10.795)
        high = validate_decision(sample_decision("high"), ids)
        guarded, flags = safety_filter(high, payload)
        self.assertEqual(guarded["risk_level"], "unknown")
        self.assertTrue(any("High risk downgraded" in s for s in flags))

    def test_low_downgraded_with_people_and_unknown_lane_relation(self):
        payload, ids = build_model_input(fixture(), 10.795)
        low = validate_decision(sample_decision("low"), ids)
        guarded, flags = safety_filter(low, payload)
        self.assertEqual(guarded["risk_level"], "unknown")
        self.assertTrue(any("Low risk downgraded" in s for s in flags))

    def test_medium_kept_as_candidate_but_suggestion_is_safe(self):
        payload, ids = build_model_input(fixture(), 10.795)
        medium = validate_decision(sample_decision("medium"), ids)
        guarded, flags = safety_filter(medium, payload)
        self.assertEqual(guarded["risk_level"], "medium")
        self.assertNotEqual(guarded["driving_suggestion"],
                            medium["driving_suggestion"])
        self.assertIn("仿真研究", guarded["driving_suggestion"])

    def test_no_detections_does_not_equal_safe(self):
        fusion = fixture()
        fusion["yolo_visual_evidence"]["detections"] = []
        payload, ids = build_model_input(fusion, None)
        candidate = {
            "risk_level": "low",
            "key_factors": ["未检测到可见目标"],
            "evidence_refs": [],
            "driving_suggestion": "注意道路",
            "uncertainty": ["视野有限"],
        }
        # Non-unknown level requires a citation, and none is available.
        with self.assertRaises(ValueError):
            validate_decision(candidate, ids)
        self.assertEqual(fallback_unknown("insufficient")["risk_level"], "unknown")

    def test_speed_must_be_realistic(self):
        with self.assertRaises(ValueError):
            build_model_input(fixture(), -10)
        with self.assertRaises(ValueError):
            build_model_input(fixture(), float("nan"))

    def test_candidate_must_have_uncertainty(self):
        _, ids = build_model_input(fixture(), 10.795)
        candidate = sample_decision()
        candidate["uncertainty"] = []
        with self.assertRaises(ValueError):
            validate_decision(candidate, ids)


    def test_real_observed_overlength_output_is_invalid(self):
        observed = {
            "risk_level": "unknown",
            "key_factors": [
                "交通信号灯颜色", "行人移动语言", "行人在车道中的位置",
                "交通标志的颜色", "行人数量", "交通信号灯的方向"
            ],
            "evidence_refs": ["ego:speed", "yolo:0", "yolo:1", "yolo:2"],
            "driving_suggestion": "谨慎的人工参考建议",
            "uncertainty": [
                "缺乏关键证据来确认交通信号灯的颜色、行人移动的语言以及行人位置",
                "交通标志的颜色和方向可能与原始报告不符",
                "行人数量和交通标志的颜色可能不一致",
                "没有几何验证的交通标志位置",
                "缺乏独立的安全性保证",
                "缺乏用于计算空间相关性的绝对距离、相对速度或交通流量",
                "缺乏对潜在危险的独立评估和对象身份的确认",
                "缺乏已知的高风险标签，原低风险并不构成安全保证",
            ],
        }
        from app.scene_llm_risk import evaluate_raw_response
        _, evidence_ids = build_model_input(fixture(), 10.795)
        candidate, error = evaluate_raw_response(
            "```json\n" + json.dumps(observed, ensure_ascii=False) + "\n```",
            evidence_ids,
        )
        self.assertIsNone(candidate)
        self.assertIn("key_factors", error)

    def test_single_repair_instruction_preserves_full_decision_schema(self):
        from app.scene_llm_risk import make_repair_messages
        msgs = prompt_for(build_model_input(fixture(), 10.795)[0])
        fixed = make_repair_messages(msgs, '{"risk_level":"unknown"}',
                                     "key_factors must be a list of at most 5")
        self.assertEqual(len(fixed), len(msgs) + 2)
        self.assertIn("1～3", fixed[-1]["content"])
        self.assertIn("key_factors", fixed[-1]["content"])
        self.assertIn("uncertainty", fixed[-1]["content"])
        self.assertIn('"risk_level"', fixed[-2]["content"])

    def test_repaired_answer_still_strictly_validated(self):
        from app.scene_llm_risk import evaluate_raw_response
        _, ids = build_model_input(fixture(), 10.795)
        repaired = {
            "risk_level": "unknown",
            "key_factors": ["图像存在行人，但距离未知"],
            "evidence_refs": ["yolo:0"],
            "driving_suggestion": "仿真研究中谨慎观察",
            "uncertainty": ["没有时序和净距离"],
        }
        candidate, error = evaluate_raw_response(
            json.dumps(repaired, ensure_ascii=False), ids
        )
        self.assertIsNone(error)
        self.assertEqual(candidate["risk_level"], "unknown")

    def test_repair_cannot_use_unsupported_evidence(self):
        from app.scene_llm_risk import evaluate_raw_response
        _, ids = build_model_input(fixture(), 10.795)
        bad = {
            "risk_level": "medium",
            "key_factors": ["看到了红灯"],
            "evidence_refs": ["carla:traffic_light_true"],
            "driving_suggestion": "等待",
            "uncertainty": ["未知距离"],
        }
        candidate, error = evaluate_raw_response(
            json.dumps(bad, ensure_ascii=False), ids
        )
        self.assertIsNone(candidate)
        self.assertIn("unknown evidence ID", error)


if __name__ == "__main__":
    unittest.main()
