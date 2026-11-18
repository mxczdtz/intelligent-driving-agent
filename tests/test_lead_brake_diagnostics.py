"""Phase 4.1 diagnostic tests; no CARLA, GPU or learned model needed."""
from __future__ import annotations
import csv
import io
import json
import tempfile
from pathlib import Path
import unittest

from scripts.lead_brake_diagnostics import (
    analyze, validate_timeline, csv_report, html_report, create_file,
)

# Representative frame-by-frame trajectory from the 25-frame run.
# Values are fixture input, not derived risk ground truth.
SEPARATION = [
    17.996274, 17.987904, 17.997990, 18.008271, 18.017534,
    18.026124, 18.034387, 18.042513, 18.050577, 18.058630,
    18.059794, 18.028185, 17.953770, 17.836901, 17.677448,
    17.474739, 17.229826, 16.942228, 16.612000, 16.239181,
    15.823819, 15.365888, 14.865911, 14.320214, 13.648340,
]
LEAD_SPEED = [
    16.1587,24.7095,25.1320,25.1667,25.1702,
    25.1695,25.1687,25.1682,25.1680,25.1679,
    24.6131,23.1668,21.6294,20.0934,18.5607,
    16.8462,15.3082,13.7720,12.2377,10.7051,
    9.1742,7.5654,6.0235,3.7005,.002538,
]
DETECTION_COUNTS = [
    2,2,2,2,3,2,4,3,2,5,3,2,2,3,2,3,3,1,4,4,2,3,3,3,1
]


def existing_timeline_fixture():
    rows=[]
    for i,frame in enumerate(range(2349,2374)):
        risk = "medium" if frame in (2355,2358) else "low"
        classes = ["car"] * DETECTION_COUNTS[i]
        if frame == 2373:
            classes=["truck"]
        rows.append({
            "world_frame":frame,
            "ego_speed_kmh":16.1021 if i == 0 else 25.18582,
            "lead_speed_kmh":LEAD_SPEED[i],
            "origin_longitudinal_separation_m":SEPARATION[i],
            "original_yolo_rule_risk":risk,
            "yolo_detection_count":len(classes),
            "yolo_detected_classes":classes,
        })
    return {"schema_version":"carla_lead_brake_timeline.v1",
            "joined_frame_count":25,
            "yolo_frame_matches":25,
            "rows":rows}


class LeadBrakeDiagnosticTests(unittest.TestCase):
    def test_original_25_frame_profile_gives_audit_flags(self):
        result=analyze(existing_timeline_fixture())
        self.assertEqual(result["world_frame_count"],25)
        self.assertEqual(result["original_rule_risk_counts"],{"low":23,"medium":2})
        self.assertEqual(result["closing_context_frame_count"],14)
        self.assertEqual(result["low_with_closing_context_count"],14)
        self.assertEqual(result["flagged_world_frames"],list(range(2360,2374)))
        self.assertFalse(result["is_safety_ground_truth"])

    def test_first_frame_has_no_temporal_delta(self):
        out=analyze(existing_timeline_fixture())
        row=out["rows"][0]
        self.assertIsNone(row["origin_sep_change_from_prior_saved_frame_m"])
        self.assertFalse(row["low_with_closing_context_flag"])

    def test_precise_pair_frames_have_expected_direction(self):
        out=analyze(existing_timeline_fixture())
        lookup={r["world_frame"]:r for r in out["rows"]}
        self.assertEqual(lookup[2358]["original_yolo_rule_risk"],"medium")
        self.assertEqual(lookup[2373]["original_yolo_rule_risk"],"low")
        self.assertGreater(
            lookup[2358]["origin_longitudinal_separation_m"],
            lookup[2373]["origin_longitudinal_separation_m"])
        self.assertGreater(lookup[2373]["ego_minus_lead_speed_kmh"],25)

    def test_threshold_sensitivity_is_explicit(self):
        original=existing_timeline_fixture()
        mild=analyze(original)
        strict=analyze(original,speed_diff_kmh=20)
        self.assertLess(strict["closing_context_frame_count"],
                        mild["closing_context_frame_count"])
        self.assertEqual(strict["thresholds_for_review_only"][
            "ego_minus_lead_speed_at_least_kmh"],20)

    def test_missing_gap_fails_closed(self):
        doc=existing_timeline_fixture()
        for row in doc["rows"]:
            row["origin_longitudinal_separation_m"]=None
        report=analyze(doc)
        self.assertEqual(report["closing_context_frame_count"],0)
        self.assertEqual(report["low_with_closing_context_count"],0)
        self.assertIsNone(report[
            "low_fraction_among_context_frames_not_false_negative_rate"])

    def test_missing_lead_speed_fails_closed(self):
        doc=existing_timeline_fixture()
        for row in doc["rows"]:
            row["lead_speed_kmh"]=None
        report=analyze(doc)
        self.assertEqual(report["closing_context_frame_count"],0)

    def test_low_label_alone_is_not_a_review_flag(self):
        doc=existing_timeline_fixture()
        for row in doc["rows"]:
            row["ego_speed_kmh"]=row["lead_speed_kmh"]
        result=analyze(doc)
        self.assertEqual(result["original_rule_risk_counts"]["low"],23)
        self.assertEqual(result["low_with_closing_context_count"],0)

    def test_unknown_risk_is_not_counted_as_low(self):
        doc=existing_timeline_fixture()
        doc["rows"][-1]["original_yolo_rule_risk"]="unknown"
        result=analyze(doc)
        self.assertEqual(result["low_with_closing_context_count"],13)

    def test_invalid_thresholds_are_rejected(self):
        for speed,drop in ((0,.01),(-1,.01),(2,-.01),(2,6)):
            with self.assertRaises(ValueError):
                analyze(existing_timeline_fixture(),
                        speed_diff_kmh=speed,sep_drop_m=drop)

    def test_reject_invalid_frame_order_and_schema(self):
        original=existing_timeline_fixture()
        original["rows"][1]["world_frame"]=2349
        with self.assertRaises(ValueError):
            analyze(original)
        invalid=existing_timeline_fixture()
        invalid["schema_version"]="something_else"
        with self.assertRaises(ValueError):
            analyze(invalid)

    def test_detector_class_list_count_inconsistency_rejected(self):
        doc=existing_timeline_fixture()
        doc["rows"][5]["yolo_detection_count"]=999
        with self.assertRaises(ValueError):
            analyze(doc)

    def test_report_can_render_html_and_csv(self):
        report=analyze(existing_timeline_fixture())
        html=html_report(report)
        self.assertEqual(html.count("<svg"),2)
        self.assertIn("Lead-braking temporal review",html)
        self.assertIn("NOT safety ground truth",html)
        self.assertIn("2360",html)
        raw_csv=csv_report(report["rows"])
        parsed=list(csv.DictReader(io.StringIO(raw_csv)))
        self.assertEqual(len(parsed),25)
        self.assertEqual(parsed[-1]["world_frame"],"2373")
        self.assertEqual(parsed[-1]["low_with_closing_context_flag"],"True")

    def test_writer_does_not_overwrite_prior_results(self):
        with tempfile.TemporaryDirectory() as dirname:
            p=Path(dirname)/"diagnostics.json"
            create_file(p,"first")
            with self.assertRaises(FileExistsError):
                create_file(p,"new")
            self.assertEqual(p.read_text("utf8"),"first")


if __name__=="__main__":
    unittest.main()
