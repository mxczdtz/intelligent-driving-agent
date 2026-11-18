"""Tests for Phase 4.0 lead-braking trace (metadata and YOLO only)."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from scripts.lead_brake_timeline import (
    build_timeline, csv_text, pick_numeric, flatten_numeric_fields, save_new,
)


def fake_capture():
    """25 frames, known 18.06 -> 13.65 m, risk shift at selected frames."""
    meta = {"frames": []}
    yolo = {"frames": []}
    for n, frame in enumerate(range(2355, 2380)):
        sep = 18.06 - (18.06 - 13.65) * n / 24
        meta["frames"].append({
            "world_frame": frame,
            "image_file": f"frame_{frame:08d}.png",
            "ego_speed_kmh": 25.19,
            "lead_speed_kmh": max(0, 25 - 25 * n / 24),
            "lead_relative_longitudinal_m": sep,
        })
        risk = "medium" if frame <= 2360 else "low"
        labels = (
            [{"class_name":"car","confidence":.661},
             {"class_name":"traffic light","confidence":.577},
             {"class_name":"car","confidence":.473},
             {"class_name":"motorcycle","confidence":.445},
             {"class_name":"traffic light","confidence":.382}]
            if frame == 2358 else
            [{"class_name":"truck","confidence":.364}] if frame == 2373 else
            [{"class_name":"car","confidence":.5}]
        )
        yolo["frames"].append({
            "world_frame": frame,
            "risk_level": risk,
            "detections": labels
        })
    return meta,yolo


class LeadBrakeTimelineTests(unittest.TestCase):
    def test_full_capture_and_separation_trend(self):
        metadata,yolo=fake_capture()
        result=build_timeline(metadata,yolo)
        self.assertEqual(result["joined_frame_count"],25)
        self.assertEqual(result["yolo_frame_matches"],25)
        self.assertEqual(result["separation_trend"]["raw_end_minus_start_m"],-4.41)
        self.assertEqual(result["separation_trend"]["consecutive_decreases"],24)
        self.assertEqual(result["original_risk_distribution"]["medium"],6)
        self.assertEqual(result["original_risk_distribution"]["low"],19)

    def test_compare_two_real_observed_frame_ids(self):
        metadata,yolo=fake_capture()
        result=build_timeline(metadata,yolo)
        compare=result["selected_frame_comparison"]
        self.assertEqual(compare["earlier"]["original_yolo_rule_risk"],"medium")
        self.assertEqual(compare["later"]["original_yolo_rule_risk"],"low")
        self.assertEqual(compare["earlier"]["yolo_detection_count"],5)
        self.assertEqual(compare["later"]["yolo_detection_count"],1)
        self.assertEqual(compare["later"]["yolo_detected_classes"],["truck"])
        self.assertIsNotNone(compare["actor_origin_separation_change_m"])

    def test_no_separation_field_is_not_fabricated(self):
        metadata,yolo=fake_capture()
        for row in metadata["frames"]:
            row["unknown_3d_gap_m"] = row.pop("lead_relative_longitudinal_m")
        result=build_timeline(metadata,yolo)
        self.assertEqual(result["separation_field_matched"],[])
        self.assertIsNone(result["separation_trend"])
        self.assertIsNone(result["selected_frame_comparison"][
            "actor_origin_separation_change_m"])
        self.assertIn("unknown_3d_gap_m",
                      result["numeric_field_candidates_in_metadata"])

    def test_exact_separation_key_selects_known_field(self):
        metadata,yolo=fake_capture()
        for row in metadata["frames"]:
            row["custom_actor_origin_sep_m"] = row.pop("lead_relative_longitudinal_m")
        result=build_timeline(
            metadata,yolo,separation_key="custom_actor_origin_sep_m"
        )
        self.assertEqual(result["separation_field_matched"],
                         ["custom_actor_origin_sep_m"])
        self.assertEqual(result["separation_trend"]["raw_end_minus_start_m"],-4.41)

    def test_nested_numeric_fields_can_be_found_without_guessing(self):
        fields=flatten_numeric_fields({
            "world_frame":2358,
            "telemetry":{"lead_speed_kmh":12.5,"lead_relative_longitudinal_m":17.8},
        })
        n,path=pick_numeric(fields,("lead_relative_longitudinal_m",))
        self.assertEqual((n,path),(17.8,"telemetry.lead_relative_longitudinal_m"))

    def test_missing_yolo_record_remains_unknown(self):
        metadata,yolo=fake_capture()
        yolo["frames"]=[row for row in yolo["frames"]
                        if row["world_frame"] != 2373]
        result=build_timeline(metadata,yolo)
        self.assertEqual(result["yolo_frame_matches"],24)
        later=result["selected_frame_comparison"]["later"]
        self.assertIsNone(later["original_yolo_rule_risk"])
        self.assertEqual(later["yolo_frame_status"],"not_found")

    def test_duplicate_metadata_frame_rejected(self):
        metadata,yolo=fake_capture()
        metadata["frames"].append(metadata["frames"][0].copy())
        with self.assertRaisesRegex(ValueError,"Repeated metadata"):
            build_timeline(metadata,yolo)

    def test_bad_metadata_shape_rejected(self):
        with self.assertRaisesRegex(ValueError,"frames"):
            build_timeline({}, {"frames":[]})

    def test_csv_contains_exact_header_and_rows(self):
        meta,yolo=fake_capture()
        result=build_timeline(meta,yolo)
        text=csv_text(result["rows"])
        self.assertEqual(len(text.splitlines()),26)
        self.assertTrue(text.startswith("world_frame,ego_speed_kmh,lead_speed_kmh"))
        self.assertIn("2358",text)

    def test_existing_file_never_overwritten(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"report.json"
            save_new(p,'{"a":1}')
            with self.assertRaises(FileExistsError):
                save_new(p,'{"a":2}')
            self.assertEqual(p.read_text(encoding="utf8"),'{"a":1}')

    def test_comparison_unavailable_if_id_absent(self):
        meta,yolo=fake_capture()
        result=build_timeline(meta,yolo,compare_frames=(2358,5000))
        self.assertIsNone(result["selected_frame_comparison"])


if __name__=="__main__":
    unittest.main()
