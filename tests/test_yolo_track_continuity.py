"""Phase 4.4 vision-only track-continuity audit (no CARLA or model)."""
from __future__ import annotations

import copy
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from scripts.yolo_vehicle_box_tracks import track_frames
from scripts.yolo_track_continuity import (
    continuity_audit, csv_report, html_report, new_file, verify_report,
)


def box(cls: str, xyxy: list[int], idx: int) -> dict:
    return {
        "index": idx,
        "class_predicted": cls,
        "confidence": .76,
        "bbox_pixel_xyxy": list(xyxy),
        "bbox_parse_status": "pixel_xyxy_parsed",
        "bbox_source_field": "box_xyxy",
        "raw_detection_keys_and_types": {"class_name": "str", "box_xyxy": "list"},
    }


def detection_fixture(frames: list[int] | None = None) -> dict:
    if frames is None:
        frames = list(range(2349, 2374))
    rows = []
    num_detections = 0
    for frame in frames:
        detections = []
        # The synthetic center car is absent after frame 2368.
        if frame <= 2368:
            dx = frame - 2349
            detections.append(
                box("car", [265 + dx//2, 205, 345 + dx//2, 310], 0)
            )
        # A left-side vehicle appears only at frame 2358.
        if frame == 2358:
            detections.append(box("car", [80, 200, 130, 260], 1))
        # Another one appears briefly to create a separate track.
        if frame == 2361:
            detections.append(box("car", [495, 220, 555, 310], 1))
        # A box on another region starts at frame 2369 and persists to 2373.
        if frame >= 2369:
            dx = frame - 2369
            detections.append(box("truck", [430+dx, 215, 510+dx, 320], 0))
        num_detections += len(detections)
        rows.append({
            "world_frame": frame,
            "image_path": f"E:/fake/frame_{frame:08d}.png",
            "image_width": 640,
            "image_height": 480,
            "original_yolo_rule_risk": (
                "medium" if frame in (2355, 2358) else "low"
            ),
            "record_status": "detections_array_parsed",
            "detection_count": len(detections),
            "detections": detections,
        })
    return {
        "schema_version": "yolo_temporal_readiness.v1",
        "image_count": len(rows),
        "raw_detection_count": num_detections,
        "valid_pixel_xyxy_count": num_detections,
        "source_isolation": {
            "CARLA_metadata_or_actor_truth_read": False,
            "models_rerun": False,
            "VLM_or_Qwen_used": False,
        },
        "rows": rows,
    }


def tracked_fixture(frames=None):
    return track_frames(detection_fixture(frames))


class ContinuityTests(unittest.TestCase):
    def test_all_25_frames_and_separate_tracks(self):
        result = continuity_audit(tracked_fixture())
        self.assertEqual(result["frame_count"], 25)
        self.assertEqual(result["vehicle_box_count"], 27)
        self.assertEqual(result["tentative_track_count"], 4)
        self.assertEqual(result["shared_ids_in_selected_frames"], [])
        self.assertFalse(result["source_isolation"]["CARLA_truth_used"])

    def test_dynamic_track_start_and_end_frames(self):
        result = continuity_audit(tracked_fixture())
        histories = result["all_track_histories"]
        center = next(x for x in histories if x["first_world_frame"] == 2349)
        self.assertEqual(center["last_world_frame"], 2368)
        last = next(x for x in histories if x["last_world_frame"] == 2373)
        self.assertEqual(last["first_world_frame"], 2369)
        self.assertNotEqual(center["hypothesis_track_id"],
                            last["hypothesis_track_id"])

    def test_real_comparison_frame_2358_and_2373(self):
        out = continuity_audit(tracked_fixture())
        a = out["selected_frames"]["2358"]
        b = out["selected_frames"]["2373"]
        self.assertEqual(a["original_yolo_rule_risk"], "medium")
        self.assertEqual(b["original_yolo_rule_risk"], "low")
        self.assertEqual(len(a["track_ids"]), 2)
        self.assertEqual(len(b["track_ids"]), 1)
        self.assertFalse(a["boxes"][0]["center_region_heuristic"] is False)

    def test_label_change_is_not_physical_identity(self):
        doc = detection_fixture([2358, 2359])
        doc["rows"][1]["detections"][0]["class_predicted"] = "truck"
        output = continuity_audit(track_frames(doc, compare_frames=(2358,2359)),
                                  compare_frames=(2358,2359))
        changes = sum(len(x["class_change_events"])
                      for x in output["all_track_histories"])
        self.assertEqual(changes, 1)
        self.assertTrue(all("not_actor_identity" in x["state"]
                            for x in output["all_track_histories"]))

    def test_input_does_not_contain_carla_ground_truth_dependency(self):
        doc = tracked_fixture()
        result = continuity_audit(doc)
        doc["secret_carla_truth"] = {"walker":"NEVER_ECHO_THIS"}
        self.assertEqual(continuity_audit(doc), result)
        self.assertNotIn("NEVER_ECHO_THIS",str(result))

    def test_input_schema_validation(self):
        doc = tracked_fixture()
        doc["schema_version"] = "other"
        with self.assertRaisesRegex(ValueError,"Expected"):
            verify_report(doc)

    def test_bad_source_provenance_rejected(self):
        doc = tracked_fixture()
        doc["source_isolation"]["CARLA_metadata_or_actor_truth_read"] = True
        with self.assertRaisesRegex(ValueError,"provenance"):
            continuity_audit(doc)

    def test_track_id_gap_is_rejected(self):
        doc = tracked_fixture()
        first = doc["frames"][0]["tracked_vehicle_detections"][0]
        doc["frames"][2]["tracked_vehicle_detections"][0]["hypothesis_track_id"] = "T999"
        # Gap by forging the original T001 track to skip a middle frame.
        doc["frames"][1]["tracked_vehicle_detections"][0]["hypothesis_track_id"] = "T998"
        # Also update report ID count so validation reaches continuity checks.
        doc["number_of_tentative_tracks"] = 6
        with self.assertRaises(ValueError):
            continuity_audit(doc)

    def test_duplicate_id_same_frame_is_rejected(self):
        doc = tracked_fixture()
        frame = next(x for x in doc["frames"] if x["world_frame"] == 2358)
        frame["tracked_vehicle_detections"][1]["hypothesis_track_id"] = (
            frame["tracked_vehicle_detections"][0]["hypothesis_track_id"]
        )
        with self.assertRaisesRegex(ValueError,"Duplicate track ID"):
            continuity_audit(doc)

    def test_reused_id_requires_adjacent_match(self):
        doc = tracked_fixture()
        frame = doc["frames"][1]
        frame["tracked_vehicle_detections"][0]["association_state"] = "new_unlinked_track"
        with self.assertRaisesRegex(ValueError,"reused an ID"):
            continuity_audit(doc)

    def test_wrong_class_change_flag_is_rejected(self):
        doc = tracked_fixture()
        doc["frames"][1]["tracked_vehicle_detections"][0]["label_changed_from_prior"] = True
        with self.assertRaisesRegex(ValueError,"class-change flag"):
            continuity_audit(doc)

    def test_invalid_bbox_fails_closed(self):
        doc = tracked_fixture()
        doc["frames"][0]["tracked_vehicle_detections"][0]["box_pixel_xyxy"][2] = 2000
        with self.assertRaisesRegex(ValueError,"outside"):
            continuity_audit(doc)

    def test_invalid_center_fails_closed(self):
        doc = tracked_fixture()
        doc["frames"][0]["tracked_vehicle_detections"][0]["normalized_center_x"] = .1
        with self.assertRaisesRegex(ValueError,"center x does not match"):
            continuity_audit(doc)

    def test_compare_frames_missing_without_fabrication(self):
        output = continuity_audit(tracked_fixture(),
                                  compare_frames=(2358,4000))
        self.assertEqual(output["selected_frames"]["4000"]["status"],"frame_absent")
        self.assertEqual(output["shared_ids_in_selected_frames"],[])

    def test_csv_contains_one_row_per_vehicle_box(self):
        output = continuity_audit(tracked_fixture())
        data = list(csv.DictReader(io.StringIO(csv_report(output))))
        self.assertEqual(len(data), 27)
        self.assertEqual(data[0]["world_frame"], "2349")
        self.assertEqual(data[-1]["world_frame"], "2373")

    def test_html_embeds_all_saved_frames_and_preserves_images(self):
        with tempfile.TemporaryDirectory() as folder:
            doc = tracked_fixture()
            paths = []
            for frame in doc["frames"]:
                image_path = Path(folder)/f'frame_{frame["world_frame"]:08d}.png'
                Image.new("RGB",(640,480),(180,185,190)).save(image_path)
                frame["image_path"] = str(image_path)
                paths.append(image_path)
            before = [p.read_bytes() for p in paths]
            result = continuity_audit(doc)
            html = html_report(result)
            self.assertEqual(html.count("data:image/jpeg;base64,"),25)
            self.assertIn("Track continuity matrix",html)
            self.assertIn("Frame 2373",html)
            self.assertEqual(before,[p.read_bytes() for p in paths])

    def test_html_escapes_untrusted_strings(self):
        with tempfile.TemporaryDirectory() as folder:
            doc = tracked_fixture([2358, 2359])
            for frame in doc["frames"]:
                image_path = Path(folder)/f'frame_{frame["world_frame"]:08d}.png'
                Image.new("RGB",(640,480),(180,185,190)).save(image_path)
                frame["image_path"] = str(image_path)
            doc["frames"][0]["original_yolo_rule_risk"] = '<img onerror=alert(1)>'
            result = continuity_audit(doc,compare_frames=(2358,2359))
            html = html_report(result)
            self.assertIn("&lt;img",html)
            self.assertNotIn('<img onerror=alert(1)>',html)

    def test_writer_is_append_only(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"test.json"
            new_file(path,'{"x":1}')
            with self.assertRaises(FileExistsError):
                new_file(path,'{"x":2}')
            self.assertEqual(path.read_text("utf8"),'{"x":1}')


if __name__ == "__main__":
    unittest.main()
