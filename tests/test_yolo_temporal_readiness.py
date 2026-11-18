"""Phase 4.2 YOLO schema audit tests: no CARLA truth, inference or GPU."""
from __future__ import annotations

import copy
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from scripts.yolo_temporal_readiness import (
    audit_yolo, best_matching_record, capture_images, csv_report,
    detection_summary, html_report, new_file, parse_bbox,
)


def make_image_list():
    return [
        {"world_frame": frame, "path": f"E:/demo/frame_{frame:08d}.png",
         "width": 640, "height": 480}
        for frame in range(2349, 2374)
    ]


def sample_yolo():
    frames=[]
    for frame in range(2349, 2374):
        dets=[
            {"class_name":"car", "confidence":0.6,
             "bbox": [10, 30, 60, 90]},
            {"class_name":"traffic light", "confidence":0.5},
        ]
        if frame == 2358:
            dets.extend([
                {"class_name":"car", "confidence":0.47},
                {"class_name":"motorcycle", "confidence":0.44},
                {"class_name":"traffic light", "confidence":0.38},
            ])
        if frame == 2373:
            dets=[{"class_name":"truck", "confidence":0.364}]
        risk = "medium" if frame in (2355, 2358) else "low"
        frames.append({
            "world_frame": frame,
            "scene_response": {
                "risk_level": risk,
                "detections": dets,
            }
        })
    return {"frames":frames}


class BBoxReadinessTests(unittest.TestCase):
    def test_known_25_frames_and_classes(self):
        result=audit_yolo(sample_yolo(), make_image_list())
        self.assertEqual(result["image_count"],25)
        self.assertEqual(result["risk_counts_from_original_yolo"],
                         {"low":23, "medium":2})
        self.assertEqual(
            result["selected_frame_audit"]["2358"]["detection_count"], 5)
        self.assertEqual(
            result["selected_frame_audit"]["2373"]["detection_count"], 1)
        self.assertEqual(
            result["selected_frame_audit"]["2373"]["detections"][0]["class_predicted"],"truck")
        self.assertEqual(result["selected_frame_audit"]["2373"]["raw_frame_top_level_keys"],
                         ["scene_response","world_frame"])

    def test_bbox_without_format_is_not_assumed_xyxy(self):
        result=audit_yolo(sample_yolo(),make_image_list())
        self.assertEqual(result["valid_pixel_xyxy_count"],0)
        self.assertEqual(result["bbox_availability_status"],
                         "no_trustworthy_pixel_boxes_in_existing_report")
        self.assertGreater(result["bbox_status_counts"]["bbox_format_unknown"],0)

    def test_explicit_pixel_xyxy_array_parsed(self):
        source=sample_yolo()
        source["frames"][0]["scene_response"]["detections"][0] = {
            "class_name":"car","confidence":.6,"bbox_xyxy":[10,20,110,120],
        }
        result=audit_yolo(source,make_image_list())
        self.assertEqual(result["valid_pixel_xyxy_count"],1)
        self.assertEqual(result["bbox_availability_status"],
                         "partial_pixel_boxes_for_tracking_research_only")

    def test_xyxy_dict_keys_parsed(self):
        bbox,status,key=parse_bbox(
            {"bounding_box":{"xmin":10,"ymin":20,"xmax":80,"ymax":150}},
            640,480)
        self.assertEqual(status,"pixel_xyxy_parsed")
        self.assertEqual(bbox,[10.,20.,80.,150.])
        self.assertIn("bounding_box",key)

    def test_xyxy_nested_named_format_parsed(self):
        bbox,status,key=parse_bbox(
            {"box":{"xyxy":[10,20,200,220]}},640,480)
        self.assertEqual(status,"pixel_xyxy_parsed")
        self.assertEqual(key,"box.xyxy")

    def test_normalized_coordinates_are_not_assumed_pixels(self):
        box,status,key=parse_bbox(
            {"bbox_xyxy":[0.1,0.2,0.3,0.4]},640,480)
        self.assertIsNone(box)
        self.assertEqual(status,"named_xyxy_normalized_or_ambiguous")

    def test_out_of_range_coords_are_not_parsed(self):
        bbox,status,key=parse_bbox(
            {"xyxy":[10,20,1200,300]},640,480)
        self.assertIsNone(bbox)
        self.assertEqual(status,"named_xyxy_outside_image_or_normalized")

    def test_bad_coords_fails_closed(self):
        for data in (
            {"xyxy":[30,10,20,80]},
            {"xyxy":[True,10,20,80]},
            {"xyxy":[10,20,30]},
            {"xyxy":[float("nan"),20,30,40]},
        ):
            box,status,key=parse_bbox(data,640,480)
            self.assertIsNone(box)

    def test_no_detection_array_is_explicitly_missing(self):
        source=sample_yolo()
        del source["frames"][0]["scene_response"]["detections"]
        result=audit_yolo(source,make_image_list())
        self.assertEqual(result["yolo_frame_status_counts"]["no_explicit_detections_array"],1)
        self.assertEqual(result["rows"][0]["raw_frame_top_level_keys"],
                         ["scene_response","world_frame"])

    def test_no_frame_record_never_invents_results(self):
        source=sample_yolo()
        source["frames"] = source["frames"][1:]
        result=audit_yolo(source,make_image_list())
        self.assertEqual(result["rows"][0]["record_status"],"yolo_frame_not_found")
        self.assertIsNone(result["rows"][0]["original_yolo_rule_risk"])

    def test_invalid_confidence_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Invalid confidence"):
            detection_summary({"class_name":"car","confidence":1.4},640,480,0)

    def test_unlabeled_numeric_class_does_not_become_a_name(self):
        det=detection_summary({"cls":2,"conf":.8},640,480,0)
        self.assertIsNone(det["class_predicted"])

    def test_image_indexing_needs_no_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory)
            for frame in (2358, 2373):
                Image.new("RGB",(640,480),(140,160,170)).save(
                    folder/f"frame_{frame:08d}.png")
            (folder/"metadata.json").write_text(
                '{"frames":[{"world_frame":2358,"secret":"NEVER_READ"}]}',
                encoding="utf8")
            images=capture_images(folder)
            self.assertEqual([a["world_frame"] for a in images],[2358,2373])
            self.assertEqual(images[0]["width"],640)
            self.assertNotIn("NEVER_READ",str(images))

    def test_multiple_detection_arrays_are_rejected(self):
        record={
            "world_frame":2358,
            "scene_response":{"detections":[{"class_name":"car"}]},
            "aux":{"detections":[{"class_name":"truck"}]},
        }
        from scripts.yolo_temporal_readiness import detection_array_from_record
        with self.assertRaisesRegex(ValueError,"Multiple"):
            detection_array_from_record(record)

    def test_html_and_csv_are_valid_artifacts(self):
        result=audit_yolo(sample_yolo(),make_image_list())
        html=html_report(result)
        csv_data=csv_report(result["rows"])
        self.assertIn("YOLO bounding-box readiness", html)
        self.assertIn("Frame 2358",html)
        self.assertIn("frame_00002373.png",html)
        data=list(csv.DictReader(io.StringIO(csv_data)))
        self.assertEqual(len(data),25)
        self.assertEqual(data[-1]["world_frame"],"2373")
        self.assertEqual(data[-1]["detection_count"],"1")

    def test_output_files_are_append_only(self):
        with tempfile.TemporaryDirectory() as directory:
            dest=Path(directory)/"report.json"
            new_file(dest,"test")
            with self.assertRaises(FileExistsError):
                new_file(dest,"test2")
            self.assertEqual(dest.read_text("utf8"),"test")

    def test_model_only_vision_input_provenance(self):
        result=audit_yolo(sample_yolo(),make_image_list())
        self.assertFalse(result["source_isolation"]["CARLA_metadata_or_actor_truth_read"])
        self.assertFalse(result["source_isolation"]["models_rerun"])
        self.assertFalse(result["source_isolation"]["VLM_or_Qwen_used"])


if __name__=="__main__":
    unittest.main()
