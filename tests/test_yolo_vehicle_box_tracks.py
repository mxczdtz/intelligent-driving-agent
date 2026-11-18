"""Phase 4.3 conservative YOLO box tracking tests; no models or CARLA needed."""
from __future__ import annotations

import csv
import io
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from scripts.yolo_vehicle_box_tracks import (
    associate_two_frames, box_metrics, candidate_edge, csv_report,
    html_report, iou, normalize_frames, save_new, track_frames,
)

def det(label="car", box=(270, 200, 350, 320), index=0):
    return {
        "index":index,
        "class_predicted":label,
        "confidence":.8,
        "bbox_pixel_xyxy":list(box),
        "bbox_parse_status":"pixel_xyxy_parsed",
        "bbox_source_field":"box_xyxy",
        "raw_detection_keys_and_types":{
            "class_name":"str","box_xyxy":"list","confidence":"float"},
    }

def source(frames=(2358,2359,2360,2361,2373)):
    rows=[]
    total=0
    for frame in frames:
        shifted = frame-2358 if frame <= 2361 else 4
        box=(270+shifted,200+shifted,350+shifted,320+shifted)
        if frame == 2358:
            detections=[
                det("car",box,0),
                det("traffic light",(200,45,215,80),1),
                det("car",(80,190,120,250),2),
                det("motorcycle",(540,245,580,300),3),
                det("traffic light",(430,55,445,85),4),
            ]
        else:
            detections=[det("truck" if frame == 2373 else "car",box,0)]
        rows.append({
            "world_frame":frame,
            "image_path":f"E:/data/frame_{frame:08d}.png",
            "image_width":640,"image_height":480,
            "original_yolo_rule_risk":"medium" if frame ==2358 else "low",
            "record_status":"detections_array_parsed",
            "detection_count":len(detections),
            "detections":detections,
        })
        total+=len(detections)
    return {
        "schema_version":"yolo_temporal_readiness.v1",
        "image_count":len(rows),
        "raw_detection_count":total,
        "valid_pixel_xyxy_count":total,
        "bbox_availability_status":"pixel_boxes_present_but_identity_tracking_unverified",
        "source_isolation":{
            "CARLA_metadata_or_actor_truth_read":False,
            "VLM_or_Qwen_used":False,
            "models_rerun":False,
        },
        "rows":rows,
        "secret_carla_truth":"NEVER_USE_FOR_TRACKING",
    }

class VisionTrackingTests(unittest.TestCase):
    def test_frame_input_keeps_only_vehicle_classes(self):
        rows=normalize_frames(source((2358,)))
        self.assertEqual(len(rows[0]["vehicle_detections"]),2)
        self.assertEqual(
            [item["class_predicted"] for item in rows[0]["vehicle_detections"]],
            ["car","car"],
        )

    def test_iou_and_center_math(self):
        a=[10,10,30,30]
        self.assertAlmostEqual(iou(a,a),1.0)
        self.assertAlmostEqual(iou(a,[30,30,50,50]),0)
        self.assertAlmostEqual(iou(a,[20,10,40,30]),1/3)
        features=box_metrics([0,0,320,240],640,480)
        self.assertAlmostEqual(features["bbox_area_fraction"],.25)
        self.assertAlmostEqual(features["normalized_center_x"],.25)

    def test_adjacent_car_to_truck_can_be_linked_as_hypothesis(self):
        doc=source((2358,2359))
        doc["rows"][1]["detections"][0]["class_predicted"]="truck"
        result=track_frames(doc,compare_frames=(2358,2359))
        first=result["frame_comparisons"]["2358"]["vehicle_detections"][0]
        second=result["frame_comparisons"]["2359"]["vehicle_detections"][0]
        self.assertEqual(first["hypothesis_track_id"],second["hypothesis_track_id"])
        self.assertTrue(second["label_changed_from_prior"])
        self.assertEqual(result["class_change_on_tentative_link_count"],1)
        self.assertEqual(result["tentative_adjacent_links"],1)

    def test_distant_frame_does_not_bridge_gap(self):
        result=track_frames(source((2358,2373)),compare_frames=(2358,2373))
        self.assertEqual(result["same_tentative_ids_in_comparison_frames"],[])
        self.assertEqual(result["tentative_adjacent_links"],0)
        self.assertEqual(result["class_change_on_tentative_link_count"],0)

    def test_realistic_2358_2373_comparison_not_assumed_same_actor(self):
        result=track_frames(source())
        self.assertEqual(result["world_frame_count"],5)
        a=result["frame_comparisons"]["2358"]
        b=result["frame_comparisons"]["2373"]
        self.assertEqual(a["original_yolo_rule_risk"],"medium")
        self.assertEqual(a["all_detection_count"],5)
        self.assertEqual(b["original_yolo_rule_risk"],"low")
        self.assertEqual(b["all_detection_count"],1)
        self.assertEqual(result["same_tentative_ids_in_comparison_frames"],[])

    def test_ambiguous_prior_candidates_not_linked(self):
        a={
            "image_width":640,"image_height":480,
            "class_predicted":"car",
            "normalized_center_x":.5,"normalized_center_y":.5,
            "bbox_area_fraction":.04,
            "box_pixel_xyxy":[270,200,370,300],
        }
        b=dict(a)
        current=[dict(a)]
        matches, ambiguous=associate_two_frames([a,b],current)
        self.assertEqual(matches,[])
        self.assertEqual(ambiguous,{0})

    def test_center_displacement_rejects_far_object(self):
        first={
            "image_width":640,"image_height":480,"class_predicted":"car",
            "normalized_center_x":.1,"normalized_center_y":.5,
            "bbox_area_fraction":.03,"box_pixel_xyxy":[10,200,90,310],
        }
        second={
            **first,
            "normalized_center_x":.9,
            "box_pixel_xyxy":[540,200,620,310],
        }
        self.assertIsNone(candidate_edge(first,second))

    def test_bbox_unparsed_does_not_create_track(self):
        doc=source((2358,))
        row=doc["rows"][0]
        row["detections"][0]["bbox_parse_status"]="bbox_format_unknown"
        row["detections"][0]["bbox_pixel_xyxy"]=None
        rows=normalize_frames(doc)
        self.assertEqual(len(rows[0]["vehicle_detections"]),1)

    def test_bad_bbox_status_with_coords_rejected(self):
        doc=source((2358,))
        doc["rows"][0]["detections"][0]["bbox_parse_status"]="bbox_format_unknown"
        with self.assertRaisesRegex(ValueError,"Non-XYXY"):
            normalize_frames(doc)

    def test_bbox_outside_image_rejected(self):
        doc=source((2358,))
        doc["rows"][0]["detections"][0]["bbox_pixel_xyxy"][2]=999
        with self.assertRaisesRegex(ValueError,"outside"):
            normalize_frames(doc)

    def test_nonfinite_confidence_rejected(self):
        doc=source((2358,))
        doc["rows"][0]["detections"][0]["confidence"]=float("nan")
        with self.assertRaisesRegex(ValueError,"finite"):
            normalize_frames(doc)

    def test_bad_provenance_not_accepted(self):
        doc=source((2358,))
        doc["source_isolation"]["CARLA_metadata_or_actor_truth_read"]=True
        with self.assertRaisesRegex(ValueError,"CARLA truth"):
            normalize_frames(doc)

    def test_duplicate_frames_rejected(self):
        doc=source((2358,2359))
        doc["rows"][1]["world_frame"]=2358
        with self.assertRaisesRegex(ValueError,"increasing"):
            normalize_frames(doc)

    def test_track_logic_independent_of_simulator_truth(self):
        doc=source((2358,2359))
        res=track_frames(doc,compare_frames=(2358,2359))
        doc["secret_carla_truth"]="opposite simulated truth"
        res2=track_frames(doc,compare_frames=(2358,2359))
        self.assertEqual(res,res2)
        self.assertFalse(res["source_isolation"]["CARLA_metadata_or_actor_truth_read"])

    def test_track_is_not_physical_risk(self):
        result=track_frames(source())
        self.assertNotIn("ttc_s",result)
        self.assertNotIn("danger_level",result)
        self.assertIn("hypothesis",result["track_summaries"][0]["identity_status"])
        self.assertIn("NOT calibrated",result["limitations"][5])

    def test_csv_contains_each_vehicle_box(self):
        report=track_frames(source((2358,2359)))
        rows=list(csv.DictReader(io.StringIO(csv_report(report))))
        self.assertEqual(len(rows),3)
        self.assertEqual(rows[0]["world_frame"],"2358")
        self.assertEqual(rows[-1]["world_frame"],"2359")

    def test_html_overlays_saved_images_without_modifying_originals(self):
        with tempfile.TemporaryDirectory() as temp:
            doc=source((2358,2359))
            for item in doc["rows"]:
                path=Path(temp)/f'frame_{item["world_frame"]:08d}.png'
                Image.new("RGB",(640,480),(80,110,150)).save(path)
                item["image_path"]=str(path)
            original_data = [Path(r["image_path"]).read_bytes() for r in doc["rows"]]
            result=track_frames(doc,compare_frames=(2358,2359))
            html=html_report(result,(2358,2359))
            self.assertEqual(html.count("data:image/jpeg;base64,"),2)
            self.assertIn("Frame 2358",html)
            self.assertIn("Track hypotheses, not actor IDs",html)
            self.assertEqual(
                original_data,
                [Path(r["image_path"]).read_bytes() for r in doc["rows"]],
            )

    def test_writer_does_not_overwrite_existing_artifact(self):
        with tempfile.TemporaryDirectory() as temp:
            output=Path(temp)/"report.json"
            save_new(output,'{"test":1}')
            with self.assertRaises(FileExistsError):
                save_new(output,'{"test":2}')
            self.assertEqual(output.read_text("utf8"),'{"test":1}')


if __name__=="__main__":
    unittest.main()
