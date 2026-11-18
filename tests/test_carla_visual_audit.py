"""Phase 3.9 visual-audit offline tests (no GPU or CARLA)."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from scripts.carla_visual_audit import (
    SCENES, build_record, html_page, require_one_frame,
    save_without_overwrite, pick_yolo_record,
)


def local_fixture(root: Path) -> None:
    (root / "evidence" / "vlm_smolvlm2").mkdir(parents=True, exist_ok=True)
    for scenario in SCENES:
        capture = root / "evidence" / scenario["capture_relative"]
        capture.mkdir(parents=True, exist_ok=True)
        num = scenario["frame"]
        image = capture / f"frame_{num:08d}.png"
        Image.new("RGB", (640, 480), (110, 140, 150)).save(image)

        metadata_path = capture / "metadata.json"
        if metadata_path.exists():
            meta = json.loads(metadata_path.read_text("utf-8"))
        else:
            meta = {"frames": []}
        truth = {
            "world_frame": num, "image_frame": num,
            "image_file": image.name, "ego_speed_kmh": 10.795,
        }
        if scenario["scenario"] == "traffic_light":
            truth["traffic_light_state"] = "Red"
            truth["traffic_light_command"] = "Red"
        else:
            truth["phase"] = "lead_braking"
            truth["lead_speed_kmh"] = 20
        meta["frames"].append(truth)
        metadata_path.write_text(json.dumps(meta), encoding="utf8")

        yolo_path = capture / "yolo_risk_fixture.json"
        if yolo_path.exists():
            yolo = json.loads(yolo_path.read_text("utf8"))
        else:
            yolo = {"frames": []}
        yolo["frames"].append({
            "world_frame": num,
            "scene_response": {
                "risk_level": "low",
                "detections": [
                    {"class_name": "car", "confidence": .76}
                ],
            },
        })
        yolo_path.write_text(json.dumps(yolo), encoding="utf8")

        path = root / "evidence" / "vlm_smolvlm2" / f"vlm_single_image_{num}.json"
        vlm_text = ("A yellow traffic light is visible in the image"
                    if scenario["scenario"] == "traffic_light"
                    else "A red car is driving away.")
        path.write_text(json.dumps({
            "image_path": str(image),
            "output_text": vlm_text,
            "generation_ms": 3100.5,
        }), encoding="utf8")


class VisualAuditTests(unittest.TestCase):
    def test_existing_scenes_are_three_unique_frames(self):
        self.assertEqual([s["frame"] for s in SCENES], [2358, 2373, 52924])

    def test_build_three_rows_from_only_local_sources(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            local_fixture(root)
            rows = [build_record(root, s) for s in SCENES]
            self.assertEqual(len(rows), 3)
            self.assertEqual([r["yolo_detection_count"] for r in rows], [1, 1, 1])
            self.assertEqual(rows[2]["simulator_context_for_human_audit_only"][
                "traffic_light_state"], "Red")
            self.assertEqual(rows[2]["review_status"], "not_yet_visually_checked")
            self.assertNotIn("traffic_light_state",
                             rows[2]["vlm_description_unverified"])

    def test_html_contains_actual_images_and_unverified_labels(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            local_fixture(root)
            html = html_page([build_record(root, s) for s in SCENES])
            self.assertEqual(html.count("data:image/jpeg;base64,"), 3)
            self.assertIn("The text is a prediction, not ground truth", html)
            self.assertIn("FRAME 52924", html)
            self.assertIn("Traffic-light sequence", html)

    def test_html_escapes_untrusted_vlm_description(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            local_fixture(root)
            row = build_record(root, SCENES[0])
            row["vlm_description_unverified"] = '<script>alert("bad")</script>'
            html = html_page([row])
            self.assertIn("&lt;script&gt;", html)
            self.assertNotIn('<script>alert("bad")</script>', html)

    def test_wrong_vlm_image_never_joined(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            local_fixture(root)
            vlm = root / "evidence" / "vlm_smolvlm2" / "vlm_single_image_2358.json"
            data = json.loads(vlm.read_text("utf8"))
            data["image_path"] = "C:/different_run/frame_00002358.png"
            vlm.write_text(json.dumps(data), encoding="utf8")
            with self.assertRaisesRegex(ValueError, "No VLM report"):
                build_record(root, SCENES[0])

    def test_missing_yolo_frame_is_error(self):
        with self.assertRaisesRegex(ValueError, "absent"):
            pick_yolo_record({"frames": [{"world_frame": 1}]}, 100)

    def test_frame_count_mismatch_is_error(self):
        with self.assertRaises(ValueError):
            require_one_frame({"frames": []}, 2358)
        with self.assertRaises(ValueError):
            require_one_frame({"frames": [
                {"world_frame": 2358},
                {"world_frame": 2358},
            ]}, 2358)

    def test_json_writer_does_not_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp) / "report.json"
            save_without_overwrite(p, '{"a": 1}')
            with self.assertRaises(FileExistsError):
                save_without_overwrite(p, '{"a": 2}')
            self.assertEqual(p.read_text(encoding="utf8"), '{"a": 1}')


if __name__ == "__main__":
    unittest.main()
