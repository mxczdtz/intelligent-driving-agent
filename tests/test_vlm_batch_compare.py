"""Unit tests for Phase 3.8 image-only SmolVLM2 batch driver."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from scripts.vlm_batch_compare import (
    make_image_messages, per_frame_report, validate_args, write_new_json,
    output_filename, REPORT_VERSION,
)
from scripts.vlm_smolvlm2_smoke import MODEL_ID, PROMPT


class VLMBatchTests(unittest.TestCase):
    def test_same_prompt_and_image_only_input(self):
        messages = make_image_messages("E:/fake/frame_00002358.png")
        self.assertEqual(messages[0]["content"][1]["text"], PROMPT)
        self.assertEqual(messages[0]["content"][0],
                         {"type": "image", "path": "E:/fake/frame_00002358.png"})
        self.assertNotIn("lead_braking", json.dumps(messages))
        self.assertNotIn("traffic_light_state", json.dumps(messages))

    def test_report_is_compatible_with_prior_single_image_schema(self):
        image = {
            "image_path": "E:/fake/frame_00002358.png",
            "image_width": 640,
            "image_height": 480,
        }
        report = per_frame_report(
            image, text="A vehicle appears ahead.",
            input_tokens=430, new_tokens=9,
            preprocess_ms=25.1, generation_ms=123.4,
            model_load_ms=2500, peak_cuda_allocated_mib=1100,
            device="cuda", torch_version="2.11.0+cu128",
            model_dir=Path("E:/models/SmolVLM2-500M-Video-Instruct"),
            image_longest_edge=640,
        )
        self.assertEqual(report["model_id"], MODEL_ID)
        self.assertEqual(report["image_path"], image["image_path"])
        self.assertEqual(report["output_text"], "A vehicle appears ahead.")
        self.assertEqual(report["generation_ms"], 123.4)
        self.assertEqual(report["batch_report_version"], REPORT_VERSION)

    def test_no_saved_output_for_empty_text(self):
        image = {"image_path":"fake.png","image_width":640,"image_height":480}
        with self.assertRaisesRegex(ValueError, "empty"):
            per_frame_report(
                image, text=" ", input_tokens=10, new_tokens=0,
                preprocess_ms=1, generation_ms=1, model_load_ms=1,
                peak_cuda_allocated_mib=None, device="cpu", torch_version="test",
                model_dir=Path("."), image_longest_edge=640,
            )

    def test_file_writer_never_overwrites_existing(self):
        with tempfile.TemporaryDirectory() as dirname:
            folder = Path(dirname)
            dest = write_new_json(folder, "test.json", {"a":1})
            self.assertEqual(json.loads(dest.read_text(encoding="utf8")), {"a":1})
            with self.assertRaises(FileExistsError):
                write_new_json(folder, "test.json", {"a":2})
            self.assertEqual(json.loads(dest.read_text(encoding="utf8")), {"a":1})

    def test_validates_exact_image_files_not_metadata(self):
        with tempfile.TemporaryDirectory() as dirname:
            root = Path(dirname)
            x = root/"frame_00002358.png"
            y = root/"frame_00052924.png"
            Image.new("RGB",(640,480),(128,128,128)).save(x)
            Image.new("RGB",(640,480),(128,128,128)).save(y)
            args = argparse.Namespace(
                images=[x,y], max_new_tokens=96,image_longest_edge=640,
            )
            entries = validate_args(args, root)
            self.assertEqual(len(entries),2)
            self.assertEqual(entries[0]["image_width"], 640)
            self.assertEqual(entries[1]["image_filename"], "frame_00052924.png")
            # No metadata input or simulator truth included anywhere.
            self.assertNotIn("metadata", str(entries))

    def test_duplicate_image_is_rejected(self):
        with tempfile.TemporaryDirectory() as dirname:
            root = Path(dirname)
            path = root/"frame_00002358.png"
            Image.new("RGB",(10,10)).save(path)
            args=argparse.Namespace(images=[path,path], max_new_tokens=96,
                                    image_longest_edge=640)
            with self.assertRaisesRegex(ValueError,"Duplicate image"):
                validate_args(args,root)

    def test_missing_image_is_reported(self):
        with tempfile.TemporaryDirectory() as dirname:
            root=Path(dirname)
            args=argparse.Namespace(images=[root/"frame_00002358.png"],
                                    max_new_tokens=96,image_longest_edge=640)
            with self.assertRaises(FileNotFoundError):
                validate_args(args,root)

    def test_bad_batch_options_are_rejected(self):
        with tempfile.TemporaryDirectory() as dirname:
            root=Path(dirname)
            args=argparse.Namespace(images=[],max_new_tokens=96,image_longest_edge=640)
            with self.assertRaises(ValueError):
                validate_args(args,root)
            args.images = [Path("whatever")]
            args.image_longest_edge = 200
            with self.assertRaises(ValueError):
                validate_args(args,root)

    def test_output_filename_is_compatible_for_scene_alignment(self):
        name = output_filename()
        self.assertTrue(name.startswith("vlm_single_image_"))
        self.assertTrue(name.endswith(".json"))


if __name__ == "__main__":
    unittest.main()
