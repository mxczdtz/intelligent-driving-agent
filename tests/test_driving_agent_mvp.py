"""MVP integration smoke tests — model calls mocked; no downloads, GPUs or CARLA."""
from __future__ import annotations

import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from PIL import Image

from scripts.driving_agent_mvp import (
    canonical_path, evidence_registry, guarded_decision,
    html_result, load_reference_sections, prompt_for_qwen, qwen_messages,
    retrieve_knowledge, run_local_qwen, safely_join_vlm, select_visual_candidate,
    validate_qwen_selection,
)
from scripts.yolo_track_continuity import continuity_audit
from scripts.yolo_vehicle_box_tracks import track_frames


def make_yolo_traces(project: Path) -> tuple[dict, Path]:
    """Synthetic YOLO record including a long centered car track."""
    frames = []
    capture = project / "evidence" / "carla_lead_brake" / "mock_run"
    capture.mkdir(parents=True, exist_ok=True)
    count = 0
    for frame in range(2349,2374):
        rgb = capture / f"frame_{frame:08d}.png"
        Image.new("RGB", (640,480), (175,185,200)).save(rgb)
        dets = []
        # center car remains visible through 2372, unlabeled at last image
        if frame <=2372:
            slide = frame - 2349
            dets.append({
                "index":0, "class_predicted":"car", "confidence":.8,
                "bbox_pixel_xyxy":[273+slide//2,214,344+slide//2,320],
                "bbox_parse_status":"pixel_xyxy_parsed",
                "bbox_source_field":"box_xyxy",
            })
        # separate left-side track from frame 2367 with class change
        if frame>=2367:
            dets.append({
                "index":1, "class_predicted":"car" if frame<=2370 else "truck",
                "confidence":.6,
                "bbox_pixel_xyxy":[45,210,115,295],
                "bbox_parse_status":"pixel_xyxy_parsed",
                "bbox_source_field":"box_xyxy",
            })
        count += len(dets)
        frames.append({
            "world_frame":frame,
            "image_path":str(rgb),
            "image_width":640,"image_height":480,
            "original_yolo_rule_risk":"medium" if frame in (2355,2358) else "low",
            "record_status":"detections_array_parsed",
            "detection_count":len(dets), "detections":dets,
        })
    doc={
        "schema_version":"yolo_temporal_readiness.v1",
        "image_count":len(frames),
        "raw_detection_count":count,
        "valid_pixel_xyxy_count":count,
        "source_isolation":{
            "CARLA_metadata_or_actor_truth_read":False,
            "VLM_or_Qwen_used":False, "models_rerun":False,
        },
        "rows":frames,
        "carla_metadata_actor_truth":"NEVER_IN_PROMPT",
    }
    tracks = track_frames(doc)
    path = project / "evidence" / "yolo_vehicle_box_tracks" / "vehicle_tracks_fake.json"
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(tracks,ensure_ascii=False,indent=2),encoding="utf8")
    return tracks, path


class DrivingMVPTests(unittest.TestCase):
    def test_selects_long_center_candidate_without_carla_truth(self):
        with tempfile.TemporaryDirectory() as dirname:
            tracks,_=make_yolo_traces(Path(dirname))
            audit=continuity_audit(tracks)
            candidate=select_visual_candidate(audit)
            self.assertIsNotNone(candidate)
            self.assertGreaterEqual(candidate["observed_frames"],20)
            self.assertTrue(.35 <= candidate["first_center"][0] <= .65)
            self.assertNotIn("NEVER_IN_PROMPT",str(audit))

    def test_evidence_is_not_physical_hazard_label(self):
        with tempfile.TemporaryDirectory() as dirname:
            tracks,_=make_yolo_traces(Path(dirname))
            audit=continuity_audit(tracks)
            f,l,ctx=evidence_registry(audit,select_visual_candidate(audit),{})
            self.assertTrue(ctx["selected_track_ends_before_final_frame"])
            self.assertIn("vision:candidate_missing_at_end",[x["id"] for x in f])
            self.assertIn("不",next(x["statement"] for x in f if x["id"]==
                                 "vision:box_area_trend"))
            self.assertNotIn("NEVER_IN_PROMPT",str(f))
            self.assertIn("limit:unknown_geometry",[x["id"] for x in l])

    def test_strict_id_selection_and_guard(self):
        f=[
            {"id":"vision:rule", "statement":"sample", "source_refs":["yolo:0"],
             "epistemic_status":"unverified"}
        ]
        limits=[{"id":"limit:unknown_geometry","text":"No geometry"}]
        chosen={
            "risk_level":"high",
            "selected_evidence_ids":["vision:rule"],
            "uncertainty_ids":["limit:unknown_geometry"],
        }
        selected=validate_qwen_selection(json.dumps(chosen),f,limits)
        decision=guarded_decision({"selection":selected},f,limits)
        self.assertEqual(decision["risk_level"],"unknown")
        self.assertEqual(decision["selected_factors"][0]["source_refs"],["yolo:0"])
        chosen["selected_evidence_ids"]=["vision:invented"]
        with self.assertRaisesRegex(ValueError,"unknown evidence ID"):
            validate_qwen_selection(json.dumps(chosen),f,limits)

    def test_invalid_json_and_extra_keys_are_rejected(self):
        f=[{"id":"vision:a"}]
        l=[{"id":"limit:x"}]
        with self.assertRaises(ValueError):
            validate_qwen_selection(
                '{"risk_level":"unknown","selected_evidence_ids":[],'
                '"uncertainty_ids":["limit:x"],"driving_command":"accelerate"}', f,l)
        with self.assertRaises(ValueError):
            validate_qwen_selection(
                '{"risk_level":"high","risk_level":"unknown",'
                '"selected_evidence_ids":[],"uncertainty_ids":["limit:x"]}',f,l)

    def test_simulated_model_no_selection_is_safe_unknown(self):
        result=guarded_decision({"selection":None},[],[
            {"id":"limit:x","text":"test"}
        ])
        self.assertEqual(result["risk_level"],"unknown")
        self.assertEqual(result["selected_factors"],[])
        self.assertIn("fallback",result["decision_status"])

    def test_knowledge_retrieval_keeps_background_role(self):
        with tempfile.TemporaryDirectory() as dirname:
            project=Path(dirname)
            folder=project/"knowledge"
            folder.mkdir()
            (folder/"mvp_risk_knowledge.md").write_text(
                "# Notes\n## Track tracking\nTrack association is not a proven actor ID.\n"
                "## RAG evidence\nRetrieval is background, not model truth.\n",
                encoding="utf8"
            )
            sources=load_reference_sections(project)
            items=retrieve_knowledge(sources,"Track tracking",2)
            self.assertTrue(items)
            self.assertTrue(all(x["role"]==
                "background_reference_not_visual_observation" for x in items))
            self.assertTrue(all(str(folder) in x["source"] for x in items))

    def test_saved_vlm_reuse_requires_exact_full_image_path(self):
        with tempfile.TemporaryDirectory() as dirname:
            project=Path(dirname)
            image_a=project/"evidence"/"carla_lead_brake"/"run_A"/"frame_00002358.png"
            image_b=project/"evidence"/"carla_lead_brake"/"run_B"/"frame_00002358.png"
            image_a.parent.mkdir(parents=True)
            image_b.parent.mkdir(parents=True)
            Image.new("RGB",(640,480)).save(image_a)
            Image.new("RGB",(640,480)).save(image_b)
            folder=project/"evidence"/"vlm_smolvlm2"
            folder.mkdir(parents=True)
            (folder/"vlm_single_image_saved.json").write_text(json.dumps({
                "mode":"image_only_vlm_smoke_test",
                "image_path":str(image_a),
                "output_text":"There is a red car.",
                "generation_ms":2100,
            }), encoding="utf8")
            matched,missing=safely_join_vlm(project,{
                2358:str(image_b)
            })
            self.assertFalse(matched)
            self.assertEqual(missing,[2358])
            matched,missing=safely_join_vlm(project,{
                2358:str(image_a)
            })
            self.assertEqual(sorted(matched),[2358])
            self.assertFalse(missing)

    def test_model_prompt_has_no_carla_ground_truth_or_speed(self):
        f=[
            {"id":"vision:c","statement":"unverified car box",
             "epistemic_status":"model_prediction"}
        ]
        limits=[{"id":"limit:g","text":"unknown geometry"}]
        prompt=prompt_for_qwen(f,limits,[],list(range(2349,2374)))
        text=json.dumps(qwen_messages(prompt),ensure_ascii=False)
        self.assertIn("vision:c",text)
        self.assertNotIn("CARLA actor ID:",text)
        self.assertNotIn("NEVER_IN_PROMPT",text)

    def test_one_command_mvp_writes_25frame_player_model_mock(self):
        # Simulates actual end-to-end file IO and HTML without running GPU
        # inside unit tests. On the user's PC, model calls run for real.
        with tempfile.TemporaryDirectory() as tmp:
            project=Path(tmp)
            tracks,track_path=make_yolo_traces(project)
            (project/"knowledge").mkdir()
            (project/"knowledge"/"mvp_risk_knowledge.md").write_text(
                "# Notes\n## Tracking\nTracking boxes are unverified.\n",
                encoding="utf8"
            )
            qwen_dir=project/"qwen_mock"
            qwen_dir.mkdir()
            for frame in (2358,2366,2373):
                img=project/"evidence"/"carla_lead_brake"/"mock_run"/f"frame_{frame:08d}.png"
                vlm_dir=project/"evidence"/"vlm_smolvlm2"
                vlm_dir.mkdir(parents=True,exist_ok=True)
                (vlm_dir/f"vlm_single_image_{frame}.json").write_text(
                    json.dumps({
                        "mode":"image_only_vlm_smoke_test",
                        "image_path":str(img),
                        "output_text":f"Unverified visual description of frame {frame}",
                        "generation_ms":1200.0,
                    }), encoding="utf8",
                )

            from scripts import driving_agent_mvp as module
            project_root=Path(module.__file__).resolve().parent.parent
            # Read-only project root simulation: patch Path.__... not desired,
            # instead patch module's __file__ in the integration call.
            def fake_qwen(model_dir,payload,factors,limits,device):
                return {
                    "validation_status":"valid",
                    "selection":{
                        "risk_level":"unknown",
                        "selected_evidence_ids":["vision:original_rule_distribution"],
                        "uncertainty_ids":["limit:unknown_geometry"],
                    },
                    "attempts":[{"attempt":1,"json_valid":True}],
                    "model_load_ms":10.0, "generation_total_ms":20.0,
                }
            import sys
            argv=[
                "driving_agent_mvp",
                "--tracks-report",str(track_path),
                "--qwen-model-dir",str(qwen_dir),
            ]
            with mock.patch.object(module,"__file__",str(project/"scripts"/"driving_agent_mvp.py")), \
                 mock.patch.object(module,"run_local_qwen",side_effect=fake_qwen), \
                 mock.patch.object(sys,"argv",argv), \
                 mock.patch("sys.stdout",new_callable=io.StringIO) as out:
                status=module.main()
            self.assertEqual(status,0)
            self.assertIn("PASS: YOLO -> temporal boxes",out.getvalue())
            saved=list((project/"evidence"/"driving_agent_mvp").glob("*.json"))
            pages=list((project/"evidence"/"driving_agent_mvp").glob("*.html"))
            self.assertEqual((len(saved),len(pages)),(1,1))
            result=json.loads(saved[0].read_text("utf8"))
            html=pages[0].read_text("utf8")
            self.assertEqual(result["frame_count"],25)
            self.assertEqual(sorted(result["vlm_keyframes"]),["2358","2366","2373"])
            self.assertEqual(result["qwen_research_decision"]["risk_level"],"unknown")
            self.assertEqual(html.count("data:image/jpeg;base64,"),25)
            self.assertIn("id=\"playBtn\"",html)
            self.assertNotIn("NEVER_IN_PROMPT",html)
            self.assertTrue(result["source_isolation"]["qwen_input_contains_CARLA_truth"] is False)


if __name__=="__main__":
    unittest.main()
