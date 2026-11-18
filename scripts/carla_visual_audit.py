r"""
Phase 3.9 — multi-scene visual audit, using only EXISTING local artifacts.

Produce a standalone HTML contact sheet (embedded resized CARLA PNG frames),
plus a machine-readable JSON index, to review *unverified* VLM descriptions
against YOLO detections and simulator audit metadata. No model inference.

From the project root:
  & $PY -m scripts.carla_visual_audit

Outputs:
  evidence/carla_visual_audit/visual_audit_*.html
  evidence/carla_visual_audit/visual_audit_*.json

The CARLA ground truth is displayed ONLY for human audit. It is never used
as model input. The script never changes or overwrites original reports.
Requires Pillow already installed in llm-gpu, no new dependencies.
"""
from __future__ import annotations

import argparse
import base64
from datetime import datetime
from html import escape
from io import BytesIO
import json
from pathlib import Path
import re
import sys
from typing import Any

from PIL import Image

from scripts.scene_fusion_preview import (
    detection_info, find_records, frame_matches, latest_yolo,
    load, matching_vlm, original_risk,
)

SCENES = (
    {
        "name": "Lead vehicle — pre-brake",
        "scenario": "lead_brake",
        "capture_relative": "carla_lead_brake/20261116_140208_590540",
        "frame": 2358,
    },
    {
        "name": "Lead vehicle — late braking",
        "scenario": "lead_brake",
        "capture_relative": "carla_lead_brake/20261116_140208_590540",
        "frame": 2373,
    },
    {
        "name": "Traffic-light sequence — planned red phase",
        "scenario": "traffic_light",
        "capture_relative": "carla_traffic_light/20261116_142759_601135",
        "frame": 52924,
    },
)
AUDIT_META_KEYS = {
    "lead_brake": (
        "phase", "scenario_step", "ego_speed_kmh", "lead_speed_kmh",
        "longitudinal_origin_separation_m", "closing_speed_mps",
        "lead_brake_command",
    ),
    "traffic_light": (
        "traffic_light_command", "traffic_light_state",
        "ego_speed_kmh", "stop_waypoint_longitudinal_distance_m",
        "traffic_light_id",
    ),
}


def require_one_frame(metadata: dict[str, Any], frame_id: int) -> dict[str, Any]:
    rows = metadata.get("frames")
    if not isinstance(rows, list):
        raise ValueError("metadata.json does not have a frames array.")
    matches = [r for r in rows if frame_matches(r, frame_id)]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one metadata record for {frame_id}, found {len(matches)}."
        )
    return matches[0]


def pick_yolo_record(doc: Any, frame_id: int) -> dict[str, Any]:
    matches = list(find_records(doc, frame_id))
    if not matches:
        raise ValueError(f"Frame {frame_id} is absent from YOLO JSON.")
    return max(matches, key=lambda x: len(json.dumps(x, ensure_ascii=False)))


def build_record(
    root: Path, descriptor: dict[str, Any],
    vlm_path_override: Path | None = None,
) -> dict[str, Any]:
    capture_dir = root / "evidence" / descriptor["capture_relative"]
    metadata_path = capture_dir / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"No metadata.json: {metadata_path}")
    metadata = load(metadata_path)
    frame = descriptor["frame"]
    truth = require_one_frame(metadata, frame)
    filename = truth.get("image_file")
    if not isinstance(filename, str) or not filename.strip():
        raise ValueError(f"metadata for {frame} has no image_file.")
    image_path = (capture_dir / filename).resolve()
    if not image_path.is_relative_to(capture_dir.resolve()):
        raise ValueError(f"Image path points outside capture directory: {image_path}")
    if not image_path.is_file():
        raise FileNotFoundError(f"Missing image: {image_path}")

    yolo_report = latest_yolo(capture_dir)
    yolo_frame = pick_yolo_record(load(yolo_report), frame)
    yolo = detection_info(yolo_frame)
    if yolo["count"] is None:
        raise ValueError(f"Could not extract YOLO detection count for {frame}.")
    vlm_report, vlm = matching_vlm(root, image_path, vlm_path_override)
    description = vlm.get("output_text")
    if not isinstance(description, str) or not description.strip():
        raise ValueError(f"VLM description for frame {frame} is empty.")
    if not isinstance(vlm.get("generation_ms"), (int, float)):
        raise ValueError(f"VLM generation time missing for frame {frame}.")

    # Whitelist ground-truth keys, and never treat actor light state
    # as proof that the same light or color is visible in the image.
    context = {
        key: truth[key]
        for key in AUDIT_META_KEYS[descriptor["scenario"]]
        if key in truth
    }
    return {
        "scene": descriptor["name"],
        "scenario": descriptor["scenario"],
        "frame": frame,
        "image_path": str(image_path),
        "yolo_report_path": str(yolo_report),
        "vlm_report_path": str(vlm_report),
        "metadata_path": str(metadata_path),
        "yolo_baseline_risk": original_risk(yolo_frame),
        "yolo_detection_count": yolo["count"],
        "yolo_objects_unverified": yolo["objects"],
        "vlm_description_unverified": description.strip(),
        "vlm_generation_ms": vlm["generation_ms"],
        "simulator_context_for_human_audit_only": context,
        "review_status": "not_yet_visually_checked",
        "visual_review_prompts": [
            "Are the described road users actually visible, with the correct types and counts?",
            "Can the claimed traffic-light color be read in this specific image?",
            "Are the claimed signs or scooters present, or invented by the VLM?",
            "Do not infer braking, time-to-collision, or motion direction from a single frame.",
        ],
    }


def encode_thumbnail(path: Path) -> str:
    with Image.open(path) as im:
        if im.width <= 0 or im.height <= 0 or im.width * im.height > 16_000_000:
            raise ValueError(f"Image dimensions are not supported: {path}")
        rgb = im.convert("RGB")
        rgb.thumbnail((960, 720), Image.Resampling.LANCZOS)
        out = BytesIO()
        rgb.save(out, format="JPEG", quality=88, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(out.getvalue()).decode("ascii")


def short_objects(objects: Any) -> str:
    if objects is None:
        return "Only a detection count is available in the YOLO report."
    if not isinstance(objects, list):
        return "Could not display structured detector objects."
    parts = []
    for obj in objects:
        if not isinstance(obj, dict):
            continue
        label = str(obj.get("class") or "unknown")
        confidence = obj.get("confidence")
        if isinstance(confidence, (float, int)) and not isinstance(confidence, bool):
            parts.append(f"{label} ({float(confidence):.3f})")
        else:
            parts.append(label)
    return ", ".join(parts) if parts else "(none)"


def html_page(records: list[dict[str, Any]]) -> str:
    cards: list[str] = []
    for record in records:
        img_uri = encode_thumbnail(Path(record["image_path"]))
        description = escape(record["vlm_description_unverified"])
        meta = escape(json.dumps(
            record["simulator_context_for_human_audit_only"],
            ensure_ascii=False, indent=2,
        ))
        objects = escape(short_objects(record["yolo_objects_unverified"]))
        image_name = escape(Path(record["image_path"]).name)
        title = escape(record["scene"])
        cards.append(f"""
        <section class="scene">
            <header><span class="index">FRAME {record['frame']}</span><h2>{title}</h2></header>
            <div class="two">
                <figure>
                    <img src="{img_uri}" alt="Original CARLA RGB frame {record['frame']}">
                    <figcaption>{image_name} · original frame (review visually)</figcaption>
                </figure>
                <div class="notes">
                    <h3>Unverified VLM text</h3>
                    <p>{description}</p>
                    <h3>YOLO baseline (independent prediction)</h3>
                    <p>Risk: <strong>{escape(str(record['yolo_baseline_risk']))}</strong> ·
                    Objects: {record['yolo_detection_count']}</p>
                    <p>{objects}</p>
                    <p class="metric">VLM generation: {record['vlm_generation_ms']:.2f} ms</p>
                </div>
            </div>
            <details><summary>CARLA simulator metadata — audit only</summary><pre>{meta}</pre></details>
            <div class="review">
                <strong>Review against the actual image</strong>
                <label><input type="checkbox"> Road-user types and number checked</label>
                <label><input type="checkbox"> Traffic-light color checked visually</label>
                <label><input type="checkbox"> Claimed signs / scooter checked</label>
                <p>Checkboxes are visual reminders only; they do not save or generate accuracy metrics.</p>
            </div>
        </section>
        """)
    return """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>CARLA / YOLO / VLM Visual Audit</title>
<style>
:root{font-family:Segoe UI,Arial,sans-serif;color:#253b42;background:#f4f5f0}
*{box-sizing:border-box}
body{max-width:1180px;margin:0 auto;padding:28px 22px 64px}
header.top{padding:20px 0 24px;border-bottom:1px solid #ccd4d0}
.eyebrow{letter-spacing:.14em;text-transform:uppercase;color:#46706d;font-size:12px;font-weight:700}
h1{font-size:35px;line-height:1.17;margin:10px 0}
.lead{color:#53676b;max-width:850px;line-height:1.6}
.scene{background:white;margin:28px 0;border-radius:18px;overflow:hidden;border:1px solid #d8e0dc}
.scene>header{padding:22px 26px 10px}
.scene h2{margin:5px 0 14px;font-size:22px}
.index{color:#7a8f8b;font-size:12px;letter-spacing:.13em;font-weight:bold}
.two{display:grid;grid-template-columns: 1.15fr 1fr;gap:26px;padding:0 26px 24px;align-items:start}
figure{margin:0}
figure img{max-width:100%;width:100%;height:auto;display:block;border-radius:12px;background:#dfebe7}
figcaption{font-size:12px;color:#637571;margin-top:9px}
h3{font-size:14px;margin:0 0 8px;color:#28645f}
.notes{font-size:14px;line-height:1.62}
.notes p{margin:0 0 18px}
.metric{color:#647974;font-size:12px}
details{background:#f8faf8;border-top:1px solid #e5ebe8;padding:14px 26px}
summary{cursor:pointer;font-size:13px;font-weight:600}
pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}
.review{padding:19px 26px 23px;border-top:1px solid #e5ebe8;display:grid;gap:10px;font-size:13px}
.review label{display:block}
.review p{color:#73827e;font-size:12px;margin:2px 0}
footer{color:#647772;font-size:12px;line-height:1.65;margin-top:30px}
@media(max-width:850px){.two{grid-template-columns:1fr}h1{font-size:28px}}
</style></head><body>
<header class="top"><span class="eyebrow">CARLA multimodal scene research / visual review</span>
<h1>Three-frame visual audit</h1>
<p class="lead">Compare existing RGB images with YOLO detection output and SmolVLM2 descriptions.
The text is a prediction, not ground truth. The CARLA actor metadata below is for human review only;
in particular, an actor's signal state does not prove that same lamp is visible in the RGB image.</p></header>
""" + "\n".join(cards) + """
<footer>No inference is performed here. No new risk or hallucination-rate metric is computed.
This HTML embeds local frame thumbnails and can be viewed offline. Review markings are not stored.</footer>
</body></html>"""


def save_without_overwrite(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(content)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root", type=Path, default=None,
        help="Existing vehicle-agent-rag-demo project (default: derive from script path).",
    )
    args = parser.parse_args()
    root = (args.project_root or Path(__file__).resolve().parent.parent).resolve()

    rows = [build_record(root, scene) for scene in SCENES]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out_dir = root / "evidence" / "carla_visual_audit"
    html_path = out_dir / f"visual_audit_{timestamp}.html"
    json_path = out_dir / f"visual_audit_{timestamp}.json"
    payload = {
        "report_type": "human_visual_review_not_automated_accuracy",
        "source_data_reused_without_inference": True,
        "CARLA_metadata_is_human_audit_only": True,
        "verified_visual_claims": None,
        "record_count": len(rows),
        "records": rows,
    }
    save_without_overwrite(
        json_path, json.dumps(payload, ensure_ascii=False, indent=2)
    )
    save_without_overwrite(html_path, html_page(rows))

    print("=== Phase 3.9: existing-data visual audit ===")
    for row in rows:
        print(
            f"Frame {row['frame']} | YOLO={row['yolo_detection_count']} detections "
            f"| original risk={row['yolo_baseline_risk']} "
            f"| VLM generation={row['vlm_generation_ms']:.2f} ms"
        )
    print("HTML:", html_path)
    print("JSON:", json_path)
    print("PASS: visual audit assembled; inspect the embedded RGB images manually")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, FileNotFoundError, OSError, KeyError, TypeError) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
