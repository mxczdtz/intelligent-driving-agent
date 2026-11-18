"""Run existing YOLO and risk modules on synchronized CARLA captures.

From the project root:
    python -m scripts.carla_analyze_capture --capture-dir evidence/carla_capture_demo/RUN_ID

Offline simulation only; this script does not connect to or control vehicles.
"""
import argparse
import json
import math
import os
from datetime import datetime
from pathlib import Path
from time import perf_counter

from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", required=True, type=Path)
    parser.add_argument("--device", default="0", help="0 for first CUDA GPU; cpu for CPU")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    capture_dir = args.capture_dir.resolve()
    metadata = json.loads((capture_dir / "metadata.json").read_text(encoding="utf-8"))
    frames = metadata["frames"]
    if not frames or metadata["matched_frame_count"] != len(frames):
        raise ValueError("Invalid matched frame count")

    # Check every image/state pair before model inference.
    validated = []
    seen = set()
    for record in frames:
        frame = record["image_frame"]
        if frame != record["world_frame"] or frame in seen:
            raise ValueError(f"Invalid or duplicated frame: {frame}")
        seen.add(frame)
        image_time = float(record["image_timestamp_s"])
        world_time = float(record["world_timestamp_s"])
        if not all(math.isfinite(t) for t in (image_time, world_time)):
            raise ValueError(f"Invalid timestamp: {frame}")
        if abs(image_time - world_time) > 0.001:
            raise ValueError(f"Timestamp mismatch: {frame}")
        speed = float(record["ego_speed_kmh"])
        if not math.isfinite(speed) or not 0 <= speed <= 200:
            raise ValueError(f"Invalid ego speed: {frame}")
        image_path = (capture_dir / record["image_file"]).resolve()
        if image_path.parent != capture_dir or image_path.suffix.lower() != ".png":
            raise ValueError(f"Unexpected image path: {image_path}")
        with Image.open(image_path) as img:
            if img.format != "PNG":
                raise ValueError(f"Expected PNG: {image_path}")
            if img.size != (record["image_width"], record["image_height"]):
                raise ValueError(f"Image dimensions mismatch: {frame}")
            img.verify()
        validated.append((record, image_path, speed))

    # Use the already installed model. Never trigger a model download.
    model_path = root / "models" / "yolo11n.pt"
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    os.environ["SCENE_YOLO_MODEL"] = str(model_path)
    os.environ["SCENE_YOLO_DEVICE"] = args.device

    from app.scene_detector import detect_objects
    from app.scene_risk import SceneInput, SceneResponse, assess_scene

    results = []
    print("Map:", metadata["map"], flush=True)
    print("Validated image/state pairs:", len(validated), flush=True)
    print("Device:", args.device, flush=True)

    for record, image_path, speed in sorted(validated, key=lambda item: item[0]["image_frame"]):
        start = perf_counter()
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        width, height = image.size
        detections = detect_objects(image)
        scene = SceneInput(
            image_width=width,
            image_height=height,
            ego_speed_kmh=speed,
            detections=detections,
        )
        risk = assess_scene(scene)
        response = SceneResponse(
            image_width=width,
            image_height=height,
            ego_speed_kmh=speed,
            detections=detections,
            risk=risk,
            inference_ms=round((perf_counter() - start) * 1000, 2),
        )
        results.append({
            "world_frame": record["world_frame"],
            "image_frame": record["image_frame"],
            "image_file": record["image_file"],
            "world_timestamp_s": record["world_timestamp_s"],
            "image_timestamp_s": record["image_timestamp_s"],
            **response.model_dump(mode="json"),
        })
        print(
            f"Frame {record['image_frame']} | speed={speed:.3f} km/h "
            f"| detections={len(detections)} | risk={risk.risk_level.value} "
            f"| analysis={response.inference_ms:.2f} ms",
            flush=True,
        )

    report = {
        "map": metadata["map"],
        "scenario": metadata["scenario"],
        "source_metadata": "metadata.json",
        "model": model_path.name,
        "device": args.device,
        "method": "Offline YOLO inference + existing 2D heuristic risk engine",
        "timing_note": "Per-image wall-clock analysis; first frame may include model loading.",
        "limitations": [
            "Not ground-truth risk labels or a risk accuracy evaluation.",
            "No VLM/LLM inference, real distance estimation or real vehicle control.",
        ],
        "frame_count": len(results),
        "frames": results,
    }
    output = capture_dir / ("yolo_risk_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".json")
    with output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False, allow_nan=False)
    print("Report:", output, flush=True)
    print(f"PASS: CARLA -> YOLO -> risk ({len(results)} frames)", flush=True)


if __name__ == "__main__":
    main()
