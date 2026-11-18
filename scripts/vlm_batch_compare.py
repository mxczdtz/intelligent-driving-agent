r"""
Phase 3.8: Multi-scenario SmolVLM2 image-only comparison.

Reuse the existing SmolVLM2-500M-Video-Instruct checkpoint and the same prompt
as scripts/vlm_smolvlm2_smoke.py. Load the model ONCE, then infer several
existing CARLA PNG/JPG images sequentially. This script DOES NOT read CARLA
metadata, YOLO reports, risk labels, or frame phases and sends none to the VLM.

Outputs:
  - each per-image report: evidence/vlm_smolvlm2/vlm_single_image_*.json
    (compatible with scripts.scene_fusion_preview.py)
  - a separate batch index: evidence/vlm_batch_compare/vlm_batch_*.json

Nothing is overwritten. Offline inference only; no weight downloads, package
installation, GPU concurrency, or real vehicle controls.

Example from project root:
  & $PY -m scripts.vlm_batch_compare --images `
    ".\evidence\carla_lead_brake\20261116_140208_590540\frame_00002358.png" `
    ".\evidence\carla_lead_brake\20261116_140208_590540\frame_00002373.png" `
    ".\evidence\carla_traffic_light\20261116_142759_601135\frame_00052924.png"
"""

from __future__ import annotations

import argparse
from datetime import datetime
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

# Importing constants does not load either model or allocate CUDA resources.
from scripts.vlm_smolvlm2_smoke import MODEL_ID, PROMPT

REPORT_VERSION = "vlm_batch_compare.v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--images", nargs="+", type=Path, required=True,
        help="One to eight local CARLA PNG/JPG paths, analyzed in the given order.",
    )
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--image-longest-edge", type=int, default=640)
    return parser.parse_args()


def configure_local_only_cache(project_root: Path) -> Path:
    """Set caches under project BEFORE importing HF or Transformers."""
    cache = project_root / "evidence" / "hf_cache"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache)
    os.environ["HF_HUB_CACHE"] = str(cache / "hub")
    os.environ["HF_XET_CACHE"] = str(cache / "xet")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    return cache


def validate_args(args: argparse.Namespace, root: Path) -> list[dict[str, Any]]:
    if not 1 <= len(args.images) <= 8:
        raise ValueError("Analyze 1 to 8 images per batch.")
    if not 16 <= args.max_new_tokens <= 256:
        raise ValueError("--max-new-tokens must be 16..256.")
    if not 256 <= args.image_longest_edge <= 1024:
        raise ValueError("--image-longest-edge must be 256..1024.")

    from PIL import Image

    inputs = []
    seen = set()
    for i, source in enumerate(args.images, start=1):
        path = source.expanduser().resolve()
        if path in seen:
            raise ValueError(f"Duplicate image path: {path}")
        seen.add(path)
        if not path.is_file():
            raise FileNotFoundError(f"Image #{i} does not exist: {path}")
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            raise ValueError(f"Image #{i} is not PNG/JPG/JPEG: {path}")
        # Only validate image data; never read capture metadata.
        with Image.open(path) as im:
            im.verify()
        with Image.open(path) as im:
            width, height = im.size
        if width <= 0 or height <= 0 or width * height > 16_000_000:
            raise ValueError(f"Image #{i} size unsupported: {width} x {height}")
        inputs.append({
            "image_path": str(path),
            "image_width": width,
            "image_height": height,
            "image_filename": path.name,
            "capture_dir": str(path.parent),
        })
    return inputs


def make_image_messages(image_path: str) -> list[dict[str, Any]]:
    """Precisely the same image-only prompt used for the earlier single frame."""
    return [{
        "role": "user",
        "content": [
            {"type": "image", "path": image_path},
            {"type": "text", "text": PROMPT},
        ],
    }]


def output_filename() -> str:
    return f"vlm_single_image_{datetime.now():%Y%m%d_%H%M%S_%f}.json"


def write_new_json(folder: Path, name: str, data: dict[str, Any]) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    destination = folder / name
    # 'x' prevents accidental overwrites and keeps the original 3488 report.
    with destination.open("x", encoding="utf-8") as fp:
        json.dump(data, fp, ensure_ascii=False, indent=2)
    return destination


def per_frame_report(
    original_input: dict[str, Any],
    *,
    text: str,
    input_tokens: int,
    new_tokens: int,
    preprocess_ms: float,
    generation_ms: float,
    model_load_ms: float,
    peak_cuda_allocated_mib: float | None,
    device: str,
    torch_version: str,
    model_dir: Path,
    image_longest_edge: int,
) -> dict[str, Any]:
    """Match the prior vlm_single_image_*.json schema for downstream reuse."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("VLM generated an empty description.")
    if input_tokens <= 0 or new_tokens <= 0:
        raise ValueError("VLM generated no text tokens.")
    return {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "mode": "image_only_vlm_smoke_test",
        "batch_report_version": REPORT_VERSION,
        "model_id": MODEL_ID,
        "model_path": str(model_dir),
        "image_path": original_input["image_path"],
        "image_width": original_input["image_width"],
        "image_height": original_input["image_height"],
        "device": device,
        "torch_version": torch_version,
        "processor_longest_edge": image_longest_edge,
        "dtype": "float16" if device == "cuda" else "float32",
        "attention": "sdpa",
        "prompt": PROMPT,
        "output_text": text,
        "input_tokens": input_tokens,
        "new_tokens": new_tokens,
        "model_load_ms": round(model_load_ms, 2),
        "preprocess_ms": round(preprocess_ms, 2),
        "generation_ms": round(generation_ms, 2),
        "peak_cuda_memory_allocated_mib": peak_cuda_allocated_mib,
        "limitations": [
            "Each image independently processed; no multi-frame tracking is performed.",
            "No simulator phase, traffic-light actor state, CARLA truth or YOLO labels were supplied.",
            "Single-frame text cannot establish temporal lead-vehicle braking or TTC.",
            "VLM descriptions may be inaccurate, even if generation succeeds.",
            "GPU memory metric represents PyTorch peak allocated memory, not total GPU use.",
            "Simulation research only; no real vehicle control.",
        ],
    }


def infer_many(inputs: list[dict[str, Any]], root: Path,
               device: str, max_new_tokens: int,
               longest_edge: int) -> dict[str, Any]:
    """Load the checkpoint and processor once; process each image in series."""
    model_dir = root / "models" / "SmolVLM2-500M-Video-Instruct"
    missing = [name for name in ("config.json", "model.safetensors")
               if not (model_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing local SmolVLM2 model files: {missing}. "
            "Use the already downloaded checkpoint; this batch tool is offline."
        )
    if importlib.util.find_spec("num2words") is None:
        raise RuntimeError("num2words is missing in the existing environment.")

    configure_local_only_cache(root)
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable, close CARLA or use --device cpu.")
        free, total = torch.cuda.mem_get_info()
        print(f"Free GPU memory: {free / 1024**3:.2f} / {total / 1024**3:.2f} GiB")
        if free < 2 * 1024 ** 3:
            raise RuntimeError("Less than 2 GiB GPU memory free; close CARLA.")

    dtype = torch.float16 if device == "cuda" else torch.float32
    t0 = time.perf_counter()
    processor = AutoProcessor.from_pretrained(str(model_dir), local_files_only=True)
    processor.image_processor.do_resize = True
    processor.image_processor.size = {"longest_edge": longest_edge}
    model = AutoModelForImageTextToText.from_pretrained(
        str(model_dir), local_files_only=True,
        dtype=dtype, attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    if device == "cuda":
        torch.cuda.synchronize()
    load_ms = round((time.perf_counter() - t0) * 1000, 2)
    print(f"Checkpoint loaded once: {load_ms:.2f} ms")

    items: list[dict[str, Any]] = []
    reports_dir = root / "evidence" / "vlm_smolvlm2"
    for i, item in enumerate(inputs, start=1):
        print(f"\n[{i}/{len(inputs)}] Image: {item['image_path']}")
        try:
            messages = make_image_messages(item["image_path"])
            t1 = time.perf_counter()
            tokens = processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt",
            ).to(device, dtype=dtype)
            if device == "cuda":
                torch.cuda.synchronize()
            preprocess_ms = round((time.perf_counter() - t1) * 1000, 2)

            if device == "cuda":
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
            t2 = time.perf_counter()
            with torch.inference_mode():
                output = model.generate(
                    **tokens, do_sample=False,
                    max_new_tokens=max_new_tokens,
                )
            if device == "cuda":
                torch.cuda.synchronize()
            generation_ms = round((time.perf_counter() - t2) * 1000, 2)
            input_count = int(tokens["input_ids"].shape[-1])
            generated = output[:, input_count:]
            text = processor.batch_decode(
                generated, skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0].strip()

            peak = (
                round(torch.cuda.max_memory_allocated() / 1024**2, 1)
                if device == "cuda" else None
            )
            report = per_frame_report(
                item, text=text, input_tokens=input_count,
                new_tokens=int(generated.shape[-1]),
                preprocess_ms=preprocess_ms,
                generation_ms=generation_ms,
                model_load_ms=load_ms if i == 1 else 0,
                peak_cuda_allocated_mib=peak,
                device=device, torch_version=torch.__version__,
                model_dir=model_dir, image_longest_edge=longest_edge,
            )
            saved = write_new_json(reports_dir, output_filename(), report)
            print("VLM text (unverified):", text)
            print(f"Generation: {generation_ms:.2f} ms | Report: {saved}")
            items.append({
                "image_path": item["image_path"],
                "status": "pass",
                "report_path": str(saved),
                "generation_ms": generation_ms,
                "new_tokens": report["new_tokens"],
            })
            del tokens, output, generated
        except Exception as exc:
            print(f"ERROR for {item['image_filename']}: {type(exc).__name__}: {exc}")
            items.append({
                "image_path": item["image_path"],
                "status": "error",
                "error": f"{type(exc).__name__}: {str(exc)[:400]}",
            })
            if device == "cuda" and isinstance(exc, torch.cuda.OutOfMemoryError):
                torch.cuda.empty_cache()
    return {
        "schema_version": REPORT_VERSION,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "model_id": MODEL_ID,
        "model_dir": str(model_dir),
        "prompt": PROMPT,
        "model_load_ms": load_ms,
        "image_count": len(inputs),
        "passed": sum(x["status"] == "pass" for x in items),
        "failed": sum(x["status"] == "error" for x in items),
        "frames": items,
        "no_CARLA_metadata_read": True,
        "no_YOLO_or_risk_labels_read": True,
        "model_loaded_once": True,
    }


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parent.parent
    inputs = validate_args(args, root)
    print(f"Validated {len(inputs)} local image(s).")
    manifest = infer_many(
        inputs, root, args.device, args.max_new_tokens,
        args.image_longest_edge,
    )
    path = write_new_json(
        root / "evidence" / "vlm_batch_compare",
        f"vlm_batch_{datetime.now():%Y%m%d_%H%M%S_%f}.json",
        manifest,
    )
    print("\n=== Phase 3.8 batch summary ===")
    print(f"Image-only VLM: {manifest['passed']}/{manifest['image_count']} passed")
    print("Batch index:", path)
    if manifest["failed"] == 0:
        print("PASS: independent CARLA images -> VLM reports (single model load)")
        return 0
    print("PARTIAL_FAILURE: some images did not yield usable reports.")
    return 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, ImportError) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
