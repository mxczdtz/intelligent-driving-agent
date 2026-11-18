"""
Phase 3 / SmolVLM2 single-image CARLA smoke test (Windows-friendly).

Small story: read one existing CARLA RGB frame; ask a small VLM for
observable road-scene facts; save the raw model response in evidence/.
No simulator truth, YOLO labels, or vehicle-control actions are supplied.

Modes:
  --download  Fetch only Transformers files and the official safetensors
              checkpoint to models/ on the current project drive.
  --image     Run ONE local CARLA frame fully offline, in float16 + SDPA.

Safety/resource constraints:
  * No new environment; no pip installs; no updating torch.
  * No CARLA needed. Close CARLA and heavy GPU apps before --image.
  * Model download is explicit; inference is strictly local/offline.
  * All cache/output directories are under the existing project root.
  * This is an exploratory visual-language output, not a safety judgment.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

MODEL_ID = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
PROMPT = (
    "Describe this driving camera image using only what is visible. "
    "Mention visible road users and their approximate positions, "
    "traffic lights and their colors ONLY if clearly recognizable, "
    "and possible road hazards. State uncertainty explicitly. "
    "Do not invent distances, relative speeds, or future motion. "
    "Write 2 to 4 concise sentences in English."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download the official SmolVLM2-500M model OR analyze one CARLA RGB image."
    )
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--download", action="store_true",
                        help="Explicitly download required Hugging Face files.")
    choice.add_argument("--image", type=Path,
                        help="A CARLA PNG/JPEG image already captured locally.")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--max-new-tokens", type=int, default=96,
                        help="Maximum new response tokens (default: 96).")
    parser.add_argument("--image-longest-edge", type=int, default=640,
                        help="Processor longest-edge limit; initial default 640.")
    return parser.parse_args()


def project_paths() -> tuple[Path, Path, Path]:
    root = Path(__file__).resolve().parent.parent
    model_dir = root / "models" / "SmolVLM2-500M-Video-Instruct"
    cache_dir = root / "evidence" / "hf_cache"
    return root, model_dir, cache_dir


def confine_hf_cache(cache_dir: Path) -> None:
    """Set cache paths BEFORE importing huggingface_hub or transformers."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    # Deliberately override an old HF_HOME that might point to C: or D:.
    os.environ["HF_HOME"] = str(cache_dir)
    os.environ["HF_HUB_CACHE"] = str(cache_dir / "hub")
    os.environ["HF_XET_CACHE"] = str(cache_dir / "xet")


def download_model(model_dir: Path) -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is not installed. Check the existing llm-gpu environment "
            "before installing any package."
        ) from exc

    print("Official model:", MODEL_ID)
    print("Model destination:", model_dir)
    print("Download mode: model weights + small config/tokenizer files, excluding ONNX.")
    print("Note: the official model.safetensors is approximately 2.03 GB.")
    model_dir.mkdir(parents=True, exist_ok=True)

    # Do not fetch the repo's large onnx/ directory (repo total ~7.9GB).
    snapshot_download(
        repo_id=MODEL_ID,
        revision="main",
        local_dir=str(model_dir),
        allow_patterns=[
            "*.safetensors",
            "*.json",
            "*.txt",
            "*.model",
            "*.jinja",
        ],
        ignore_patterns=[
            "onnx/*",
            "*.onnx",
            "*.onnx_data",
            "*.bin",
            "*.pt",
        ],
        max_workers=2,
    )
    required = (model_dir / "config.json", model_dir / "model.safetensors")
    missing = [p.name for p in required if not p.is_file()]
    if missing:
        raise RuntimeError("Download is incomplete, missing: " + ", ".join(missing))
    print("PASS: Official model checkpoint and config found locally.")
    print("Model path:", model_dir)


def cuda_synchronize(torch: Any, device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def infer(
    image_path: Path, model_dir: Path, root: Path,
    device: str, max_new_tokens: int, image_longest_edge: int
) -> None:
    if not 16 <= max_new_tokens <= 256:
        raise ValueError("--max-new-tokens must be between 16 and 256.")
    if not 256 <= image_longest_edge <= 1024:
        raise ValueError("--image-longest-edge must be between 256 and 1024.")

    image_path = image_path.expanduser().resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if image_path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
        raise ValueError("Only local PNG/JPG/JPEG images are supported.")

    required = ("model.safetensors", "config.json")
    missing = [name for name in required if not (model_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            "Local VLM files are missing: " + ", ".join(missing)
            + ". Run with --download first."
        )
    if importlib.util.find_spec("num2words") is None:
        raise RuntimeError(
            "SmolVLM processor needs the small package num2words. "
            "Check it with `& $PY -m pip show num2words`, then (only if absent) "
            "install it into the existing environment; do not update torch."
        )

    # Image validation before loading weights.
    from PIL import Image
    with Image.open(image_path) as img:
        img.verify()
    with Image.open(image_path) as img:
        width, height = img.size

    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable. Close CARLA or use --device cpu.")
        free, total = torch.cuda.mem_get_info()
        print(
            f"CUDA free: {free / (1024 ** 3):.2f} GiB / "
            f"{total / (1024 ** 3):.2f} GiB"
        )
        if free < 2.0 * 1024 ** 3:
            raise RuntimeError(
                "Less than 2 GiB VRAM currently free. "
                "Close CARLA and other GPU-heavy programs before inference."
            )

    # No network in inference mode; only the explicit --download mode may fetch.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    print("Image:", image_path)
    print("Image size:", f"{width} x {height}")
    print("Loading local SmolVLM2 checkpoint in float16 with SDPA...")
    t0 = time.perf_counter()
    processor = AutoProcessor.from_pretrained(
        str(model_dir), local_files_only=True
    )
    # Important: avoid SmolVLM's potentially large default image upsampling.
    processor.image_processor.do_resize = True
    processor.image_processor.size = {"longest_edge": image_longest_edge}

    model = AutoModelForImageTextToText.from_pretrained(
        str(model_dir),
        local_files_only=True,
        dtype=torch.float16 if device == "cuda" else torch.float32,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    cuda_synchronize(torch, device)
    load_ms = round((time.perf_counter() - t0) * 1000, 2)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "path": str(image_path)},
                {"type": "text", "text": PROMPT},
            ],
        }
    ]
    t1 = time.perf_counter()
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(device, dtype=torch.float16 if device == "cuda" else torch.float32)
    cuda_synchronize(torch, device)
    preprocess_ms = round((time.perf_counter() - t1) * 1000, 2)

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    t2 = time.perf_counter()
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
        )
    cuda_synchronize(torch, device)
    generate_ms = round((time.perf_counter() - t2) * 1000, 2)

    input_tokens = int(inputs["input_ids"].shape[-1])
    generated = output_ids[:, input_tokens:]
    text = processor.batch_decode(
        generated,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()

    peak_allocated_mib = (
        round(torch.cuda.max_memory_allocated() / (1024 ** 2), 1)
        if device == "cuda" else None
    )
    report: dict[str, Any] = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "mode": "image_only_vlm_smoke_test",
        "model_id": MODEL_ID,
        "model_path": str(model_dir),
        "image_path": str(image_path),
        "image_width": width,
        "image_height": height,
        "device": device,
        "torch_version": torch.__version__,
        "processor_longest_edge": image_longest_edge,
        "dtype": "float16" if device == "cuda" else "float32",
        "attention": "sdpa",
        "prompt": PROMPT,
        "output_text": text,
        "input_tokens": input_tokens,
        "new_tokens": int(generated.shape[-1]),
        "model_load_ms": load_ms,
        "preprocess_ms": preprocess_ms,
        "generation_ms": generate_ms,
        "peak_cuda_memory_allocated_mib": peak_allocated_mib,
        "limitations": [
            "Image-only VLM; no simulator truth or YOLO detection is used.",
            "Generated content may be inaccurate or hallucinated.",
            "No independent risk ground truth or calibrated risk score.",
            "Single-run timing is not an official performance benchmark.",
            "Simulation research only; no real vehicle control.",
        ],
    }

    out_dir = root / "evidence" / "vlm_smolvlm2"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"vlm_single_image_{datetime.now():%Y%m%d_%H%M%S_%f}.json"
    out_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("\n=== VLM image-only observations (NOT ground truth) ===")
    print(text if text else "(empty response)")
    print(f"Model load: {load_ms:.2f} ms")
    print(f"Preprocess: {preprocess_ms:.2f} ms")
    print(f"Generation: {generate_ms:.2f} ms")
    print(f"New tokens: {report['new_tokens']}")
    print(f"Peak CUDA allocated during generation: {peak_allocated_mib} MiB")
    print("Report:", out_path)
    print("PASS: Local CARLA image -> VLM text (single frame)")


def main() -> int:
    args = parse_args()
    root, model_dir, cache_dir = project_paths()
    confine_hf_cache(cache_dir)
    print("Project root:", root)
    print("HF cache:", cache_dir)
    if args.download:
        download_model(model_dir)
    else:
        infer(
            args.image,
            model_dir,
            root,
            args.device,
            args.max_new_tokens,
            args.image_longest_edge,
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, ImportError, OSError, ValueError) as exc:
        print(f"\nERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
