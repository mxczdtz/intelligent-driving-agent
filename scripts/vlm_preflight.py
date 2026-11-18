"""
Phase 3 VLM preflight for vehicle-agent-rag-demo.

Purpose: read-only checks of Python packages, CUDA/GPU memory, local
model paths, and disk space before choosing a vision-language model.
This script does not install packages, download weights, or modify
the existing runtime. It writes only its own JSON report under evidence/.
"""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

PACKAGES = {
    "torch": "torch",
    "torchvision": "torchvision",
    "transformers": "transformers",
    "accelerate": "accelerate",
    "huggingface_hub": "huggingface-hub",
    "safetensors": "safetensors",
    "Pillow": "Pillow",
    "bitsandbytes": "bitsandbytes",
    "qwen_vl_utils": "qwen-vl-utils",
    "sentencepiece": "sentencepiece",
    "protobuf": "protobuf",
    "flash_attn": "flash-attn",
    "numpy": "numpy",
    "ultralytics": "ultralytics",
    "einops": "einops",
}


def gib(value: int | float) -> float:
    return round(float(value) / (1024 ** 3), 2)


def package_versions() -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for key, dist_name in PACKAGES.items():
        try:
            result[key] = importlib.metadata.version(dist_name)
        except importlib.metadata.PackageNotFoundError:
            result[key] = None
    return result


def check_cuda() -> dict[str, Any]:
    result: dict[str, Any] = {}
    try:
        import torch

        result["torch_version"] = torch.__version__
        result["torch_cuda_runtime"] = torch.version.cuda
        result["cuda_available"] = bool(torch.cuda.is_available())
        if result["cuda_available"]:
            devices: list[dict[str, Any]] = []
            for index in range(torch.cuda.device_count()):
                properties = torch.cuda.get_device_properties(index)
                device: dict[str, Any] = {
                    "index": index,
                    "name": properties.name,
                    "total_vram_gib": gib(properties.total_memory),
                    "compute_capability": list(torch.cuda.get_device_capability(index)),
                }
                try:
                    with torch.cuda.device(index):
                        free_bytes, total_bytes = torch.cuda.mem_get_info()
                    device["free_vram_gib"] = gib(free_bytes)
                    device["reported_total_vram_gib"] = gib(total_bytes)
                except Exception as exc:
                    device["memory_query_error"] = str(exc)
                devices.append(device)
            result["devices"] = devices
    except Exception as exc:
        result["torch_probe_error"] = f"{type(exc).__name__}: {exc}"
    return result


def nvidia_smi() -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=name,driver_version,memory.total,memory.free",
        "--format=csv,noheader,nounits",
    ]
    try:
        proc = subprocess.run(
            command, capture_output=True, text=True, timeout=12, check=False
        )
        if proc.returncode != 0:
            return {"error": (proc.stderr or proc.stdout).strip()[:400]}
        return {"gpu_rows": [line.strip() for line in proc.stdout.splitlines() if line.strip()],
                "columns": ["name", "driver_version", "memory_total_mib", "memory_free_mib"]}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def drive_usage() -> dict[str, Any]:
    result: dict[str, Any] = {}
    for letter in ("C", "D", "E"):
        drive = Path(f"{letter}:\\")
        try:
            total, used, free = shutil.disk_usage(drive)
            result[letter] = {
                "total_gib": gib(total),
                "used_gib": gib(used),
                "free_gib": gib(free),
            }
        except OSError as exc:
            result[letter] = {"error": f"{type(exc).__name__}: {exc}"}
    return result


def existing_models(project_root: Path) -> dict[str, Any]:
    models_dir = project_root / "models"
    if not models_dir.is_dir():
        return {"models_dir_exists": False}
    entries = sorted(
        (
            {
                "name": p.name,
                "type": "directory" if p.is_dir() else "file",
            }
            for p in models_dir.iterdir()
        ),
        key=lambda item: item["name"].casefold(),
    )
    return {
        "models_dir_exists": True,
        "model_entries": entries[:80],
        "entries_truncated": len(entries) > 80,
    }


def main() -> int:
    project_root = Path(__file__).resolve().parent.parent
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = project_root / "evidence" / "vlm_preflight"
    report_dir.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "project_root": str(project_root),
        "python": {
            "executable": sys.executable,
            "version": sys.version.split()[0],
            "platform": platform.platform(),
        },
        "package_versions": package_versions(),
        "cuda": check_cuda(),
        "nvidia_smi": nvidia_smi(),
        "drives": drive_usage(),
        "existing_models": existing_models(project_root),
        "cache_environment": {
            key: os.environ.get(key)
            for key in ("HF_HOME", "HF_HUB_CACHE", "TRANSFORMERS_CACHE")
        },
        "notes": [
            "Read-only environment probes; no package installation or model downloads.",
            "No CARLA images or scene metadata are read or changed.",
            "GPU memory is a momentary measurement, not a VLM inference benchmark.",
            "Do not post a full environment dump; use this report only for VLM planning.",
        ],
    }

    path = report_dir / f"vlm_preflight_{stamp}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=== Phase 3: VLM preflight ===")
    print("Python:", report["python"]["executable"], report["python"]["version"])
    print("Torch:", report["package_versions"]["torch"])
    print("Transformers:", report["package_versions"]["transformers"])
    print("Accelerate:", report["package_versions"]["accelerate"])
    print("Bitsandbytes:", report["package_versions"]["bitsandbytes"])
    print("CUDA available:", report["cuda"].get("cuda_available", "unknown"))
    for gpu in report["cuda"].get("devices", []):
        print(
            f"GPU {gpu['index']}: {gpu['name']} "
            f"total={gpu['total_vram_gib']} GiB "
            f"free={gpu.get('free_vram_gib', 'unknown')} GiB"
        )
    for name, usage in report["drives"].items():
        print(f"Drive {name}: free={usage.get('free_gib', 'unavailable')} GiB")
    model_entries = report["existing_models"].get("model_entries", [])
    print("Local model entries:", ", ".join(e["name"] for e in model_entries) or "(none)")
    print("Report:", path)
    print("PASS: VLM preflight report written (no installations/downloads)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
