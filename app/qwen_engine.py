"""Lazy local Qwen inference engine.

Environment:
    QWEN_MODEL_DIR : path to downloaded Qwen2.5-0.5B-Instruct model folder

Run from the project root:
    $env:QWEN_MODEL_DIR = "C:\\models\\Qwen2.5-0.5B-Instruct"

This module does not download models and never connects to a real car.
"""

import os
from functools import lru_cache
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def resolve_model_dir() -> Path:
    """Use the environment variable, or a repository-local placeholder path."""
    value = os.getenv("QWEN_MODEL_DIR")
    model_dir = Path(value).expanduser() if value else (
        PROJECT_ROOT / "models" / "Qwen2.5-0.5B-Instruct"
    )
    if not model_dir.is_absolute():
        model_dir = PROJECT_ROOT / model_dir
    return model_dir.resolve()


@lru_cache(maxsize=1)
def load_qwen():
    """Load the local Qwen model once per Python process."""
    if not torch.cuda.is_available():
        raise RuntimeError("当前PyTorch无法使用CUDA；请确认GPU环境及PyTorch版本。")

    model_dir = resolve_model_dir()
    if not (model_dir / "config.json").is_file():
        raise FileNotFoundError(
            f"本地Qwen模型目录无效：{model_dir}。"
            "请在PowerShell中设置 QWEN_MODEL_DIR 环境变量。"
        )

    print("[Qwen] 正在加载模型...")
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_dir),
        local_files_only=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        dtype=torch.float16,
        local_files_only=True,
    ).to("cuda")

    model.eval()
    print("[Qwen] 模型加载成功")
    return tokenizer, model


@torch.inference_mode()
def generate_text(messages):
    """Generate only the new model tokens; caller handles validation."""
    tokenizer, model = load_qwen()

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = {key: value.to("cuda") for key, value in inputs.items()}

    outputs = model.generate(
        **inputs,
        max_new_tokens=128,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )

    prompt_length = inputs["input_ids"].shape[-1]
    new_tokens = outputs[0, prompt_length:]

    return tokenizer.decode(
        new_tokens,
        skip_special_tokens=True,
    ).strip()
