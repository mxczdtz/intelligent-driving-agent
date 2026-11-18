"""
Phase 3.4.1: Local Qwen structured scene-risk research prototype.

INPUT: Phase 3.3 scene_fusion.py fusion_frame_*.json
        + optional ego speed, supplied independently from logged ego telemetry.
OUTPUT: A new evidence/scene_llm_risk/*.json with the model candidate,
        strict schema validation, and deterministic safety downgrades.

Key safeguards:
- CARLA actor ground truth (audit_only) is never shown to the language model.
- Original YOLO risk is kept only for post-run comparison, NOT in the prompt.
- No network, no model download, no new environment or dependency installations.
- No new label is treated as ground truth; no actual vehicle control.
- Invalid / unparseable model output degrades to risk_level="unknown".
- A single frame cannot support high risk via unsupported TTC/distance claims.
- A low-risk claim with pedestrian(s) and no lane/TTC evidence degrades to unknown.

From the project root:
  python -m app.scene_llm_risk --fusion-report "evidence/.../fusion_frame_...json" \
    --model-dir "PATH_TO_LOCAL_QWEN_MODEL" \
    --ego-speed-kmh 10.795

Use only for simulation experiments and human review.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
import os
from pathlib import Path
import re
import sys
import time
from typing import Any

SCHEMA_VERSION = "scene_llm_risk.v1"
VALID_LEVELS = {"unknown", "low", "medium", "high"}
DECISION_FIELDS = {
    "risk_level", "key_factors", "evidence_refs",
    "driving_suggestion", "uncertainty",
}
CONSERVATIVE_SUGGESTION = (
    "仅供仿真研究和人工复核；保持谨慎观察，必要时采取保守措施，"
    "不得据此直接控制真实车辆。"
)
SYSTEM_PROMPT = """你是智能驾驶仿真研究中的证据整理助手，不是自动驾驶控制器。
请只根据给定的 YOLO 检测、未经核实的 VLM 声明和可选自车速度输出 JSON。
规则：
- 不要臆造交通标志、灯色、行驶方向、车道位置、距离、TTC 或碰撞风险。
- VLM 声明是未经验证的文字，并非场景真值；YOLO 检测同样不是场景真值。
- 缺少可靠的相对位置、距离和时序信息时，允许且优先使用 risk_level="unknown"。
- 区分「YOLO/VLM 报告了什么」和「危险风险是否能被证实」：前者是带来源的模型观察，后者可能仍为 unknown。
- 当 YOLO_predictions 中有目标时，在 key_factors 中优先写一条可核查的 YOLO 检测观察（例如「YOLO预测有两个人」），使用对应的 yolo:* ID 放入 evidence_refs；这不代表行人位于自车道，也不表示已确认危险。
- risk_level="unknown" 也可以且通常应该引用与 key_factors 相符的已有视觉证据；若证据完全不可用，evidence_refs 才可为空。不要为了凑引用而编造或强行引用无关 ID。
- VLM_claims_unverified 只能作为未经验证的描述；交通灯颜色和单帧车辆运动声明若缺乏证实，应写入 uncertainty 而非当成风险事实。
- key_factors 尽量陈述具体观察；uncertainty 陈述距离、TTC、车道归属、时序等缺失信息，避免重复空泛的安全免责声明。
- 只引用 available_evidence_refs 中的证据 ID；没有足够证据时 evidence_refs 可为空。
- 只输出一个完整 JSON 对象，不要 Markdown 代码围栏、说明或额外字段。
- key_factors、uncertainty 各写 1～3 条短句，绝对不得超过 5 条。
- evidence_refs 最多 4 条；driving_suggestion 仅用于仿真研究和人工复核。

固定格式示例（仅示范结构，不代表本帧事实）：
{"risk_level":"unknown","key_factors":["证据不足"],"evidence_refs":[],"driving_suggestion":"仅供研究人员复核，谨慎观察","uncertainty":["无法从单帧确定目标位置和运动趋势"]}
risk_level 只能是 unknown、low、medium、high。
"""


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as fp:
        val = json.load(fp)
    if not isinstance(val, dict):
        raise ValueError("Expected a JSON object.")
    return val


def short_text(value: Any, max_len: int = 180) -> str:
    if value is None:
        return ""
    return str(value)[:max_len]


def build_model_input(fusion: dict[str, Any], ego_speed_kmh: float | None) -> tuple[dict[str, Any], set[str]]:
    """Whitelist visual evidence fields: never read audit_only or original baseline."""
    if fusion.get("schema_version") != "scene_fusion_evidence.v1":
        raise ValueError("Expected Phase 3.3 scene_fusion_evidence.v1 report.")

    world_frame = fusion.get("world_frame")
    if not isinstance(world_frame, int) or isinstance(world_frame, bool) or world_frame < 0:
        raise ValueError("Missing or invalid world_frame.")

    vis = fusion.get("yolo_visual_evidence", {})
    objects = vis.get("detections")
    if not isinstance(objects, list) or len(objects) > 100:
        raise ValueError("Invalid YOLO detections list.")

    evidence_ids: set[str] = set()
    detections: list[dict[str, Any]] = []
    for i, item in enumerate(objects):
        if not isinstance(item, dict) or not isinstance(item.get("class"), str):
            raise ValueError(f"Invalid detection {i}.")
        confidence = item.get("confidence")
        if confidence is not None:
            try:
                confidence = float(confidence)
            except (ValueError, TypeError) as exc:
                raise ValueError("Invalid YOLO confidence.") from exc
            if not math.isfinite(confidence) or not 0 <= confidence <= 1:
                raise ValueError("YOLO confidence not in [0,1].")
        bbox = item.get("bbox_xyxy_from_alignment")
        if bbox is not None:
            if not (isinstance(bbox, list) and len(bbox) == 4):
                raise ValueError("Invalid XYXY bbox.")
            if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in bbox):
                raise ValueError("Invalid XYXY numeric value.")
        det_id = f"yolo:{i}"
        evidence_ids.add(det_id)
        detections.append({
            "id": det_id,
            "class": short_text(item["class"], 50),
            "confidence": round(confidence, 4) if confidence is not None else None,
            "bbox_xyxy_from_alignment": bbox,
            "source": "YOLO_model_prediction_not_truth",
        })

    vlm = fusion.get("vlm_text_evidence", {})
    claims = vlm.get("claims_unverified")
    if not isinstance(claims, list) or len(claims) > 100:
        raise ValueError("Missing/invalid VLM claims.")
    sanitized_claims: list[dict[str, Any]] = []
    for i, claim in enumerate(claims):
        if not isinstance(claim, dict):
            raise ValueError(f"Invalid VLM claim {i}.")
        value = claim.get("value")
        if not isinstance(value, (str, int, bool, list)):
            continue
        if isinstance(value, list):
            value = [short_text(x, 30) for x in value[:5]]
        if isinstance(value, str):
            value = short_text(value, 50)
        claim_id = f"vlm:{i}"
        evidence_ids.add(claim_id)
        sanitized_claims.append({
            "id": claim_id,
            "type": short_text(claim.get("type"), 50),
            "value": value,
            "status": "VLM_claim_unverified",
        })

    checks: list[dict[str, Any]] = []
    for check in fusion.get("cross_model_checks", [])[:20]:
        if not isinstance(check, dict):
            continue
        checks.append({
            "comparison": short_text(check.get("comparison"), 50),
            "status": short_text(check.get("status"), 100),
        })

    uncertainties = [
        short_text(item, 150)
        for item in fusion.get("uncertainties", [])[:12]
        if isinstance(item, str)
    ]
    telemetry = None
    if ego_speed_kmh is not None:
        if not isinstance(ego_speed_kmh, (int, float)) or not math.isfinite(ego_speed_kmh):
            raise ValueError("ego_speed_kmh must be finite.")
        if not 0 <= ego_speed_kmh <= 300:
            raise ValueError("ego_speed_kmh out of plausible bounds.")
        telemetry = {"id": "ego:speed", "ego_speed_kmh": round(ego_speed_kmh, 3),
                     "source": "user_supplied_from_previously_logged_ego_telemetry"}
        evidence_ids.add("ego:speed")

    payload = {
        "world_frame": world_frame,
        "YOLO_predictions": detections,
        "VLM_claims_unverified": sanitized_claims,
        "cross_checks": checks,
        "known_uncertainties": uncertainties,
        "ego_telemetry": telemetry,
        "available_evidence_refs": sorted(evidence_ids),
        "has_calibrated_distance_or_TTC": False,
        "input_scope": "single_RGB_frame_visual_evidence_only",
    }
    return payload, evidence_ids


def prompt_for(payload: dict[str, Any]) -> list[dict[str, str]]:
    # Raw VLM prose/quoted_context and CARLA actor metadata never enter this prompt.
    user = (
        "依据下方 JSON 证据，给出谨慎的结构化风险推理。"
        "若缺乏明确的空间关联或时序信息，应承认不确定性。"
        "只能引用 available_evidence_refs 里的 ID。\n"
        "若 YOLO 检测目标存在，请用一条具体的 YOLO 观察作为 key_factors，"
        "并引用对应 yolo:* ID；风险等级仍可为 unknown。"
        "切勿把未经核实的 VLM 颜色/运动判断当成事实。\n"
        + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    )
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user}]


def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key: " + key)
        result[key] = value
    return result


def parse_model_json(raw: str) -> dict[str, Any]:
    stripped = raw.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*\n?(.*?)\n?```", stripped,
                          flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        stripped = fenced.group(1).strip()
    result = json.loads(stripped, object_pairs_hook=reject_duplicate_keys)
    if not isinstance(result, dict):
        raise ValueError("Model output must be a JSON object.")
    return result


def validate_decision(decision: dict[str, Any], evidence_ids: set[str]) -> dict[str, Any]:
    """Strict stdlib validator matching bundled JSON Schema (no extra dependency)."""
    if set(decision) != DECISION_FIELDS:
        raise ValueError(
            f"Decision keys must be exactly {sorted(DECISION_FIELDS)}; "
            f"received {sorted(decision)}"
        )
    if decision["risk_level"] not in VALID_LEVELS:
        raise ValueError("Invalid risk_level.")

    for key, max_items, max_len in (
        ("key_factors", 5, 180),
        ("uncertainty", 5, 180),
        ("evidence_refs", 8, 50),
    ):
        value = decision[key]
        if not isinstance(value, list) or len(value) > max_items:
            raise ValueError(f"{key} must be a list of at most {max_items}.")
        if len(value) == 0 and key != "evidence_refs":
            raise ValueError(f"{key} must be nonempty.")
        if not all(isinstance(v, str) and 1 <= len(v.strip()) <= max_len
                   for v in value):
            raise ValueError(f"{key} contains invalid strings.")
        if len(set(value)) != len(value):
            raise ValueError(f"{key} contains repeated entries.")

    advice = decision["driving_suggestion"]
    if not isinstance(advice, str) or not 1 <= len(advice.strip()) <= 180:
        raise ValueError("Invalid driving_suggestion.")
    if any(ref not in evidence_ids for ref in decision["evidence_refs"]):
        raise ValueError("Model cited an unknown evidence ID.")
    if decision["risk_level"] != "unknown" and not decision["evidence_refs"]:
        raise ValueError("Non-unknown risk must cite at least one evidence ID.")
    return decision


def fallback_unknown(reason: str) -> dict[str, Any]:
    return {
        "risk_level": "unknown",
        "key_factors": ["当前视觉证据不足以形成可信的风险等级判断"],
        "evidence_refs": [],
        "driving_suggestion": CONSERVATIVE_SUGGESTION,
        "uncertainty": [short_text(reason, 180)],
    }


def safety_filter(decision: dict[str, Any], payload: dict[str, Any]
                  ) -> tuple[dict[str, Any], list[str]]:
    """Conservatively gate unsupported single-frame conclusions."""
    edited = {
        "risk_level": decision["risk_level"],
        "key_factors": list(decision["key_factors"]),
        "evidence_refs": list(decision["evidence_refs"]),
        "driving_suggestion": CONSERVATIVE_SUGGESTION,
        "uncertainty": list(decision["uncertainty"]),
    }
    flags = ["Replaced model driving suggestion with human-review-only guidance."]
    pedestrians = any(
        obj["class"] == "person" for obj in payload["YOLO_predictions"]
    )
    no_objects = len(payload["YOLO_predictions"]) == 0
    # Current fusion does not supply metric distance, motion history or TTC.
    # Do not accept an unsupported high-severity conclusion as an established risk.
    if edited["risk_level"] == "high":
        edited["risk_level"] = "unknown"
        flags.append("High risk downgraded: single-frame evidence has no TTC/verified geometry.")
        edited["uncertainty"].append("缺少目标的可靠距离、运动趋势与冲突时间。")
    elif edited["risk_level"] == "low" and (pedestrians or no_objects):
        edited["risk_level"] = "unknown"
        flags.append("Low risk downgraded: pedestrian proximity unknown or no detections.")
        edited["uncertainty"].append("不能仅凭单帧目标信息认定当前场景安全。")
    # Keep limits consistent with schema if any guard appended more strings.
    edited["uncertainty"] = list(dict.fromkeys(edited["uncertainty"]))[:5]
    return edited, flags


def evaluate_raw_response(
    raw: str, evidence_ids: set[str]
) -> tuple[dict[str, Any] | None, str | None]:
    """Accept only exact-schema output; never trim extra model items."""
    try:
        candidate = validate_decision(parse_model_json(raw), evidence_ids)
    except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        return None, f"{type(exc).__name__}: {short_text(exc, 200)}"
    return candidate, None


def make_repair_messages(
    base_messages: list[dict[str, str]], first_raw: str, error: str
) -> list[dict[str, str]]:
    """A single explicit repair, preserving the invalid original for audit."""
    feedback = (
        "你上一次给出的 JSON 没有通过格式校验："
        + short_text(error, 240)
        + "。请重新输出【完整】JSON 对象，而非局部补丁。"
        + "key_factors 只能有 1～3 条、uncertainty 只能有 1～3 条；"
        + "不要编造没有视觉证据的交通标志或颜色；"
        + "evidence_refs 只能从原始证据 ID 中选择。"
        + "不要 Markdown 代码块，不要解释；如果证据不充分请使用 unknown。"
        + "绝对不要直接截取旧列表来假装校验成功。"
    )
    return [
        *base_messages,
        {"role": "assistant", "content": first_raw[:2500]},
        {"role": "user", "content": feedback},
    ]


def infer_qwen(
    messages: list[dict[str, str]], model_dir: Path,
    device: str, max_new_tokens: int, evidence_ids: set[str]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load local Qwen ONCE, then generate at most two candidate JSON responses."""
    if not (model_dir / "config.json").is_file():
        raise FileNotFoundError("Local Qwen config.json missing at " + str(model_dir))
    if not ((model_dir / "model.safetensors").is_file()
            or list(model_dir.glob("model-*.safetensors"))
            or (model_dir / "pytorch_model.bin").is_file()):
        raise FileNotFoundError("Local Qwen weights missing in " + str(model_dir))

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable. Use --device cpu if required.")
        free, _ = torch.cuda.mem_get_info()
        if free < 1.5 * (1024 ** 3):
            raise RuntimeError("Less than 1.5 GiB GPU memory free; close CARLA first.")
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_dir), local_files_only=True, trust_remote_code=False,
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir), local_files_only=True, trust_remote_code=False,
        dtype=torch_dtype, low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    if device == "cuda":
        torch.cuda.synchronize()
    load_ms = round((time.perf_counter() - t0) * 1000, 2)

    attempt_logs: list[dict[str, Any]] = []
    active_messages = messages
    for number in (1, 2):
        tokens = tokenizer.apply_chat_template(
            active_messages, add_generation_prompt=True,
            tokenize=True, return_dict=True, return_tensors="pt",
        ).to(device)
        prompt_len = int(tokens["input_ids"].shape[-1])
        if prompt_len > 3500:
            raise ValueError("Prompt too long; trim evidence.")
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        with torch.inference_mode():
            output = model.generate(
                **tokens, do_sample=False, max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
            )
        if device == "cuda":
            torch.cuda.synchronize()
        generate_ms = round((time.perf_counter() - t1) * 1000, 2)
        generated = output[0][prompt_len:]
        raw = tokenizer.decode(generated, skip_special_tokens=True).strip()

        candidate, error = evaluate_raw_response(raw, evidence_ids)
        attempt_logs.append({
            "attempt_number": number,
            "raw_model_output": raw,
            "validation_status": "valid" if candidate is not None else "invalid",
            "validation_error": error,
            "candidate": candidate,
            "prompt_tokens": prompt_len,
            "generated_tokens": int(generated.shape[-1]),
            "generate_ms": generate_ms,
            "peak_cuda_memory_allocated_mib": (
                round(torch.cuda.max_memory_allocated() / 1048576, 1)
                if device == "cuda" else None
            ),
        })
        if candidate is not None:
            break
        if number == 1:
            active_messages = make_repair_messages(messages, raw, error or "Invalid JSON")

    stats = {
        "model_load_ms": load_ms,
        "generate_ms": round(sum(a["generate_ms"] for a in attempt_logs), 2),
        "attempt_count": len(attempt_logs),
        "prompt_tokens": attempt_logs[-1]["prompt_tokens"],
        "generated_tokens": sum(a["generated_tokens"] for a in attempt_logs),
        "peak_cuda_memory_allocated_mib": max(
            (a["peak_cuda_memory_allocated_mib"] or 0 for a in attempt_logs),
            default=0,
        ) if device == "cuda" else None,
    }
    return attempt_logs, stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fusion-report", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path,
                        help="Existing local Qwen2.5-0.5B-Instruct directory.")
    parser.add_argument("--ego-speed-kmh", type=float, default=None)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--max-new-tokens", type=int, default=320)
    args = parser.parse_args()
    if not 100 <= args.max_new_tokens <= 512:
        raise ValueError("--max-new-tokens must be in [100,512].")
    source = args.fusion_report.resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Fusion report does not exist: {source}")
    fusion = read_json(source)
    payload, evidence_ids = build_model_input(fusion, args.ego_speed_kmh)
    if args.model_dir:
        model_dir = args.model_dir.expanduser().resolve()
    elif os.environ.get("QWEN_MODEL_DIR"):
        model_dir = Path(os.environ["QWEN_MODEL_DIR"]).expanduser().resolve()
    else:
        raise ValueError(
            "Please pass --model-dir with the existing local Qwen checkpoint "
            "or set QWEN_MODEL_DIR; no model download will occur."
        )
    if not model_dir.is_dir():
        raise FileNotFoundError(
            f"Qwen model directory not found: {model_dir}. "
            "Check existing project config before downloading anything."
        )

    messages = prompt_for(payload)
    raw: str | None = None
    candidate: dict[str, Any] | None = None
    metrics: dict[str, Any] = {}
    errors: list[str] = []
    attempts: list[dict[str, Any]] = []
    status = "fallback"
    try:
        attempts, metrics = infer_qwen(
            messages, model_dir, args.device, args.max_new_tokens, evidence_ids,
        )
        for attempt in attempts:
            if attempt["validation_error"]:
                errors.append(
                    f"attempt {attempt['attempt_number']}: {attempt['validation_error']}"
                )
        raw = attempts[-1]["raw_model_output"]
        candidate = attempts[-1]["candidate"]
        if candidate is None:
            decision = fallback_unknown("两次模型生成均未通过严格的 JSON 结构校验。")
            policies = ["Fallback to unknown; unvalidated output is not a risk assessment."]
        else:
            decision, policies = safety_filter(candidate, payload)
            status = "valid" if len(attempts) == 1 else "valid_after_retry"
    except Exception as exc:
        errors.append(f"{type(exc).__name__}: {short_text(exc, 180)}")
        decision = fallback_unknown("本地模型运行或结构化校验失败；已保守降级。")
        policies = ["Fallback to unknown; model output must not control a vehicle."]

    root = Path(__file__).resolve().parent.parent
    out_dir = root / "evidence" / "scene_llm_risk"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / (
        f"llm_frame_{payload['world_frame']:08d}_{datetime.now():%Y%m%d_%H%M%S_%f}.json"
    )
    result = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "frame": payload["world_frame"],
        "sources": {
            "fusion_report": str(source),
            "local_llm_model_dir": str(model_dir),
            "baseline_yolo_rule_risk_for_comparison_only": (
                fusion.get("risk", {}).get("original_yolo_rule_baseline")
            ),
        },
        "input_to_llm": payload,
        "candidate_model_output_UNVERIFIED": candidate,
        "raw_model_output_for_debug": raw,
        "generation_attempts": attempts,
        "validation_status": status,
        "validation_errors": errors,
        "policy_actions": policies,
        "final_research_decision": decision,
        "runtime": {
            "device": args.device, "max_new_tokens": args.max_new_tokens,
            **metrics,
        },
        "limitations": [
            "Research-only; no vehicle control or production safety guarantees.",
            "CARLA audit truth deliberately excluded from model input.",
            "Single-frame image evidence does not establish physical TTC or imminent collision.",
            "risk_level is a conservative hypothesis, not an independent ground-truth label.",
        ],
    }
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite existing report: {out}")
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2),
                   encoding="utf-8")

    print("=== Phase 3.4: local Qwen risk experiment ===")
    print("Frame:", payload["world_frame"])
    print("Original YOLO baseline:", result["sources"]["baseline_yolo_rule_risk_for_comparison_only"])
    print("Structured JSON validation:", status)
    for attempt in attempts:
        print(
            f"Attempt {attempt['attempt_number']}: {attempt['validation_status']}"
            + (f" ({attempt['validation_error']})" if attempt["validation_error"] else "")
        )
    print("Model candidate:", json.dumps(candidate, ensure_ascii=False) if candidate else "(invalid)")
    print("Final research risk:", decision["risk_level"])
    for action in policies:
        print("Safety guard:", action)
    if errors:
        print("Error:", errors[0])
    print("Model generation ms:", metrics.get("generate_ms", "N/A"))
    print("Report:", out)
    if status in {"valid", "valid_after_retry"}:
        print("PASS: local Qwen generated schema-valid candidate (safety guards applied)")
    else:
        print("SAFE_FALLBACK: unknown (the actual LLM candidate was not validated)")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, TypeError, KeyError) as err:
        print(f"ERROR: {type(err).__name__}: {err}", file=sys.stderr)
        raise SystemExit(1)
