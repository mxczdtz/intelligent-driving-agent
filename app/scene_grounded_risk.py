r"""
Phase 3.7 — grounded Qwen risk hypotheses for CARLA research (offline).

Input: Phase 3.6 grounded_frame_*.json, NOT CARLA metadata or an image.
Output: evidence/scene_grounded_risk/grounded_risk_frame_*.json.

The local Qwen model selects EXISTING candidate statement IDs and limitation IDs,
rather than writing arbitrary risk factors or evidence citations. The program
resolves each selected ID to its deterministic statement and exact provenance.
Neither source agreement nor JSON validity is independent driving-risk truth.

Qwen loads once; failed schema selection gets one repair attempt. If both fail,
output a conservative UNKNOWN response, preserving both raw outputs. The
original YOLO baseline, all earlier reports, and the CARLA truth stay untouched.

Windows PowerShell example (project root):
    & $PY -m app.scene_grounded_risk `
      --packet-report ".\evidence\scene_grounded_packet\grounded_frame_00003488_20261116_170946_592370.json" `
      --model-dir "PATH_TO_LOCAL_QWEN_MODEL"
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

PACKET_SCHEMA = "scene_grounded_packet.v1"
OUTPUT_SCHEMA = "scene_grounded_risk.v1"
RISK_LEVELS = {"unknown", "low", "medium", "high"}
SELECTION_FIELDS = {"risk_level", "selected_claim_ids", "limitation_ids"}
REVIEW_ONLY_ADVICE = (
    "仅用于 CARLA 仿真研究及人工审核；不能据此控制真实车辆，"
    "缺少可靠距离和时序证据时应保持谨慎。"
)

SYSTEM_PROMPT = """你是智能驾驶仿真实验中的证据选择助手，不是汽车控制器。
你只需要选择已有的候选陈述ID和局限性ID，不要自创陈述、证据ID或驾驶指令。
候选陈述只描述模型输出及其相互关系，并不代表道路实情已验证。
仅凭单帧RGB不能知道实际车道关联、TTC、接近速度、灯色真值或将来碰撞。
若缺少可靠的空间关联或时序证据，优先选择 risk_level=unknown。
你必须且只能输出如下三个键组成的一个完整 JSON 对象：
{"risk_level":"unknown","selected_claim_ids":["candidate:pedestrian_count_agreement"],"limitation_ids":["limit:unknown_geometry"]}
risk_level 只能为 unknown/low/medium/high。
selected_claim_ids 只能从用户给出的 candidate_claim_ids 中选择 0～2 个；
limitation_ids 只能从用户给出的 limitation_ids 中选择 1～2 个。
不要多写键、解释、Markdown代码块或新事实。"""


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError("Expected top-level JSON object.")
    return data


def as_string(value: Any, label: str, max_len: int = 300) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > max_len:
        raise ValueError(f"{label} must be nonempty text, <= {max_len} characters.")
    return value.strip()


def packet_index(packet: dict[str, Any]) -> tuple[dict[str, Any], dict[str, dict[str, Any]],
                                                  dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Build a narrowly whitelisted LLM input and validate cross-reference topology."""
    if packet.get("schema_version") != PACKET_SCHEMA:
        raise ValueError(f"Expected {PACKET_SCHEMA} packet.")
    frame = packet.get("world_frame")
    if not isinstance(frame, int) or isinstance(frame, bool) or frame < 0:
        raise ValueError("Invalid world_frame.")

    isolation = packet.get("source_isolation")
    if not isinstance(isolation, dict) or any(
        isolation.get(key) is not False
        for key in (
            "CARLA_audit_truth_used",
            "original_YOLO_rule_risk_used",
            "Qwen_generated_claims_used",
        )
    ):
        raise ValueError("Source-isolation declaration missing or not False.")

    raw_evidence = packet.get("evidence")
    raw_candidates = packet.get("candidate_statements_for_human_review")
    if not isinstance(raw_evidence, list) or not 1 <= len(raw_evidence) <= 220:
        raise ValueError("Invalid evidence list.")
    if not isinstance(raw_candidates, list) or len(raw_candidates) > 30:
        raise ValueError("Invalid candidate-statement list.")

    evidence_by_id: dict[str, dict[str, Any]] = {}
    for evidence in raw_evidence:
        if not isinstance(evidence, dict):
            raise ValueError("Evidence item is not an object.")
        eid = as_string(evidence.get("id"), "evidence ID", 80)
        if eid in evidence_by_id:
            raise ValueError("Duplicate evidence ID: " + eid)
        evidence_by_id[eid] = evidence

    limitation_by_id: dict[str, dict[str, Any]] = {}
    for eid, evidence in evidence_by_id.items():
        if eid.startswith("limit:"):
            if evidence.get("kind") != "measurement_limitation":
                raise ValueError(f"Limitation {eid} lacks measurement_limitation kind.")
            observation = as_string(evidence.get("observation"), "limitation text", 300)
            limitation_by_id[eid] = {
                "id": eid,
                "observation": observation,
                "epistemic_status": as_string(
                    evidence.get("epistemic_status"), "limitation status", 100
                ),
            }
    if not limitation_by_id:
        raise ValueError("No declared limitation evidence IDs found.")

    candidates_by_id: dict[str, dict[str, Any]] = {}
    for candidate in raw_candidates:
        if not isinstance(candidate, dict):
            raise ValueError("Candidate is not an object.")
        cid = as_string(candidate.get("claim_id"), "candidate ID", 100)
        if not cid.startswith("candidate:") or cid in candidates_by_id:
            raise ValueError("Duplicate or malformed candidate ID.")
        statement = as_string(candidate.get("text"), "candidate statement", 400)
        status = as_string(candidate.get("status"), "candidate status", 100)
        if status not in {
            "model_report_agreement_not_scene_truth",
            "model_report_count_difference_unverified",
            "unverified_visual_color_claim",
        }:
            raise ValueError("Unsupported candidate epistemic status.")
        refs = candidate.get("support_refs")
        if not isinstance(refs, list) or not 1 <= len(refs) <= 32:
            raise ValueError(f"{cid}: invalid support_refs.")
        if len(set(refs)) != len(refs):
            raise ValueError(f"{cid}: duplicate support_refs.")
        missing = [r for r in refs if r not in evidence_by_id]
        if missing:
            raise ValueError(f"{cid}: unknown support refs {missing}.")

        # A comparison ID must carry the original compared evidence IDs.
        for ref in refs:
            if ref.startswith("cross:"):
                cross = evidence_by_id[ref]
                if cross.get("kind") != "cross_model_comparison":
                    raise ValueError(f"{ref}: expected cross_model_comparison.")
                deps = cross.get("dependencies")
                if not isinstance(deps, list) or not deps:
                    raise ValueError(f"{ref}: missing dependencies.")
                if len(set(deps)) != len(deps):
                    raise ValueError(f"{ref}: duplicate dependencies.")
                if not set(deps).issubset(refs):
                    raise ValueError(f"{cid}: source refs missing comparison dependencies.")
                if any(dep not in evidence_by_id for dep in deps):
                    raise ValueError(f"{ref}: unknown dependency.")
        candidates_by_id[cid] = {
            "claim_id": cid,
            "text": statement,
            "support_refs": list(refs),
            "status": status,
        }

    # Include only enumerated, validated items, not raw CARLA truth, old risk,
    # free-form VLM prose, or an old Qwen decision.
    ego = evidence_by_id.get("ego:speed")
    speed = None
    if ego is not None:
        observation = ego.get("observation")
        if not isinstance(observation, dict):
            raise ValueError("Malformed ego:speed evidence.")
        value = observation.get("speed_kmh")
        if isinstance(value, bool) or not isinstance(value, (float, int)):
            raise ValueError("Invalid ego_speed_kmh.")
        if not math.isfinite(value) or not 0 <= value <= 300:
            raise ValueError("Implausible ego_speed_kmh.")
        speed = round(float(value), 3)

    prompt_payload = {
        "world_frame": frame,
        "input_scope": "single_RGB_frame_model_outputs_only",
        "ego_speed_kmh_logged": speed,
        "candidate_claim_ids": [
            {"id": cid, "text": c["text"], "status": c["status"]}
            for cid, c in candidates_by_id.items()
        ],
        "limitation_ids": [
            {"id": lid, "limitation": l["observation"]}
            for lid, l in limitation_by_id.items()
        ],
    }
    return prompt_payload, candidates_by_id, limitation_by_id, evidence_by_id


def make_messages(payload: dict[str, Any]) -> list[dict[str, str]]:
    user = (
        "请只根据下面列出的已有证据ID选择，不要生成新的描述。"
        "候选陈述并非真实危险标签；若无法判断应输出unknown。"
        "请仅输出三个键的JSON对象。\n"
        + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, val in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key: " + key)
        result[key] = val
    return result


def parse_selection(raw: str, candidates: dict[str, dict[str, Any]],
                    limits: dict[str, dict[str, Any]]) -> dict[str, Any]:
    stripped = raw.strip()
    match = re.fullmatch(r"```(?:json)?\s*\n?(.*?)\n?```", stripped,
                         flags=re.I | re.S)
    if match:
        stripped = match.group(1).strip()
    selected = json.loads(stripped, object_pairs_hook=reject_duplicate_keys)
    if not isinstance(selected, dict) or set(selected) != SELECTION_FIELDS:
        raise ValueError(f"Output must contain exactly {sorted(SELECTION_FIELDS)}.")
    if selected["risk_level"] not in RISK_LEVELS:
        raise ValueError("Invalid risk_level.")
    claim_ids = selected["selected_claim_ids"]
    limitation_ids = selected["limitation_ids"]
    if not isinstance(claim_ids, list) or len(claim_ids) > 2:
        raise ValueError("selected_claim_ids must be an array of 0 to 2 IDs.")
    if not isinstance(limitation_ids, list) or not 1 <= len(limitation_ids) <= 2:
        raise ValueError("limitation_ids must be an array of 1 to 2 IDs.")
    if any(not isinstance(x, str) for x in claim_ids + limitation_ids):
        raise ValueError("Selection IDs must be strings.")
    if len(set(claim_ids)) != len(claim_ids) or len(set(limitation_ids)) != len(limitation_ids):
        raise ValueError("Duplicate selection IDs.")
    if not set(claim_ids).issubset(candidates):
        raise ValueError("Selected candidate ID not found in input packet.")
    if not set(limitation_ids).issubset(limits):
        raise ValueError("Selected limitation ID not found in input packet.")
    if selected["risk_level"] != "unknown" and not claim_ids:
        raise ValueError("Non-unknown risk must reference at least one candidate.")
    return selected


def parse_or_error(raw: str, candidates: dict[str, dict[str, Any]],
                   limits: dict[str, dict[str, Any]]
                   ) -> tuple[dict[str, Any] | None, str | None]:
    try:
        return parse_selection(raw, candidates, limits), None
    except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        return None, f"{type(exc).__name__}: {str(exc)[:190]}"


def repair_messages(original: list[dict[str, str]], raw: str, error: str
                    ) -> list[dict[str, str]]:
    return [
        *original,
        {"role": "assistant", "content": raw[:1800]},
        {
            "role": "user",
            "content": (
                "之前的输出未通过严格校验：" + error[:200] + "。"
                "请重新生成完整JSON对象，只包含 risk_level、"
                "selected_claim_ids、limitation_ids 三个字段。"
                "必须从上一条消息列出的候选ID和局限性ID中选择；"
                "selected_claim_ids 最多2个，limitation_ids 1～2个。"
                "不能新增事实、ID、文字因素、解释或Markdown；"
                "信息不够就选择 risk_level=unknown。"
            ),
        },
    ]


def load_and_generate(
    messages: list[dict[str, str]],
    model_dir: Path,
    device: str,
    max_new_tokens: int,
    candidates: dict[str, dict[str, Any]],
    limits: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Single weight load; maximum two model-generation attempts."""
    if not (model_dir / "config.json").is_file():
        raise FileNotFoundError("Missing Qwen config.json in " + str(model_dir))
    if not (
        (model_dir / "model.safetensors").is_file()
        or list(model_dir.glob("model-*.safetensors"))
        or (model_dir / "pytorch_model.bin").is_file()
    ):
        raise FileNotFoundError("Missing Qwen model weights in " + str(model_dir))

    # Force local-only mode. There is no downloading in this module.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable. Close competing GPU apps or use CPU.")
        free, _ = torch.cuda.mem_get_info()
        if free < 1.5 * 1024 ** 3:
            raise RuntimeError("Less than 1.5 GiB VRAM free; close CARLA.")
        dtype = torch.float16
    else:
        dtype = torch.float32

    start_load = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_dir), local_files_only=True, trust_remote_code=False
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir), local_files_only=True, trust_remote_code=False,
        dtype=dtype, low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    if device == "cuda":
        torch.cuda.synchronize()
    load_ms = round((time.perf_counter() - start_load) * 1000, 2)

    attempt_logs: list[dict[str, Any]] = []
    active_messages = messages

    for n in (1, 2):
        tokens = tokenizer.apply_chat_template(
            active_messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt",
        ).to(device)
        prompt_len = int(tokens["input_ids"].shape[-1])
        if prompt_len > 3000:
            raise ValueError("Prompt longer than 3000 tokens; reduce packet size.")

        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        start_generate = time.perf_counter()
        with torch.inference_mode():
            output = model.generate(
                **tokens,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
            )
        if device == "cuda":
            torch.cuda.synchronize()
        generated = output[0][prompt_len:]
        raw = tokenizer.decode(generated, skip_special_tokens=True).strip()
        elapsed = round((time.perf_counter() - start_generate) * 1000, 2)

        selection, error = parse_or_error(raw, candidates, limits)
        attempt_logs.append({
            "attempt_number": n,
            "raw_output": raw,
            "candidate_selection": selection,
            "validation_status": "valid" if selection else "invalid",
            "validation_error": error,
            "prompt_tokens": prompt_len,
            "generated_tokens": int(generated.shape[-1]),
            "generate_ms": elapsed,
            "peak_cuda_allocated_mib": (
                round(torch.cuda.max_memory_allocated() / (1024 ** 2), 1)
                if device == "cuda" else None
            ),
        })
        if selection is not None:
            break
        if n == 1:
            active_messages = repair_messages(messages, raw, error or "Invalid JSON.")

    stats = {
        "model_load_ms": load_ms,
        "generation_total_ms": round(sum(x["generate_ms"] for x in attempt_logs), 2),
        "generation_attempt_count": len(attempt_logs),
        "generated_tokens_total": sum(x["generated_tokens"] for x in attempt_logs),
        "peak_cuda_allocated_mib": (
            max(x["peak_cuda_allocated_mib"] for x in attempt_logs)
            if device == "cuda" else None
        ),
    }
    return attempt_logs, stats


def assemble_decision(selection: dict[str, Any] | None,
                      candidates: dict[str, dict[str, Any]],
                      limits: dict[str, dict[str, Any]]
                      ) -> tuple[dict[str, Any], list[str]]:
    """Ground statements by ID, then conservatively gate unsupported severity."""
    if selection is None:
        return {
            "risk_level": "unknown",
            "factors": [],
            "limitations": [
                {"id": lid, "description": record["observation"]}
                for lid, record in list(limits.items())[:2]
            ],
            "driving_suggestion": REVIEW_ONLY_ADVICE,
            "decision_status": "schema_fallback_no_valid_model_selection",
        }, ["Invalid or unavailable LLM response; defaulted to unknown."]

    selected_factors = [
        {
            "claim_id": cid,
            "statement": candidates[cid]["text"],
            "source_refs": candidates[cid]["support_refs"],
            "epistemic_status": candidates[cid]["status"],
        }
        for cid in selection["selected_claim_ids"]
    ]
    selected_limits = [
        {"id": lid, "description": limits[lid]["observation"]}
        for lid in selection["limitation_ids"]
    ]

    requested_risk = selection["risk_level"]
    guarded_risk = requested_risk
    actions: list[str] = []
    # The initial Phase 3.6 packet contains no verified lane geometry, distance
    # or temporal motion. A high/low claim should NOT become an established
    # severe/safe class merely because a language model selected it.
    if requested_risk in {"high", "low"}:
        guarded_risk = "unknown"
        actions.append(
            f"{requested_risk} downgraded to unknown: no validated distance, "
            "lane association, TTC or independent danger label."
        )
    if requested_risk == "medium":
        actions.append(
            "Medium is an uncalibrated cautionary research hypothesis, not a verified hazard."
        )
    if requested_risk == "unknown":
        actions.append("Insufficient evidence accepted as unknown.")

    return {
        "risk_level": guarded_risk,
        "factors": selected_factors,
        "limitations": selected_limits,
        "driving_suggestion": REVIEW_ONLY_ADVICE,
        "decision_status": "unverified_research_hypothesis",
    }, actions


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet-report", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=None)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--max-new-tokens", type=int, default=220)
    args = parser.parse_args()

    if not 80 <= args.max_new_tokens <= 400:
        raise ValueError("--max-new-tokens must be from 80 to 400.")
    packet_path = args.packet_report.expanduser().resolve()
    if not packet_path.is_file():
        raise FileNotFoundError(f"Evidence packet not found: {packet_path}")
    packet = read_json(packet_path)
    prompt_payload, candidates, limits, _ = packet_index(packet)

    model_path = args.model_dir
    if model_path is None and os.environ.get("QWEN_MODEL_DIR"):
        model_path = Path(os.environ["QWEN_MODEL_DIR"])
    if model_path is None:
        raise ValueError("Provide the existing local --model-dir or QWEN_MODEL_DIR.")
    model_path = model_path.expanduser().resolve()
    if not model_path.is_dir():
        raise FileNotFoundError(f"Local Qwen directory not found: {model_path}")

    attempts: list[dict[str, Any]] = []
    stats: dict[str, Any] = {}
    status = "fallback"
    error: str | None = None
    selection: dict[str, Any] | None = None
    try:
        attempts, stats = load_and_generate(
            make_messages(prompt_payload), model_path, args.device,
            args.max_new_tokens, candidates, limits,
        )
        selection = attempts[-1]["candidate_selection"]
        if selection is not None:
            status = "valid" if len(attempts) == 1 else "valid_after_retry"
        else:
            error = "Both Qwen output attempts failed strict selection validation."
    except (RuntimeError, OSError, ValueError, TypeError, ImportError) as exc:
        error = f"{type(exc).__name__}: {str(exc)[:240]}"

    decision, guard_actions = assemble_decision(selection, candidates, limits)
    result = {
        "schema_version": OUTPUT_SCHEMA,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "world_frame": prompt_payload["world_frame"],
        "source_packet_report": str(packet_path),
        "local_model_dir": str(model_path),
        "llm_prompt_payload": prompt_payload,
        "llm_selection_unverified": selection,
        "attempts": attempts,
        "validation_status": status,
        "failure_reason": error,
        "post_validation_decision": decision,
        "safety_guards_applied": guard_actions,
        "runtime": {"device": args.device, **stats},
        "source_isolation": {
            "CARLA_audit_truth_used": False,
            "original_YOLO_rule_risk_used": False,
            "prior_Qwen_decisions_used": False,
        },
        "limitations": [
            "Per-factor references show provenance but do not verify truth.",
            "A one-frame visual description cannot prove distance, TTC, or road safety.",
            "No real vehicle control, calibrated risk estimate, or independent danger label.",
        ],
    }

    root = Path(__file__).resolve().parent.parent
    directory = root / "evidence" / "scene_grounded_risk"
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / (
        f"grounded_risk_frame_{prompt_payload['world_frame']:08d}_"
        f"{datetime.now():%Y%m%d_%H%M%S_%f}.json"
    )
    if destination.exists():
        raise FileExistsError("Refusing to overwrite an existing risk report.")
    destination.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("=== Phase 3.7: per-factor grounded Qwen experiment ===")
    print("Frame:", prompt_payload["world_frame"])
    print("Validated candidate statements available:", len(candidates))
    print("Qwen JSON validation:", status)
    for attempt in attempts:
        print(
            f"Attempt {attempt['attempt_number']}: {attempt['validation_status']}"
            + (f" ({attempt['validation_error']})" if attempt["validation_error"] else "")
        )
    print("Qwen selected:", json.dumps(selection, ensure_ascii=False) if selection else "(invalid)")
    print("Final research risk:", decision["risk_level"])
    for factor in decision["factors"]:
        print("Factor:", factor["statement"])
        print("  Source refs:", json.dumps(factor["source_refs"], ensure_ascii=False))
    for action in guard_actions:
        print("Safety guard:", action)
    if error:
        print("Error:", error)
    print("Generation total ms:", stats.get("generation_total_ms", "N/A"))
    print("Report:", destination)
    if selection:
        print("PASS: Qwen selected valid existing evidence IDs; per-factor refs attached")
    else:
        print("SAFE_FALLBACK: unknown (no schema-valid Qwen selection)")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
