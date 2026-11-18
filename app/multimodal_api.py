"""FastAPI adapter for the verified Phase 5.0 offline multimodal replay.

The route delegates to the existing CLI pipeline instead of reimplementing it.
It only reads saved YOLO/track evidence and writes new JSON/HTML reports; it does
not connect to CARLA live streams or issue vehicle-control commands.
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import sys
import threading
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/scene", tags=["Multimodal Driving Risk Agent"])

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRACKS_DIR = (PROJECT_ROOT / "evidence" / "yolo_vehicle_box_tracks").resolve()
OUTPUT_DIR = (PROJECT_ROOT / "evidence" / "driving_agent_mvp").resolve()
RUN_ID_PATTERN = re.compile(r"agent_mvp_\d{8}_\d{6}_\d{6}")
_RUN_LOCK = threading.Lock()


class MultimodalAnalyzeRequest(BaseModel):
    """Run the offline multimodal agent on an existing saved track report."""

    tracks_report: str | None = Field(
        default=None,
        max_length=1024,
        description=(
            "Optional JSON path relative to the project root, or an absolute "
            "path inside evidence/yolo_vehicle_box_tracks. Defaults to newest report."
        ),
    )
    device: Literal["cuda", "cpu"] = Field(
        default="cuda",
        description="Inference device. CUDA is recommended for the configured RTX GPU.",
    )
    reuse_only_vlm: bool = Field(
        default=False,
        description="Reuse saved VLM reports only; do not infer missing keyframes.",
    )
    use_existing_rag: bool = Field(
        default=True,
        description=(
            "Use the existing local VehicleRAG (Chroma+BGE+BM25); "
            "fall back to BM25 or the MVP lexical notes without downloading models."
        ),
    )


def _resolve_tracks_report(value: str | None) -> Path:
    if value is None or not value.strip():
        if not TRACKS_DIR.is_dir():
            raise HTTPException(
                status_code=404,
                detail="未找到 evidence/yolo_vehicle_box_tracks 目录。",
            )
        candidates = list(TRACKS_DIR.glob("*.json"))
        if not candidates:
            raise HTTPException(
                status_code=404,
                detail="轨迹目录中没有 JSON 报告，请先运行现有 YOLO 轨迹脚本。",
            )
        selected = max(candidates, key=lambda item: item.stat().st_mtime)
    else:
        candidate = Path(value).expanduser()
        if not candidate.is_absolute():
            candidate = PROJECT_ROOT / candidate
        selected = candidate

    try:
        resolved = selected.resolve(strict=True)
    except OSError as exc:
        raise HTTPException(status_code=404, detail="指定的轨迹 JSON 文件不存在。") from exc

    if not resolved.is_file() or resolved.suffix.lower() != ".json":
        raise HTTPException(status_code=422, detail="tracks_report 必须是 JSON 文件。")
    if not resolved.is_relative_to(TRACKS_DIR):
        raise HTTPException(
            status_code=403,
            detail="为限制本地文件访问，tracks_report 只能位于 evidence/yolo_vehicle_box_tracks。",
        )
    return resolved


def _resolve_qwen_dir() -> Path:
    configured = os.getenv("QWEN_MODEL_DIR")
    if configured:
        candidate = Path(configured).expanduser()
        if not candidate.is_absolute():
            candidate = PROJECT_ROOT / candidate
    else:
        # Matches the project handover layout: <pythonProject2>/models/Qwen...
        candidate = PROJECT_ROOT.parent / "models" / "Qwen2.5-0.5B-Instruct"
    resolved = candidate.resolve()
    if not (resolved / "config.json").is_file():
        raise HTTPException(
            status_code=503,
            detail=(
                "未找到本地 Qwen 模型。请设置环境变量 QWEN_MODEL_DIR，"
                "指向已有的 Qwen2.5-0.5B-Instruct 目录；接口不会下载模型。"
            ),
        )
    return resolved


def _report_file(run_id: str, suffix: str) -> Path:
    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise HTTPException(status_code=404, detail="报告 ID 无效。")
    path = (OUTPUT_DIR / f"{run_id}.{suffix}").resolve()
    if not path.is_relative_to(OUTPUT_DIR) or not path.is_file():
        raise HTTPException(status_code=404, detail="报告文件不存在。")
    return path


@router.post("/analyze-multimodal", summary="运行离线多模态驾驶风险 Agent")
def analyze_multimodal(request: MultimodalAnalyzeRequest):
    """Run saved YOLO tracks -> VLM -> local RAG -> Qwen -> HTML/JSON."""
    tracks_path = _resolve_tracks_report(request.tracks_report)
    qwen_dir = _resolve_qwen_dir()

    # Avoid simultaneous model loads competing for the same 6 GB GPU.
    if not _RUN_LOCK.acquire(blocking=False):
        raise HTTPException(
            status_code=409,
            detail="已有多模态分析任务正在运行，请等待该任务完成后再提交。",
        )

    try:
        command = [
            sys.executable,
            "-m",
            "scripts.driving_agent_mvp",
            "--tracks-report",
            str(tracks_path),
            "--qwen-model-dir",
            str(qwen_dir),
            "--device",
            request.device,
        ]
        if request.reuse_only_vlm:
            command.append("--reuse-only-vlm")
        if request.use_existing_rag:
            command.append("--use-existing-rag")

        child_env = os.environ.copy()
        child_env["PYTHONUTF8"] = "1"
        child_env["PYTHONIOENCODING"] = "utf-8"
        child_env["HF_HUB_OFFLINE"] = "1"
        child_env["TRANSFORMERS_OFFLINE"] = "1"

        # Resolve BGE for the child process explicitly. Do not rely only on
        # whether the shell that launched Uvicorn happened to export it.
        configured_embed = child_env.get("RAG_EMBED_MODEL", "").strip()
        embed_candidates = []
        if configured_embed:
            embed_candidates.append(Path(configured_embed).expanduser())
        embed_candidates.extend([
            PROJECT_ROOT / "models" / "bge-small-zh-v1.5",
            PROJECT_ROOT.parent / "models" / "bge-small-zh-v1.5",
        ])
        resolved_embed = next(
            (
                candidate.resolve()
                for candidate in embed_candidates
                if (candidate / "modules.json").is_file()
                and (candidate / "config.json").is_file()
            ),
            None,
        )
        if resolved_embed is not None:
            child_env["RAG_EMBED_MODEL"] = str(resolved_embed)
        child_env.setdefault("RAG_EMBED_DEVICE", "cpu")
        logger.info(
            "Launching multimodal MVP with RAG_EMBED_MODEL=%s, RAG_EMBED_DEVICE=%s",
            child_env.get("RAG_EMBED_MODEL", "<default>"),
            child_env["RAG_EMBED_DEVICE"],
        )

        try:
            completed = subprocess.run(
                command,
                cwd=str(PROJECT_ROOT),
                env=child_env,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=900,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise HTTPException(
                status_code=504,
                detail="多模态分析超过 15 分钟；请检查模型推理或显存状态。",
            ) from exc
        except OSError as exc:
            logger.exception("Unable to start multimodal MVP subprocess")
            raise HTTPException(
                status_code=503,
                detail="无法启动多模态分析进程，请确认当前 FastAPI 使用 llm-gpu 环境。",
            ) from exc

        if completed.returncode != 0:
            logger.error(
                "Multimodal MVP failed. stdout tail:\n%s\nstderr tail:\n%s",
                completed.stdout[-5000:],
                completed.stderr[-5000:],
            )
            raise HTTPException(
                status_code=503,
                detail={
                    "message": "多模态 Agent 执行失败。",
                    "stdout_tail": completed.stdout[-2500:],
                    "stderr_tail": completed.stderr[-2500:],
                },
            )

        json_match = re.search(r"^JSON:\s*(.+?)\s*$", completed.stdout, re.MULTILINE)
        html_match = re.search(r"^HTML:\s*(.+?)\s*$", completed.stdout, re.MULTILINE)
        if not json_match or not html_match:
            logger.error("MVP output paths missing. stdout:\n%s", completed.stdout[-5000:])
            raise HTTPException(
                status_code=500,
                detail="Agent 已退出，但没有在控制台输出中找到 JSON/HTML 报告路径。",
            )

        json_path = Path(json_match.group(1)).resolve()
        html_path = Path(html_match.group(1)).resolve()
        if (
            not json_path.is_relative_to(OUTPUT_DIR)
            or not html_path.is_relative_to(OUTPUT_DIR)
            or not json_path.is_file()
            or not html_path.is_file()
        ):
            raise HTTPException(
                status_code=500,
                detail="Agent 报告路径校验失败或输出文件不存在。",
            )

        try:
            report = json.loads(json_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.exception("Unable to read generated MVP JSON")
            raise HTTPException(status_code=500, detail="Agent JSON 报告无法读取。") from exc

        run_id = json_path.stem
        decision = report.get("qwen_research_decision", {})
        qwen = report.get("qwen_inference", {})
        rag_runtime = report.get("rag_runtime", {})
        return {
            "ok": True,
            "status": "pass" if "PASS:" in completed.stdout else "partial",
            "run_id": run_id,
            "frame_count": report.get("frame_count"),
            "vlm_keyframes": sorted(
                int(key) for key in report.get("vlm_keyframes", {}).keys()
            ),
            "qwen_validation": qwen.get("validation_status"),
            "qwen_generation_ms": qwen.get("generation_total_ms"),
            "research_risk_candidate": decision.get("risk_level", "unknown"),
            "selected_evidence_ids": [
                item.get("id") for item in decision.get("selected_factors", [])
            ],
            "rag_runtime": rag_runtime,
            "json_path": str(json_path.relative_to(PROJECT_ROOT)),
            "html_path": str(html_path.relative_to(PROJECT_ROOT)),
            "player_url": f"/scene/multimodal/{run_id}/player",
            "report_url": f"/scene/multimodal/{run_id}/report",
            "stdout_tail": completed.stdout[-1800:],
        }
    finally:
        _RUN_LOCK.release()


@router.get("/multimodal/{run_id}/player", summary="查看多模态 HTML 回放")
def get_multimodal_player(run_id: str):
    path = _report_file(run_id, "html")
    return FileResponse(
        path,
        media_type="text/html",
        headers={
            "Cache-Control": "no-store",
            "Content-Disposition": f'inline; filename="{run_id}.html"',
        },
    )


@router.get("/multimodal/{run_id}/report", summary="获取多模态 JSON 报告")
def get_multimodal_report(run_id: str):
    path = _report_file(run_id, "json")
    return FileResponse(
        path,
        media_type="application/json",
        filename=f"{run_id}.json",
        headers={"Cache-Control": "no-store"},
    )
