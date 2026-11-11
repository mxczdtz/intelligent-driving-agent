"""汽车知识库检索评测，默认只测检索，不加载 Qwen。

运行：
    python -m evaluation.run_rag_eval --mode both
    python -m evaluation.run_rag_eval --mode bm25

指标：
    Recall@1 / Recall@3：人工标注的目标知识编号出现在Top-K的比例；
    MRR@3：目标编号排在前3时，倒数排名的平均值。
本数据集只有12条单标签教学样本，不代表真实汽车问答的泛化表现。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.rag_engine import VehicleRAG

ROOT = Path(__file__).resolve().parent
DEFAULT_CASES = ROOT / "rag_cases.json"
DEFAULT_RESULT = ROOT / "results" / "rag_retrieval_eval.json"


def first_match_rank(predicted_ids: list[str], target: str) -> int | None:
    try:
        return predicted_ids.index(target) + 1
    except ValueError:
        return None


def summarize(ranks: list[int | None]) -> dict:
    n = len(ranks)
    if n == 0:
        raise ValueError("没有评测样本")
    return {
        "count": n,
        "recall_at_1": round(sum(r == 1 for r in ranks) / n, 4),
        "recall_at_3": round(sum(r is not None and r <= 3 for r in ranks) / n, 4),
        "mrr_at_3": round(
            sum(1 / r if r is not None and r <= 3 else 0 for r in ranks) / n,
            4,
        ),
    }


def evaluate(
    cases_file: Path = DEFAULT_CASES,
    output_file: Path = DEFAULT_RESULT,
    mode: str = "both",
) -> dict:
    if mode not in {"bm25", "hybrid", "both"}:
        raise ValueError("mode 必须是 bm25/hybrid/both")

    cases = json.loads(cases_file.read_text(encoding="utf-8"))
    if not isinstance(cases, list) or not cases:
        raise ValueError("评测文件必须是非空 JSON 数组")

    expected_set = set()
    for case in cases:
        if not {"id", "question", "expected_id"} <= set(case):
            raise ValueError(f"样本缺少必要字段：{case!r}")
        if case["id"] in expected_set:
            raise ValueError(f"重复样本编号：{case['id']}")
        expected_set.add(case["id"])

    rag = VehicleRAG()
    valid_knowledge_ids = set(rag.by_id)
    for case in cases:
        if case["expected_id"] not in valid_knowledge_ids:
            raise ValueError(
                f"样本 {case['id']} 引用了不存在的知识编号：{case['expected_id']}"
            )

    if mode in {"hybrid", "both"}:
        rag.ensure_index()

    records = []
    bm25_ranks = []
    hybrid_ranks = []

    for case in cases:
        text = case["question"]
        expected = case["expected_id"]
        record = {
            "id": case["id"],
            "question": text,
            "expected_id": expected,
        }
        if mode in {"bm25", "both"}:
            bm25 = rag.search_bm25(text, top_k=3)
            bm25_ids = [doc["id"] for doc in bm25]
            rank = first_match_rank(bm25_ids, expected)
            bm25_ranks.append(rank)
            record["bm25"] = {
                "retrieved_ids": bm25_ids,
                "expected_rank": rank,
            }

        if mode in {"hybrid", "both"}:
            hybrid = rag.search(text, top_k=3)
            hybrid_ids = [doc["id"] for doc in hybrid]
            rank = first_match_rank(hybrid_ids, expected)
            hybrid_ranks.append(rank)
            record["hybrid"] = {
                "retrieved_ids": hybrid_ids,
                "expected_rank": rank,
            }

        records.append(record)
        print(
            f"{case['id']} | 目标: {expected} | "
            f"BM25: {record.get('bm25', {}).get('expected_rank', '-')} | "
            f"Hybrid: {record.get('hybrid', {}).get('expected_rank', '-')}"
        )

    report = {
        "evaluation_type": "single_label_retrieval",
        "dataset_file": cases_file.name,
        "total_cases": len(cases),
        "mode": mode,
        "metrics": {
            **({"bm25": summarize(bm25_ranks)} if bm25_ranks else {}),
            **({"hybrid": summarize(hybrid_ranks)} if hybrid_ranks else {}),
        },
        "cases": records,
        "notes": (
            "12条人工构造的教学检索样本；"
            "准确率只衡量知识编号检索，不衡量Qwen回答事实正确性。"
        ),
    }

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n=== RAG Retrieval Evaluation ===")
    for name, metrics in report["metrics"].items():
        print(
            f"{name}: "
            f"Recall@1={metrics['recall_at_1']:.2%}, "
            f"Recall@3={metrics['recall_at_3']:.2%}, "
            f"MRR@3={metrics['mrr_at_3']:.4f}"
        )
    print("Report:", output_file)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG 检索评测")
    parser.add_argument(
        "--mode",
        choices=["bm25", "hybrid", "both"],
        default="both"
    )
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--output", type=Path, default=DEFAULT_RESULT)
    args = parser.parse_args()
    evaluate(args.cases, args.output, args.mode)


if __name__ == "__main__":
    main()
