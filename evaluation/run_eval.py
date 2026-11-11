
import sys
import json
import time
import statistics
from pathlib import Path

import torch


# ==========================================
# 0. 将项目根目录加入 Python 模块搜索路径
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ==========================================
# 1. 导入项目模块
# ==========================================

from app.agent_planner import plan_tool
from app.qwen_engine import load_qwen

# ==========================================
# 1. 导入项目模块
# ==========================================



# ==========================================
# 1. 路径配置
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

CASES_FILE = BASE_DIR / "cases.json"

RESULT_DIR = BASE_DIR / "results"

RESULT_FILE = RESULT_DIR / "eval_report.json"


# ==========================================
# 2. 百分位计算
# ==========================================

def percentile(values, percent):

    if not values:
        return 0.0

    values = sorted(values)

    position = (len(values) - 1) * percent

    lower = int(position)
    upper = min(lower + 1, len(values) - 1)

    weight = position - lower

    return (
        values[lower] * (1 - weight)
        + values[upper] * weight
    )


# ==========================================
# 3. 主评测函数
# ==========================================

def run_evaluation():

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用")

    print("=" * 55)
    print("Vehicle Qwen Agent Evaluation")
    print("=" * 55)

    print("GPU:", torch.cuda.get_device_name(0))

    # 读取评测数据
    with open(
        CASES_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        cases = json.load(f)

    print("Test cases:", len(cases))

    # ======================================
    # 4. 加载模型（只加载一次）
    # ======================================

    print("\nLoading Qwen model...")

    load_start = time.perf_counter()

    tokenizer, model = load_qwen()

    load_seconds = time.perf_counter() - load_start

    print("Model loaded successfully")

    print(
        "Model loading seconds:",
        round(load_seconds, 3)
    )

    # ======================================
    # 5. 预热模型
    # ======================================

    print("\nWarming up model...")

    plan_tool("现在车内温度是多少？")

    torch.cuda.synchronize()

    print("Warm-up completed")

    # 预热结束后，单独统计评测阶段峰值显存
    torch.cuda.reset_peak_memory_stats()

    # ======================================
    # 6. 开始逐条评测
    # ======================================

    records = []

    tool_correct = 0
    argument_correct = 0
    planner_valid = 0

    latencies = []

    print("\nStarting evaluation...\n")

    for index, case in enumerate(cases, start=1):

        case_id = case["id"]
        message = case["message"]

        expected_name = case["expected_name"]
        expected_arguments = case["expected_arguments"]

        print(
            f"[{index}/{len(cases)}] "
            f"Testing {case_id}: {message}"
        )

        torch.cuda.synchronize()

        start = time.perf_counter()

        try:

            # 真正调用本地 Qwen 模型
            prediction = plan_tool(message)

            torch.cuda.synchronize()

            elapsed = time.perf_counter() - start

            planner_valid += 1

            predicted_name = prediction["name"]

            predicted_arguments = prediction["arguments"]

            # 工具名称是否正确
            name_correct = (
                predicted_name == expected_name
            )

            # 参数是否完全正确
            args_correct = (
                predicted_arguments == expected_arguments
            )

            # 工具名称与参数都正确
            exact_correct = (
                name_correct and args_correct
            )

            if name_correct:
                tool_correct += 1

            if exact_correct:
                argument_correct += 1

            error = None

        except Exception as exc:

            torch.cuda.synchronize()

            elapsed = time.perf_counter() - start

            predicted_name = None
            predicted_arguments = None

            name_correct = False
            args_correct = False
            exact_correct = False

            error = str(exc)

        latencies.append(elapsed)

        record = {
            "id": case_id,
            "message": message,
            "expected_name": expected_name,
            "predicted_name": predicted_name,
            "expected_arguments": expected_arguments,
            "predicted_arguments": predicted_arguments,
            "planner_valid": error is None,
            "tool_correct": name_correct,
            "arguments_correct": args_correct,
            "exact_correct": exact_correct,
            "latency_seconds": round(elapsed, 4),
            "error": error
        }

        records.append(record)

        print("Expected:", expected_name)
        print("Predicted:", predicted_name)
        print("Tool correct:", name_correct)
        print("Arguments correct:", args_correct)
        print("Latency:", round(elapsed, 3), "seconds")
        print("-" * 55)

    # ======================================
    # 7. 计算评测指标
    # ======================================

    total = len(cases)

    if total == 0:
        raise ValueError("评测数据集不能为空")

    tool_accuracy = tool_correct / total

    argument_accuracy = argument_correct / total

    json_valid_rate = planner_valid / total

    average_latency = statistics.mean(latencies)

    p50_latency = percentile(latencies, 0.50)

    p95_latency = percentile(latencies, 0.95)

    # ======================================
    # 8. GPU 显存统计
    # ======================================

    torch.cuda.synchronize()

    allocated_memory = (
        torch.cuda.memory_allocated() / 1024**3
    )

    peak_memory = (
        torch.cuda.max_memory_allocated() / 1024**3
    )

    # ======================================
    # 9. 生成评测报告
    # ======================================

    report = {

        "model": "Qwen2.5-0.5B-Instruct",

        "gpu": torch.cuda.get_device_name(0),

        "total_cases": total,

        "tool_correct": tool_correct,

        "argument_exact_correct": argument_correct,

        "planner_valid": planner_valid,

        "metrics": {

            "tool_accuracy": round(
                tool_accuracy, 4
            ),

            "argument_exact_accuracy": round(
                argument_accuracy, 4
            ),

            "json_valid_rate": round(
                json_valid_rate, 4
            ),

            "average_latency_seconds": round(
                average_latency, 4
            ),

            "p50_latency_seconds": round(
                p50_latency, 4
            ),

            "p95_latency_seconds": round(
                p95_latency, 4
            ),

            "model_loading_seconds": round(
                load_seconds, 4
            ),

            "allocated_gpu_memory_gib": round(
                allocated_memory, 4
            ),

            "peak_gpu_memory_gib": round(
                peak_memory, 4
            )
        },

        "experiments": records
    }

    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        RESULT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            report,
            f,
            ensure_ascii=False,
            indent=2
        )

    # ======================================
    # 10. 输出实验结果
    # ======================================

    print("\n")
    print("=" * 55)
    print("EVALUATION SUMMARY")
    print("=" * 55)

    print("Total cases:", total)

    print(
        "Tool accuracy:",
        f"{tool_accuracy:.2%}"
    )

    print(
        "Argument exact accuracy:",
        f"{argument_accuracy:.2%}"
    )

    print(
        "JSON valid rate:",
        f"{json_valid_rate:.2%}"
    )

    print(
        "Average latency:",
        round(average_latency, 4),
        "seconds"
    )

    print(
        "P50 latency:",
        round(p50_latency, 4),
        "seconds"
    )

    print(
        "P95 latency:",
        round(p95_latency, 4),
        "seconds"
    )

    print(
        "Peak GPU memory:",
        round(peak_memory, 4),
        "GiB"
    )

    print("\nReport saved to:")
    print(RESULT_FILE)

    print("\nEvaluation finished!")


if __name__ == "__main__":

    run_evaluation()