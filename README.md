# Vehicle Agent + RAG Demo 🚗

> 基于本地 **Qwen2.5-0.5B-Instruct**、FastAPI、ChromaDB、BM25 和 RRF 的汽车智能座舱教学原型。车辆数据全部为**模拟状态**，**不连接真实车辆**。

## 项目能力

- **车辆工具 Agent**：查询模拟车辆状态、车内温度、申请修改模拟空调温度。
- **本地大模型**：Qwen 将自然语言规划为工具名称和参数；Python 对工具白名单及参数做独立校验。
- **写操作确认**：写操作先保存待确认计划，再经客户端确认执行；令牌过期、一次性消费、会话 Cookie 隔离（教学级）。
- **汽车知识库 RAG**：中文 BGE Embedding + ChromaDB 语义检索 + BM25 关键词检索 + RRF 融合。
- **高风险故障回答**：高风险匹配采用知识库原文摘录，避免小模型改写安全要点。
- **统一入口**：`POST /assistant/chat` 用保守规则分流车辆操作与汽车知识问答。
- **自动化评测**：pytest API 测试、Qwen 工具规划开发集、RAG 检索 Recall@K / MRR@K。

## 架构

```text
用户自然语言 -> /assistant/chat
                   |
                   +-> 规则分流 -> 车辆操作 -> Qwen 工具规划
                   |                        -> Python 参数/白名单校验
                   |                        -> 只读直接执行 / 写操作待确认
                   |                        -> /agent/confirm 单次执行模拟操作
                   |
                   +-> 汽车故障知识 -> BGE Embedding + ChromaDB
                                     + BM25 中文 N-gram
                                     -> RRF 融合 Top-K
                                     -> 高风险文档摘录 / Qwen 文档问答
                                     -> 回答 + 知识编号
```

## 项目结构（核心文件）

```text
app/
  main.py               # FastAPI 接口
  agent_planner.py      # Qwen 工具规划与模型结果检查
  qwen_engine.py        # 本地 Qwen 加载与推理
  pending_actions.py    # 待确认动作、一次性令牌、会话存储
  intent_router.py      # 统一入口的规则分流
  rag_engine.py         # Chroma + BM25 + RRF + 回答
knowledge/
  vehicle_knowledge.json
evaluation/
  cases.json
  run_eval.py
  rag_cases.json
  run_rag_eval.py
tests/
  test_vehicle_api.py
  test_rag_api.py
  test_rag_safety.py
  test_unified_api.py
requirements.txt
.gitignore
```

## 环境与模型准备

- 推荐 **Windows / Python 3.11**；其他系统可自行适配。
- GPU 推理在当前版本要求 NVIDIA GPU + 可用的 CUDA PyTorch。
- Qwen 和 BGE 模型权重**不在本仓库分发**，需要自行合法下载到本地。
- 首次使用时，先通过 [PyTorch 官方安装页面](https://pytorch.org/get-started/locally/) 安装适合本机 CUDA 的 PyTorch，再安装其余依赖：

```powershell
python -m pip install -r requirements.txt
```

**模型配置：** 在启动项目的同一个 PowerShell 里设置本地路径（修改成自己的路径）。

```powershell
# Qwen 模型所在文件夹内应存在 config.json、tokenizer.json、model.safetensors 等
$env:QWEN_MODEL_DIR = "D:\models\Qwen2.5-0.5B-Instruct"

# BGE 模型路径；通过 ModelScope 下载后指定本地文件夹
$env:RAG_EMBED_MODEL = (Resolve-Path ".\models\bge-small-zh-v1.5").Path

# Embedding 默认用 CPU，减少与 Qwen 争抢 GPU 显存
$env:RAG_EMBED_DEVICE = "cpu"
```

可从 [Hugging Face Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) 与 [BGE-small-zh-v1.5](https://huggingface.co/BAAI/bge-small-zh-v1.5) 获得模型信息，并遵守各模型许可条款。请勿把权重提交到 Git。

> `.env.example` 仅供参考。该项目目前读取系统环境变量，**不会自动解析 `.env` 文件**。

## 初始化知识库索引

```powershell
python -m app.rag_engine --bm25-only "发动机温度过高怎么办？"
python -m app.rag_engine --build
python -m app.rag_engine --query "发动机温度过高怎么办？"
python -m app.rag_engine --ask "发动机温度过高怎么办？"
```

首次 `--build` 会加载本地 Embedding 模型并将 Chroma 索引保存到 `rag_data/`，这个目录不应提交到 Git。

## 启动 API

```powershell
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

打开 <http://127.0.0.1:8000/docs> 查看 Swagger API 文档。

### 统一入口请求示例

`POST /assistant/chat`：

```json
{
  "message": "发动机温度过高怎么办？",
  "mode": "auto",
  "top_k": 3
}
```

常见输入与预期路由：

| 输入 | 路由 | 说明 |
|---|---|---|
| 发动机温度过高怎么办？ | `knowledge` | RAG 检索、回答和来源 |
| 现在车内温度是多少？ | `vehicle` | 查询模拟车内温度 |
| 请把空调温度设置到23度。 | `vehicle` | 仅返回待确认操作 |
| 请帮我打开车门。 | `unsupported` | 不执行任何车辆操作 |

`mode` 可设置为 `auto`、`vehicle`、`knowledge`。自动模式是**保守规则**而非高精度多意图分类器。

对于写操作，响应将包含 `confirmation_token` 和 `confirmation_required`。只有用户明确核对操作内容后，在同一会话 Cookie 下调用 `POST /agent/confirm`，才会执行一次模拟写操作：

```json
{
  "confirmation_token": "从上一步响应中获取",
  "approved": true
}
```

**切勿把示例接口接到真实车辆控制系统。**

## 测试

```powershell
python -m pytest tests -v
```

项目开发记录中，本地完整测试曾达到 **31 passed**；真实统一入口集成测试也曾返回 `PASS: Unified Agent integration`。请以当前机器、当前提交实际运行结果为准。

### RAG 检索评测

```powershell
python -m evaluation.run_rag_eval --mode both
```

本地12条人工编写的检索样本，开发记录显示 BM25 和 Hybrid 均达到 Recall@1=100%、Recall@3=100%、MRR@3=1.0000。

**解释范围：** 这些测试问题与知识库紧密相关，不代表未知真实问题的性能，也不证明 Hybrid 优于 BM25；检索正确更不等于生成回答安全可靠。

### 工具规划开发集（12条）

| Prompt 版本 | 工具选择准确率 | 工具与参数完全匹配 |
|---|---:|---:|
| V1 | 100% | 91.67% |
| V2 | 83.33% | 83.33% |
| V3 | 91.67% | 91.67% |

这12条已被用于 Prompt 迭代，属于开发集对比，不能作为独立泛化测试结果。原始运行报告可能包含本机绝对路径，因此默认不提交 `evaluation/results/`。

## 安全边界和待办

当前仅适合作为**单进程、本地、教学原型**：

1. `/tool/call` 与 `/vehicle/temperature` 可直接修改模拟温度，尚未统一鉴权；不要暴露公网。
2. 会话 Cookie 不能替代登录认证，缺少完整的 CSRF 防护与权限系统。
3. 模拟车辆状态是进程内全局字典，多个用户并不隔离；令牌存储也不能跨进程共享。
4. 汽车知识是自编教学示例，并非官方维修手册；高风险回答不能替代专业诊断。
5. 开发集规模小，需要额外收集未见问题，测试幻觉、越权请求、错误路由和多用户状态隔离。

## 发布前检查

确保 `.gitignore` 生效。**不要上传**模型权重、虚拟环境、Chroma 数据、原始敏感日志、个人绝对路径或 API 凭证。发布前检查 `git status --short`，并逐项核对暂存区内容。
