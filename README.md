# 智能驾驶场景理解与风险分析系统 🚗

**CARLA · YOLO · VLM · Qwen · RAG · FastAPI**

基于 CARLA 仿真、目标检测、视觉语言证据和本地大语言模型的**智能驾驶多模态研究原型**。项目在原有 **Vehicle Agent + RAG Demo**（模拟座舱工具 Agent、汽车知识检索）基础上，新增 CARLA/YOLO 实时感知展示、离线 VLM/Qwen 风险分析、证据约束的 JSON 输出，以及三类仿真事件的探索性评测。

> **安全与范围声明**
>
> 本项目仅用于 **CARLA 仿真研究、可视化展示和人工复核**，不接入真实车辆，不向 CARLA 或真实车辆下发由模型生成的油门、制动、转向指令。风险等级不是可用于真实驾驶的安全保证。当前系统**不是实时 YOLO + VLM + Qwen 逐帧闭环系统**。

## 1. 项目状态（2026-11-18）

| 模块 | 已实现/已验证的范围 |
| --- | --- |
| CARLA 场景与 RGB 采集 | CARLA 0.9.16、`Town10HD_Opt`；前车减速、行人横穿、交通灯变化三组采集，合计 **103 个唯一帧** |
| 实时 YOLO 展示 | CARLA RGB → FastAPI → YOLO 检测与**二维启发式**风险提示 → Dashboard；曾实际展示成功 |
| 离线视觉证据融合 | 将已保存的 YOLO 检测和 VLM 文本声明对齐，生成结构化证据；与 CARLA 事后真值隔离 |
| 离线 Qwen 风险研究 | 本地 Qwen2.5-0.5B-Instruct 输出结构化 JSON，含证据引用、校验、最多一次重试及安全降级 |
| Dashboard V2 | 上部实时 YOLO 面板、下部**历史离线 Qwen 报告**；两部分已在浏览器展示 |
| 离线时序 Agent / RAG | 已实现针对保存轨迹/关键帧的 RAG + VLM + Qwen 研究回放路径；Dashboard 回放入口可见，最新一轮未单独点击验收完整回放 |
| 三场景评测 | CARLA 事件状态与 YOLO 报告按 `world_frame` 对齐；开展阈值明确的 **Proxy（启发式代理）** 对照 |
| 自动化测试 | 历史一次全项目运行结果为 `204 passed in 4.82s`；此后有新增修改，**不代表最新代码已再次完整回归** |
| GitHub 发布 | 源码整理与发布审查阶段；此处不宣称已完成远端推送 |

其中，**“源码存在”与“当次完整功能实测成功”是不同概念**。本 README 对已有证据和限制分别说明。

## 2. 系统架构与事实边界

```text
                   ┌──────── CARLA / 研究场景 ────────┐
                   │ RGB 帧、仿真状态（分开保存）      │
                   └──────────────┬───────────────────┘
                                  │ RGB
    [实时展示链路]                 ▼
      CARLA RGB 客户端 → POST /scene/live/frame
                       → YOLO 目标检测
                       → 2D 规则风险提示
                       → /scene/live/dashboard（LIVE）

    [离线单帧研究链路]
      已保存的 YOLO / VLM 对齐报告
                       → scene_fusion.py（证据融合）
                       → scene_llm_risk.py（本地 Qwen）
                       → JSON 结构校验 / 重试 / 安全降级
                       → 历史报告 → Dashboard（OFFLINE）

    [独立离线时序研究链路]
      已保存的车辆框 / 帧间连续性
                       → VLM 关键帧 + VehicleRAG
                       → Qwen 证据选择
                       → HTML / JSON 研究回放

    [事后评测链路]
      CARLA metadata.json（事件真值）
                       + YOLO 已保存预测（world_frame 对齐）
                       → 人工设定阈值的风险 Proxy 对照
```

**三个重要区别：**

1. **LIVE** 面板的风险是目标检测与二维规则产生的提示，不是实时 Qwen 决策。
2. **OFFLINE** 面板展示的是保存的历史帧 Qwen 研究结果，**不对应当前摄像头画面**。
3. CARLA 物理状态仅用于**事后评测**，不作为 YOLO/VLM/Qwen 的“视觉识别事实”偷偷输入模型。

当证据不足时，`UNKNOWN` 表示**无法可靠判断**，不代表安全，也不等于程序运行失败。

## 3. 核心功能

### 3.1 CARLA / YOLO 实时感知

- CARLA RGB 客户端上传 JPEG/PNG 图像到 FastAPI。
- YOLO 返回类别、置信度和 2D 边界框；2D 启发式规则生成研究性风险提示。
- Dashboard 显示摄像头、检测框、帧号、检测数量、处理延迟及会话状态。
- 页面在**未连接 RGB 客户端**时显示等待状态，属于正常情况。
- 实时会话保存在进程内存中，服务重启后不会自动保留。

主要入口：

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| `GET` | `/scene/live/dashboard` | 实时感知 + 历史离线研究 Dashboard |
| `POST` | `/scene/live/frame` | 供 CARLA RGB 客户端上传图像 |
| `GET` | `/scene/live/status` | 实时会话状态 |
| `GET` | `/scene/live/snapshot` | 实时画面快照 |
| `GET` | `/scene/live/research/reports` | 读取本机已保存的 Qwen 历史报告摘要 |
| `GET` | `/scene/live/research/replay` | 历史时序研究回放入口 |

### 3.2 YOLO / VLM 证据融合与 Qwen 风险分析

- `app/scene_fusion.py`：汇总 YOLO 检测与**未独立核实的** VLM 文本声明，生成 `scene_fusion_evidence.v1`。
- `app/scene_llm_risk.py`：从融合证据提取白名单字段，交给本地 Qwen 生成 `scene_llm_risk.v1` 风险候选。
- 风险候选字段：`risk_level`、`key_factors`、`evidence_refs`、`driving_suggestion`、`uncertainty`。
- 校验约束：风险等级与证据 ID 必须合法；最多重试一次；失败或证据不足时保守处理；实际驾驶建议会被替换成**人工复核用途**的安全提示。
- VLM 关于颜色、运动、人行道或灯色的自然语言描述只是**模型声明**，不能自动视为仿真真值。

### 3.3 独立离线时序研究 Agent

`scripts/driving_agent_mvp.py` 使用保存的 YOLO 车辆框和临时帧间连续性信息，结合 VLM 关键帧、原有 VehicleRAG 知识与 Qwen 证据选择，生成 HTML/JSON 回放。

**限制：**这是离线研究 MVP；临时目标关联**不等同于已完成标准 ByteTrack**；历史回放入口存在不表示在本轮已逐项完成页面验收。

### 3.4 保留的 Vehicle Agent + RAG 功能

本仓库最早是一个智能座舱教学原型，原有功能保留：

- **车辆工具 Agent**：查询模拟车辆状态、温度及提出模拟空调调整请求。
- **文本 Qwen**：将自然语言转为工具与参数，配合 Python 侧的白名单/参数校验。
- **写操作确认流程**：对部分模拟状态修改要求确认；**不应将其理解为生产级授权系统**。
- **VehicleRAG**：中文 BGE Embedding、ChromaDB、BM25、RRF 等检索或可用环境下的回退。
- **接口**：原有 `/assistant/chat`、`/agent/chat`、`/rag/ask` 等功能继续由同一个 FastAPI 应用提供。

> 某些历史模拟车辆状态接口可能允许绕过预期确认流程，因此**不要将整个 FastAPI 服务直接暴露公网**；此仓库默认仅按本地研究环境使用。

## 4. 主要目录

```text
app/
  main.py                      # 现有 FastAPI 主入口
  live_carla_api.py            # 实时 CARLA/YOLO + Dashboard + 历史报告入口
  multimodal_api.py            # 离线多模态研究接口
  scene_fusion.py              # YOLO/VLM 视觉证据融合
  scene_llm_risk.py            # 单帧 Qwen 风险候选 / JSON 校验 / 安全降级
  scene_grounded_packet.py     # 视觉证据相关研究模块
  scene_grounded_risk.py       # 风险研究相关模块
  rag_engine.py                # 原有 RAG 基础
scripts/
  carla_live_rgb_client.py     # RGB 客户端
  carla_lead_brake_demo.py     # 前车减速实验
  carla_pedestrian_demo.py     # 行人横穿实验
  carla_traffic_light_demo.py  # 交通灯变化实验
  driving_agent_mvp.py         # 独立离线时序研究
  scene_pipeline_bridge.py     # 证据融合 → Qwen 输入契约检查
  audit_yolo_baselines.py      # YOLO 基线数据审计
  build_carla_event_reference_v2.py  # CARLA 事件对齐（正式版）
  evaluate_carla_risk_proxies.py      # 探索性 Proxy 对照
schemas/
  scene_llm_risk_v1.schema.json
knowledge/
  mvp_risk_knowledge.md
tests/
  ...                          # 单元/接口/研究模块测试
```

说明：`scripts/build_carla_event_reference.py` 旧版曾存在**浮点制动命令解析问题**；正式流程以 `build_carla_event_reference_v2.py` 为准。

本地 `models/`、`evidence/`、`rag_data/`、模型缓存及个人配置不作为公开仓库的默认发布内容。

## 5. 本地运行与展示

### 5.1 运行前提

- Windows + Python 开发环境；历史运行中 **FastAPI/YOLO/Qwen 使用 Python 3.11**，**CARLA RGB 客户端使用与 CARLA API 兼容的 Python 3.12**。
- CARLA 0.9.16（用于可选实时采集）；已验证地图为 `Town10HD_Opt`。
- 本地 YOLO 权重、Qwen2.5-0.5B-Instruct、BGE 模型以及相应依赖需自行按现有环境准备。
- 历史开发设备约 **6 GB 显存**，大模型研究推荐串行执行，而非同时逐帧推理。

**仓库不提供大模型权重及原始 CARLA 实验图像。** 克隆代码不等于具备运行全部历史研究的原始数据。具体安装依赖请以仓库实际依赖配置和运行环境为准。

### 5.2 启动本地 API / Dashboard

在项目根目录、已有依赖的 Python 环境中执行：

```powershell
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

浏览器访问：

```text
http://127.0.0.1:8000/scene/live/dashboard
```

即使暂时未启动 CARLA，也可以打开页面；**历史离线研究面板需要本地已有对应 `evidence/` 报告**，否则不会凭空出现演示数据。

### 5.3 可选：连接 CARLA 实时 RGB

先启动 CARLA、创建带 `role_name=hero` 的场景车辆并启动上述 FastAPI，再在**适配 CARLA 客户端**的 Python 环境中运行：

```powershell
python -m scripts.carla_live_rgb_client `
  --host 127.0.0.1 --port 2000 `
  --api-url http://127.0.0.1:8000 `
  --role-name hero `
  --fps 2 --width 960 --height 540
```

`hero` Actor ID 是仿真会话内的临时编号，**不能复用历史实验的 Actor ID**。上述命令仅传输图像与展示研究提示，不形成模型驾驶控制链。

### 5.4 可选：对已有融合报告单帧运行 Qwen

如果本机已经有融合报告和 Qwen 模型，可以根据 CLI 帮助及实际路径设置参数。示意如下（将字符串占位符替换为自己的本机路径）：

```powershell
python -m app.scene_llm_risk `
  --fusion-report "PATH_TO_EXISTING_FUSION_REPORT.json" `
  --model-dir "PATH_TO_LOCAL_QWEN_MODEL" `
  --device cuda
```

此命令是**离线单帧研究**，不是 Dashboard 接收每帧时自动调用的在线 Qwen。

## 6. 三场景探索性评测

三组数据为三段**相关的仿真序列**，不是 103 个彼此独立的道路交通场景。行人场景曾对相同 48 帧重复运行 YOLO，重复报告**没有计为独立帧**。

| CARLA 场景 | 唯一帧数 | YOLO LOW | YOLO MEDIUM | YOLO UNKNOWN | YOLO HIGH |
| --- | ---: | ---: | ---: | ---: | ---: |
| 前车减速 | 25 | 23 | 2 | 0 | 0 |
| 行人横穿 | 48 | 32 | 15 | 1 | 0 |
| 交通灯绿→黄→红 | 30 | 7 | 8 | 15 | 0 |
| **合计** | **103** | **62** | **25** | **16** | **0** |

### 6.1 事件状态与规则基线

- **前车制动**：CARLA 记录制动激活 15 帧；对应的原 YOLO 二维风险规则均输出 `LOW`。这是基线**未随制动事件提高等级**的观察，不能直接称为 15 次真实危险漏报。
- **行人横穿**：48 帧中，按 CARLA 几何关系定义的“进入自车道横向范围”有 22 帧，其中原 YOLO 为 `LOW` 16 帧、`MEDIUM` 6 帧。仅进入横向范围不等同于已发生碰撞危险。
- **交通灯**：绿、黄、红状态各 10 帧，来自 CARLA 控制状态；这不证明 RGB 画面中的 VLM 能逐帧正确辨认灯色，更不意味着红灯时已发生闯红灯。

### 6.2 风险 Proxy：可复现，但不是真实危险标签

评测脚本使用人为设定的运动学/位置阈值构造风险**代理标签**，包括近似原点间距 TTC 和行人车道关注规则，交通灯组维持 `UNKNOWN` 代理。

| Proxy 对照指标 | 实际结果 |
| --- | ---: |
| 已对齐帧数 | 103 |
| 能分配非 `UNKNOWN` Proxy 的帧 | **37/103（35.9%）** |
| 在可评估帧上与代理标签严格一致的原 YOLO 输出 | **14/37（37.8%）** |
| Proxy 为 `MEDIUM/HIGH` 而原 YOLO 为 `LOW` | **21/27（77.8%）** |

**解释边界：**`37.8%` 不是独立危险真值上的分类准确率；`77.8%` **不是实际危险漏报率**。Proxy 使用人为阈值、简化几何与强相关序列，不足以证明真实道路泛化能力。

### 6.3 延迟与 Qwen 单帧记录

历史 YOLO 三场景单图像墙钟处理时长（**排除每组首帧**）：

| 场景 | 中位数 | P95 |
| --- | ---: | ---: |
| 前车减速 | 15.60 ms | 22.39 ms |
| 行人横穿 | 15.26 ms | 21.29 ms |
| 交通灯变化 | 15.47 ms | 19.19 ms |

首帧可能包含模型加载，上述数据不应描述为纯 GPU 前向耗时或多模态端到端延迟。

对**同一历史第 3488 帧**运行 Qwen2.5-0.5B-Instruct 的一次 Prompt 对照：

| 指标 | Prompt V1 | Prompt V2 |
| --- | --- | --- |
| 最终风险 | `UNKNOWN` | `UNKNOWN` |
| 有效证据引用数 | 0 | 3（`yolo:0/1/2`） |
| JSON 验证 | `valid` | `valid_after_retry` |
| 单帧文本生成耗时 | 3507.07 ms | 6665.29 ms |

V2 在这次单帧实验中增加了有效证据引用，但**经过一次重试且耗时更长**；不宣称整体风险判断准确率提高或推理加速。

## 7. 测试和可复现性边界

- 历史全量 pytest 结果：`204 passed in 4.82s`；记录发生在新增部分 Dashboard/评测修改之前。
- 已有证据桥接、YOLO 报告去重审计、事件对齐和 Proxy 分析的成功运行记录。
- 原始帧、模型权重、包含本地绝对路径的 `evidence/` 报告不随公开仓库提供；需使用自己的 CARLA 采集与模型产出才能重复相应实验。
- 对外发布前应执行一次**集中回归与安全审查**，不要把历史成功日志当作当前全部代码的最新验收。

## 8. 隐私、安全与研究局限

1. 仅供**本地研究和 CARLA 仿真**；默认监听 `127.0.0.1`，不要直接开放公网。
2. 不包含真实车辆控制；模型输出只用于解释、展示与人工复核。
3. 视觉检测框、VLM 文本声明、CARLA 状态与事后风险 Proxy **各自具有不同证据等级**，不可混为一谈。
4. 单帧 RGB 缺乏可靠的距离、速度、物体轨迹、制动距离和碰撞几何估计，`LOW` 不等于道路安全。
5. 低显存设备下，将实时 YOLO 与离线 Qwen 分区运行是**有意的资源和安全边界**。
6. 发布时应排除本地权重、原始 `evidence/`、向量数据库、`.env`、个人绝对路径、备份文件和私有截图。
7. 旧模拟车辆工具/状态接口可能不满足生产级鉴权要求；不得将其作为公共部署的安全实现示例。

## 9. 后续研究方向

后续可探索**事件触发的关键帧 VLM/Qwen 异步推理**、经标注的独立危险参考标签、跨场景/跨地图评测、稳态延迟测量与更稳健的时序关联。以上均为后续工作，**不属于本次已完成或已验证的能力**。

---

*研究原型说明截至 2026-11-18；本文中的数值均对应已记录的特定仿真序列与单次实验，不代表真实道路驾驶安全性能。*


## Dashboard V4.2 — Unified Vehicle AI Platform

### Latest features

- Unified cockpit Agent and multimodal driving research dashboard.
- Natural-language vehicle queries and simulated temperature control.
- Tool calling, parameter validation and user confirmation.
- VehicleRAG knowledge retrieval with source information.
- Real-time CARLA RGB and YOLO11n detection.
- Asynchronous keyframe analysis with SmolVLM2, Fusion and local Qwen.
- JSON validation laboratory and API request inspector.

### Web interfaces

- Unified Dashboard: http://127.0.0.1:8000/dashboard
- Driving Dashboard: http://127.0.0.1:8000/scene/live/dashboard
- Swagger API: http://127.0.0.1:8000/docs

### Validation

- Latest regression tests: 204 passed.
- Non-empty YOLO evidence verified through Fusion and Qwen input.
- Three CARLA research scenarios with 103 unique captured frames.
- Explicit evidence references and risk predictions remain model-dependent.

### Safety boundary

This project is intended for simulation, education and research.
It does not control real vehicles. Multimodal risk inference runs
asynchronously on selected keyframes, not on every live frame.
Cockpit tools operate on simulated vehicle state only.
Local model weights and experimental evidence are not included.
