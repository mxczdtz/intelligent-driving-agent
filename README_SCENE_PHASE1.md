# 阶段一：YOLO 道路图像识别与初步风险提示

这是在现有 `intelligent-driving-agent` 上**新增的独立模块**，不替换现有
`app/main.py`、Qwen、RAG、车辆工具和测试。可以单独启动，也可以通过脚本
安全地接入统一 FastAPI 实例。

## 这一步真实实现了什么

- 接收本地 JPEG/PNG 道路图像，实际调用 Ultralytics YOLO 检测道路目标。
- 输出对象类别、置信度、像素检测框。
- 根据二维检测框位置和**输入的自车速度**产生教学级风险提示。
- 使用 Pydantic 限制输出结构；生成依据说明与能力边界。
- 提供可 Mock 的单元/API 测试，不依赖 GPU 或下载权重。
- **不控制车辆、不估距、不判断灯色、不预测碰撞时间，也不是 VLM/LLM 风险推理。**

## 1. 复制文件（在原项目目录）

将 ZIP 解压后，把其中的 `app/`、`scripts/`、`tests/` 和
`requirements_scene.txt`、`README_SCENE_PHASE1.md` **合并**到原项目根目录。
不要删除或覆盖原项目同名目录；ZIP 中只包含新文件。

建议先运行：

```powershell
Set-Location "E:\SoftWareFiles\pycharmFiles\pythonProject2\vehicle-agent-rag-demo"
git status -sb
$PY = "D:\anaconda3\envs\llm-gpu\python.exe"
& $PY --version
```

如果 `git status` 显示你原来就有未提交的改动，先自行保存，别盲目覆盖。

## 2. 安装视觉阶段的额外依赖

```powershell
& $PY -m pip install -r requirements_scene.txt
```

不会自动重装本地 Qwen/BGE，理论上不需要重装 CUDA PyTorch。
如果 pip 想更新现有 torch，请停止并先核对依赖兼容性。

## 3. 跑不需要权重的测试

```powershell
& $PY -m pytest tests/test_scene_risk.py tests/test_scene_api.py -v
```

## 4. 准备 YOLO 权重并运行视觉服务

Ultralytics 可能会在首次用 `yolo11n.pt` 时自动联网下载权重。
推荐将合法获取的权重单独放在被 Git 忽略的 `models/` 中。
请核对模型许可，尤其当项目将来用于商业用途时。

```powershell
# 根据你本地真实文件位置修改，不要照抄不存在的路径
$env:SCENE_YOLO_MODEL = (Resolve-Path ".\models\yolo11n.pt").Path
$env:SCENE_YOLO_DEVICE = "cpu"  # 先验证功能；之后可改成 "0" 使用 CUDA
& $PY -m uvicorn app.scene_service:app --host 127.0.0.1 --port 8001
```

浏览器打开 `http://127.0.0.1:8001/docs`，找到
`POST /scene/analyze`，上传你有使用权的道路 JPG/PNG 图像；
`speed_kmh` 是你手工输入的演示参数，不是从图像推断出的速度。

在另一 PowerShell 终端也可使用（替换成你本地真实图片）：

```powershell
curl.exe -X POST "http://127.0.0.1:8001/scene/analyze" `
  -F "file=@.\sample_road.jpg" `
  -F "speed_kmh=30"
```

## 5. 可选：接入现有主服务

仅在独立服务测试完成之后执行：

```powershell
& $PY -m scripts.attach_scene_router
& $PY -m pytest tests -v
git diff -- app/main.py
& $PY -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

这样原来的 `/assistant/chat` 保持不变，
新的 `/scene/analyze` 同样可在 `http://127.0.0.1:8000/docs` 找到。
脚本会将旧 `app/main.py` 备份到已有忽略规则覆盖的 `evidence/`。

## 6. 结果示例（示意，非真实模型运行记录）

```json
{
  "image_width": 640,
  "image_height": 480,
  "ego_speed_kmh": 35.0,
  "detections": [
    {"class_name": "person", "confidence": 0.95,
     "box_xyxy": [260.0, 120.0, 400.0, 460.0]}
  ],
  "risk": {
    "risk_level": "high",
    "key_factors": ["行人检测框位于画面中央偏下位置，需关注潜在冲突。"],
    "evidence": ["detection_0: person, conf=0.95"],
    "driving_suggestion": "检测到较强的图像风险提示；应由更完整的感知与安全系统复核，不自动刹车或转向。",
    "method": "2D bounding-box heuristic (educational prototype)",
    "limitations": ["..."]
  },
  "inference_ms": 0.0
}
```

## 7. 严格的项目能力边界

- 只进行了二维视觉检测和 **Python 启发式风险提示**。
- `speed_kmh` 需要外部提供；绝不是图像估计出来的真实速度。
- 画面中央不等于本车道，二维框大小不能用来当真实距离；
  单帧算法无法证明前车减速或行人将要横穿。
- 交通灯只能识别检测类别；**不能断言红灯/绿灯**。
- 没有 CARLA、VLM、LLM 多模态推理和独立风险评测。
- YOLO 检测是否可用及性能需要你在本机用真实图片与权重验证。
- 禁止暴露公网或接入真车控制。
- 若发生新依赖冲突，可单独建环境，避免破坏旧 Agent 的已验证运行环境。

## 后续里程碑

阶段二接入 CARLA RGB 相机 + 车辆状态与场景标签；
阶段三接入 VLM 和 LLM 风险分析；
阶段四进行独立的多场景评测和 GitHub 文档展示。
