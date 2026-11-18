# Driving Risk Agent MVP — local research reference notes

This document is a local research reference for the offline MVP. It is not
CARLA scenario ground truth, a safety standard, or an independently validated
risk classifier. Its sections provide interpretive guardrails for model output.

## Detection boxes and geometric uncertainty
YOLO 的像素检测框是一种视觉模型预测。box_xyxy 的面积、中心位置以及图像中物体的大小，不能独立换算成以米为单位的前车距离。框变大可能与接近有关，也可能受到相机视角、遮挡、检测抖动等因素影响。没有相机标定和深度信息时，不要报告实际制动距离或真实 TTC。

## Across-frame association and target identity
对 car、truck、bus 检测框按相邻帧 IoU 与中心位移进行匹配，只能得到图像空间的暂定轨迹 ID。T001、T004 不是 CARLA actor ID；类别从 car 变成 truck 不表示已经证实为同一辆车。轨迹不再出现可能是漏检、遮挡、匹配失败、类别变化或不同目标，必须检查图像。

## Model claims and temporal risk
单帧 YOLO 原始风险 low、medium、high 是已有规则结果，不是独立的危险标签。前车是否处于接近状态，不应仅由单帧标签确定。即使跨帧框连续，也不能从框的增大直接计算闭合速度或碰撞概率。没有足够的可信证据时，研究模型可以输出 unknown。

## VLM image description verification
SmolVLM2 的文字描述可能有幻觉，包括车辆颜色、交通灯颜色、交通标志、道路参与者数量和相对位置。语言生成结果必须以 unverified 形式记录。VLM 的“车辆正在减速”等表述不能只靠一张图像当成可核验运动事实。

## RAG knowledge versus scene evidence
检索得到的知识片段只能作为研究背景，不能证明本次 CARLA 图像中真的存在危险目标。RAG 文档、YOLO 检测输出、VLM 文本以及 Qwen 风险结论应保留独立的来源与可信度说明。能通过 JSON schema 只说明结构正确，不代表推理正确。

## Safety separation and evaluation
CARLA 仿真中的车辆速度、相对位置和 actor ID 可以在独立的离线评测阶段使用，但是不能偷偷加入视觉模型提示词，使模型得到部署时无法观测的信息。本 MVP 只提供离线科研解释，不输出用于真实车辆的控制命令，也不声称完成安全验证。
