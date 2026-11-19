# -*- coding: utf-8 -*-
r"""
CARLA 仿真测试：在现有静止 ego 前方增添一辆静止测试车。

用途：让正在运行的 CARLA RGB → YOLO → VLM → Qwen 管线获得非空视觉目标。
不会移动 ego、不改变同步模式、地图、天气或其他 Actor，也不会调用任何模型。
只创建一辆测试车；按 Enter 时只删除本脚本创建的测试车。

使用（在项目根目录）：
  # Windows PowerShell：使用已安装 CARLA 的 Python 3.12 解释器。
  # 将 24 替换为当前 RGB 客户端使用的实际 ego Actor ID。
  py -3.12 -m scripts.carla_add_stationary_target --ego-id 24

要求：
  - CARLA 0.9.16 已运行，已有 ego（Actor ID 24 仅为示例，以当前会话为准）
  - RGB 客户端继续保持运行
  - FastAPI 127.0.0.1:8000 本地可访问
"""
from __future__ import annotations

import argparse
import json
import math
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

try:
    import carla
except ImportError as exc:
    raise SystemExit(
        "未找到 CARLA Python 包。请使用已安装 CARLA 的 Python 3.12 解释器运行。"
    ) from exc


def parser_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ego-id", type=int, required=True,
                        help="当前 RGB 客户端所绑定的 ego 车辆 Actor ID")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--api-url", default="http://127.0.0.1:8000")
    parser.add_argument("--scan-seconds", type=int, default=25,
                        help="测试车生成后，观察实时 YOLO 输出的时间上限（秒）")
    return parser.parse_args()


def horizontal_distance(a: carla.Location, b: carla.Location) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


def dot2(a: carla.Vector3D, b: carla.Vector3D) -> float:
    return float(a.x * b.x + a.y * b.y)


def live_latest(api_url: str) -> dict | None:
    request = Request(
        api_url.rstrip("/") + "/scene/live/latest",
        headers={"Cache-Control": "no-store", "User-Agent": "CARLALocalResearchTarget/1"},
    )
    try:
        with urlopen(request, timeout=2.5) as response:
            raw = response.read(200_001)
        if len(raw) > 200_000:
            return None
        data = json.loads(raw.decode("utf-8"))
        return data if isinstance(data, dict) else None
    except (HTTPError, URLError, TimeoutError, ValueError, OSError):
        return None


def choose_and_spawn(world, ego):
    """Safely choose a same-direction drivable lane position in front of ego."""
    start = ego.get_transform()
    forward = start.get_forward_vector()
    roads = world.get_map()
    library = world.get_blueprint_library()
    blueprint = library.find("vehicle.audi.a2")
    blueprint.set_attribute("role_name", "research_visual_target")

    # These are *simulator* distances for placing a controlled test actor.
    # They are never passed to YOLO, VLM, or Qwen.
    # Prefer a near but safe distance so a 960x540 camera can see the target.
    distances = (18.0, 22.0, 15.0, 26.0, 12.0, 30.0)
    vehicles = list(world.get_actors().filter("vehicle.*"))

    for d in distances:
        probe = carla.Location(
            x=start.location.x + forward.x * d,
            y=start.location.y + forward.y * d,
            z=start.location.z + 0.5,
        )
        waypoint = roads.get_waypoint(
            probe, project_to_road=True, lane_type=carla.LaneType.Driving
        )
        if waypoint is None:
            continue
        wp_transform = waypoint.transform
        wp_direction = wp_transform.get_forward_vector()
        # Avoid adjacent/crossing/contraflow roads and large position snapping.
        if dot2(forward, wp_direction) < 0.88:
            continue
        if horizontal_distance(probe, wp_transform.location) > 1.9:
            continue
        dx = wp_transform.location.x - start.location.x
        dy = wp_transform.location.y - start.location.y
        ahead = dx * forward.x + dy * forward.y
        lateral = abs(dx * -forward.y + dy * forward.x)
        if ahead < 10 or lateral > 2.1:
            continue
        if any(horizontal_distance(other.get_location(), wp_transform.location) < 7.0
               for other in vehicles if other.is_alive):
            continue

        location = carla.Location(
            x=wp_transform.location.x, y=wp_transform.location.y,
            z=wp_transform.location.z + 0.35,
        )
        transform = carla.Transform(location, wp_transform.rotation)
        try:
            target = world.try_spawn_actor(blueprint, transform)
        except RuntimeError:
            target = None
        if target is None:
            continue
        try:
            target.apply_control(carla.VehicleControl(
                throttle=0.0, brake=1.0, hand_brake=True
            ))
        except Exception:
            target.destroy()
            raise
        return target, ahead
    return None, None


def watch_visual_yolo(api_url: str, *, previous: dict | None, seconds: int) -> bool:
    prev_frame = (previous or {}).get("frame_id", -1)
    prev_session = (previous or {}).get("session_id")
    started = time.monotonic()
    seen_frame = -1
    print("开始只读观察 FastAPI /scene/live/latest 中的 YOLO 检测...", flush=True)
    while time.monotonic() - started < max(2, seconds):
        data = live_latest(api_url)
        if data:
            frame = data.get("frame_id")
            session = data.get("session_id")
            if type(frame) is int and frame > seen_frame:
                seen_frame = frame
                # A new frame after the target was spawned, not old persisted data.
                if (frame > prev_frame and
                        (prev_session is None or prev_session == session)):
                    detections = data.get("detections")
                    if isinstance(detections, list):
                        matched = [
                            item for item in detections
                            if (isinstance(item, dict)
                                and str(item.get("class_name", "")).strip().lower()
                                in {"car", "truck", "bus"}
                                and isinstance(item.get("box_xyxy"), list))
                        ]
                        if matched:
                            best = max(matched, key=lambda x: float(x.get("confidence", 0) or 0))
                            print(
                                "[YOLO_TARGET_CLASS_DETECTED] "
                                f"session={session} frame={frame} "
                                f"class={best.get('class_name')} "
                                f"confidence={best.get('confidence')} "
                                f"bbox={best.get('box_xyxy')}",
                                flush=True,
                            )
                            print(
                                "提示：这是图像检测证据，尚不能单凭 2D 框证明 "
                                "它一定对应刚刚创建的 CARLA Actor。",
                                flush=True,
                            )
                            return True
        time.sleep(0.7)
    print(
        "[YOLO_NOT_DETECTED] 新帧中暂未观察到 car/truck/bus 检测框。"
        "测试车仍保持在世界中；请查看 Dashboard 的摄像头画面。",
        flush=True,
    )
    return False


def main() -> int:
    args = parser_args()
    client = carla.Client(args.host, args.port)
    client.set_timeout(12.0)
    world = client.get_world()
    ego = world.get_actors().find(args.ego_id)

    if ego is None or not ego.is_alive or not ego.type_id.startswith("vehicle."):
        print(
            f"[ERROR] 找不到有效 ego Actor ID={args.ego_id}。"
            "请以 RGB 客户端当前输出的 vehicle actor id 为准。",
            flush=True,
        )
        return 2

    before = live_latest(args.api_url)
    if before:
        print(
            f"实时基线: session={before.get('session_id')}, "
            f"frame={before.get('frame_id')}, "
            f"detections={len(before.get('detections') or [])}",
            flush=True,
        )
    else:
        print("提醒：暂未读到 FastAPI 实时基线，稍后仍会尝试检测。", flush=True)

    print(f"CARLA map={world.get_map().name}, ego={ego.id} ({ego.type_id})", flush=True)
    target = None
    try:
        target, distance = choose_and_spawn(world, ego)
        if target is None:
            print(
                "[SPAWN_UNAVAILABLE] 没有找到前方安全、同向的可用生成点；"
                "未改变原有车辆或 CARLA 世界设置。",
                flush=True,
            )
            return 3
        print(
            f"[TARGET_SPAWNED] 测试车 Actor ID={target.id}, "
            f"type={target.type_id}, ego前方约{distance:.1f}m（仅布置场景）",
            flush=True,
        )
        watch_visual_yolo(args.api_url, previous=before, seconds=args.scan_seconds)
        print(
            "\n[ACTOR_HELD] 测试车将保持在 CARLA 中，"
            "供连续联动脚本选取有检测目标的关键帧。\n"
            "不要按 Enter，否则这辆测试车会被删除。",
            flush=True,
        )
        try:
            input("完成此轮验收后，按 Enter 只清理本脚本生成的测试车...")
        except EOFError:
            print("控制台输入已关闭，执行清理。", flush=True)
        return 0
    except KeyboardInterrupt:
        print("\n收到 Ctrl+C，开始清理本脚本创建的测试车。", flush=True)
        return 130
    finally:
        if target is not None:
            try:
                if target.is_alive:
                    target.destroy()
                    print(f"[TARGET_REMOVED] 已删除测试车 Actor ID={target.id}", flush=True)
            except RuntimeError as exc:
                print(f"[CLEANUP_WARNING] 清理测试车时出现错误：{exc}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
