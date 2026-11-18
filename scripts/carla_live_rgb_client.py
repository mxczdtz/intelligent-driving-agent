"""Phase 5.2.1 CARLA RGB camera client.

Attach an RGB sensor to an existing vehicle and POST JPEG frames to the local
FastAPI service. This script never calls vehicle control, world.tick(), or
changes CARLA synchronous-mode settings.
"""
from __future__ import annotations

import argparse
import io
import os
from pathlib import Path
import queue
import signal
import sys
import threading
import time
import uuid


def _load_carla(carla_root: str | None):
    """Load the installed CARLA package first.

    A .whl is an installation archive, not a safe sys.path entry for CARLA's
    native .pyd extension. The previous implementation prepended the wheel to
    sys.path, which could shadow the already-installed site-packages package.
    """
    import importlib
    import traceback

    # Primary path: the user already installed the cp312 wheel with pip.
    try:
        carla = importlib.import_module("carla")
        # Access a known API symbol to force native extension resolution now.
        _ = carla.Client
        print(f"[CARLA] Loaded installed package from: {getattr(carla, '__file__', '<namespace>')}")
        return carla
    except Exception as installed_error:
        installed_trace = traceback.format_exc()

    # Fallback for CARLA distributions that ship a directly importable .egg.
    root_value = carla_root or os.environ.get("CARLA_ROOT")
    egg_errors = []
    if root_value:
        root = Path(root_value).expanduser()
        dist = root / "PythonAPI" / "carla" / "dist"
        if dist.is_dir():
            eggs = sorted(dist.glob("carla-*.egg"))
            for egg in eggs:
                if str(egg) not in sys.path:
                    sys.path.insert(0, str(egg))
                try:
                    carla = importlib.import_module("carla")
                    _ = carla.Client
                    print(f"[CARLA] Loaded egg package from: {egg}")
                    return carla
                except Exception as exc:
                    egg_errors.append(f"{egg}: {type(exc).__name__}: {exc}")

    message = (
        "无法导入 CARLA Python API。当前客户端优先尝试已通过 pip 安装的 "
        "site-packages 包；不会再把 .whl 直接塞入 sys.path。"
        f"\nInstalled-package import error: {type(installed_error).__name__}: {installed_error}"
        f"\nPython: {sys.executable}"
    )
    if root_value:
        message += f"\nCARLA_ROOT: {root_value}"
    if egg_errors:
        message += "\nEgg fallback errors: " + " | ".join(egg_errors)
    message += (
        "\n请先用同一个解释器验证："
        '\n  "C:\\Users\\DELL\\AppData\\Local\\Programs\\Python\\Python312\\python.exe" '
        '-c "import carla; print(carla.__file__); print(carla.Client)"'
        "\n如果该命令成功而客户端仍失败，请检查项目中是否存在 carla.py 或 carla 文件夹遮蔽已安装包。"
        "\n原始导入堆栈：\n" + installed_trace
    )
    raise RuntimeError(message) from installed_error


def _select_vehicle(world, vehicle_id: int | None, role_name: str):
    vehicles = list(world.get_actors().filter("vehicle.*"))
    if vehicle_id is not None:
        for actor in vehicles:
            if int(actor.id) == vehicle_id:
                return actor
        raise RuntimeError(f"找不到 vehicle-id={vehicle_id}。当前车辆数：{len(vehicles)}")

    matches = [
        actor for actor in vehicles
        if actor.attributes.get("role_name", "") == role_name
    ]
    if len(matches) == 1:
        return matches[0]
    if len(vehicles) == 1:
        return vehicles[0]
    listing = [
        f"id={a.id}, role_name={a.attributes.get('role_name', '')}, type={a.type_id}"
        for a in vehicles[:20]
    ]
    raise RuntimeError(
        f"无法唯一确定自车（role-name={role_name!r}）。"
        "请传入 --vehicle-id。当前车辆：" + ("; ".join(listing) or "无")
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--carla-root", default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--timeout", type=float, default=8.0)
    parser.add_argument("--api-url", default="http://127.0.0.1:8000")
    parser.add_argument("--role-name", default="hero")
    parser.add_argument("--vehicle-id", type=int, default=None)
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--fov", type=float, default=90.0)
    parser.add_argument("--duration-sec", type=float, default=0.0,
                        help="0 means run until Ctrl+C.")
    args = parser.parse_args()

    if not 0.2 <= args.fps <= 10:
        parser.error("--fps 必须在 0.2 到 10 之间。")
    if not 320 <= args.width <= 1920 or not 240 <= args.height <= 1080:
        parser.error("图像尺寸范围：320..1920 x 240..1080。")

    try:
        import httpx
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("缺少 httpx 或 Pillow；它们应已在项目环境中安装。") from exc

    carla = _load_carla(args.carla_root)
    api = args.api_url.rstrip("/")
    session_id = uuid.uuid4().hex[:16]
    frame_queue: queue.Queue = queue.Queue(maxsize=1)
    stop_event = threading.Event()
    stats = {"sent": 0, "failed": 0, "replaced": 0}
    client = httpx.Client(timeout=8.0)

    # Verify API before attaching the sensor.
    health = client.get(api + "/scene/live/status")
    health.raise_for_status()

    carla_client = carla.Client(args.host, args.port)
    carla_client.set_timeout(args.timeout)
    world = carla_client.get_world()
    vehicle = _select_vehicle(world, args.vehicle_id, args.role_name)

    blueprint_library = world.get_blueprint_library()
    camera_bp = blueprint_library.find("sensor.camera.rgb")
    camera_bp.set_attribute("image_size_x", str(args.width))
    camera_bp.set_attribute("image_size_y", str(args.height))
    camera_bp.set_attribute("fov", str(args.fov))
    camera_bp.set_attribute("sensor_tick", str(1.0 / args.fps))
    if camera_bp.has_attribute("enable_postprocess_effects"):
        camera_bp.set_attribute("enable_postprocess_effects", "true")

    transform = carla.Transform(
        carla.Location(x=1.6, z=2.25),
        carla.Rotation(pitch=-8.0),
    )
    camera = world.spawn_actor(camera_bp, transform, attach_to=vehicle)

    def on_image(carla_image):
        try:
            picture = Image.frombytes(
                "RGBA",
                (carla_image.width, carla_image.height),
                bytes(carla_image.raw_data),
                "raw",
                "BGRA",
            ).convert("RGB")
            buffer = io.BytesIO()
            picture.save(buffer, format="JPEG", quality=78, optimize=True)
            item = (
                int(carla_image.frame),
                float(carla_image.timestamp),
                buffer.getvalue(),
            )
            try:
                frame_queue.put_nowait(item)
            except queue.Full:
                try:
                    frame_queue.get_nowait()
                    stats["replaced"] += 1
                except queue.Empty:
                    pass
                try:
                    frame_queue.put_nowait(item)
                except queue.Full:
                    stats["replaced"] += 1
        except Exception as exc:
            print("[CARLA camera callback]", type(exc).__name__, str(exc)[:200])

    def sender():
        with httpx.Client(timeout=12.0) as http:
            while not stop_event.is_set() or not frame_queue.empty():
                try:
                    frame_id, sim_timestamp, jpeg = frame_queue.get(timeout=0.4)
                except queue.Empty:
                    continue
                try:
                    response = http.post(
                        api + "/scene/live/frame",
                        data={
                            "session_id": session_id,
                            "frame_id": str(frame_id),
                            "sim_timestamp": str(sim_timestamp),
                        },
                        files={"file": ("carla_rgb.jpg", jpeg, "image/jpeg")},
                    )
                    response.raise_for_status()
                    result = response.json()
                    stats["sent"] += 1
                    if stats["sent"] == 1 or stats["sent"] % 5 == 0:
                        risk = result.get("risk", {}).get("risk_level", "unknown")
                        print(
                            f"[LIVE] frame={frame_id} "
                            f"detections={result.get('detection_count')} "
                            f"risk={risk} "
                            f"YOLO={result.get('inference_ms')}ms "
                            f"sent={stats['sent']} replaced={stats['replaced']}"
                        )
                except Exception as exc:
                    stats["failed"] += 1
                    print("[API POST ERROR]", type(exc).__name__, str(exc)[:350])

    worker = threading.Thread(target=sender, name="carla-frame-sender", daemon=True)
    worker.start()
    camera.listen(on_image)

    print("=== Phase 5.2.1 CARLA live RGB ===")
    print(f"CARLA: {args.host}:{args.port} | vehicle actor id: {vehicle.id}")
    print(f"Camera: {args.width}x{args.height} @ {args.fps:g} FPS")
    print(f"Session: {session_id}")
    print(f"Dashboard: {api}/scene/live/dashboard")
    print("No world.tick(), no synchronous-mode changes, no vehicle controls.")
    print("Press Ctrl+C to stop.")

    started = time.monotonic()
    try:
        while True:
            time.sleep(0.5)
            if args.duration_sec > 0 and time.monotonic() - started >= args.duration_sec:
                break
    except KeyboardInterrupt:
        print("\nStopping live RGB stream...")
    finally:
        stop_event.set()
        try:
            camera.stop()
        except Exception:
            pass
        try:
            camera.destroy()
        except Exception:
            pass
        worker.join(timeout=15.0)
        try:
            client.post(api + "/scene/live/stop", json={"session_id": session_id})
        except Exception:
            pass
        client.close()

    print(
        f"Stopped. sent={stats['sent']} failed={stats['failed']} "
        f"replaced_latest_frames={stats['replaced']}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
