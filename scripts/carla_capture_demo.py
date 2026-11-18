"""Capture five matched CARLA RGB frames and ego states (simulation only)."""
import json
import math
import queue
from datetime import datetime
from pathlib import Path

import carla

WIDTH, HEIGHT, FOV = 640, 480, 90
STEP, COUNT = 0.1, 5


def vec(v):
    return {"x": float(v.x), "y": float(v.y), "z": float(v.z)}


def read_state(snapshot, actor_id):
    actor = snapshot.find(actor_id)
    if actor is None:
        raise RuntimeError("Ego vehicle missing from snapshot")
    v = actor.get_velocity()
    t = actor.get_transform()
    return {
        "world_frame": int(snapshot.frame),
        "world_timestamp_s": float(snapshot.timestamp.elapsed_seconds),
        "ego_vehicle_id": int(actor_id),
        "ego_speed_kmh": 3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2),
        "velocity_mps": vec(v),
        "location_m": vec(t.location),
        "rotation_deg": {
            "pitch": float(t.rotation.pitch),
            "yaw": float(t.rotation.yaw),
            "roll": float(t.rotation.roll),
        },
    }


def main():
    client = carla.Client("127.0.0.1", 2000)
    client.set_timeout(30.0)
    world = client.get_world()
    original = world.get_settings()
    if original.synchronous_mode:
        raise RuntimeError("World already synchronous. Stop other tick clients or restart CARLA.")
    root = Path(__file__).resolve().parents[1]
    output = root / "evidence" / "carla_capture_demo" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=False)
    map_name = world.get_map().name
    print("Map:", map_name, flush=True)
    print("Output:", output, flush=True)

    vehicle = camera = None
    changed = False
    images = queue.Queue()
    states = {}
    records = []
    saved = set()
    cleanup_errors = []

    try:
        # Synchronous simulation with a fixed timestep.
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = STEP
        settings.no_rendering_mode = False
        changed = True
        world.apply_settings(settings)

        library = world.get_blueprint_library()
        bp = library.find("vehicle.tesla.model3")
        bp.set_attribute("role_name", "ego")
        for point in world.get_map().get_spawn_points():
            vehicle = world.try_spawn_actor(bp, point)
            if vehicle is not None:
                break
        if vehicle is None:
            raise RuntimeError("No free vehicle spawn point")
        vehicle.apply_control(carla.VehicleControl(brake=1.0, hand_brake=True))
        print("Ego vehicle spawned:", vehicle.id, flush=True)

        # Rigidly attach a forward-facing camera to the vehicle.
        bp = library.find("sensor.camera.rgb")
        for key, value in {
            "image_size_x": WIDTH, "image_size_y": HEIGHT,
            "fov": FOV, "sensor_tick": 0.0,
        }.items():
            bp.set_attribute(key, str(value))
        camera = world.spawn_actor(
            bp, carla.Transform(carla.Location(x=1.6, z=1.7)),
            attach_to=vehicle, attachment_type=carla.AttachmentType.Rigid,
        )
        camera.listen(images.put)
        print("RGB camera attached:", camera.id, flush=True)

        def accept(image):
            frame = int(image.frame)
            if frame not in states or frame in saved or len(records) >= COUNT:
                return
            state = states[frame]
            dt = abs(float(image.timestamp) - state["world_timestamp_s"])
            if dt > 0.001:
                raise RuntimeError(f"Timestamp mismatch: frame {frame}, delta={dt}")
            if (image.width, image.height) != (WIDTH, HEIGHT):
                raise RuntimeError("Unexpected image resolution")
            name = f"frame_{frame:08d}.png"
            destination = output / name
            image.save_to_disk(str(destination), carla.ColorConverter.Raw)
            if not destination.is_file() or destination.stat().st_size == 0:
                raise RuntimeError("Image save failed")
            records.append({
                **state, "image_frame": frame,
                "image_timestamp_s": float(image.timestamp),
                "timestamp_difference_s": dt, "image_file": name,
                "image_width": int(image.width), "image_height": int(image.height),
            })
            saved.add(frame)
            print(f"Matched frame {frame} | speed={state['ego_speed_kmh']:.3f} km/h | {name}", flush=True)

        # Match delayed camera output with stored vehicle state by frame ID.
        # Never pair an old image with the latest vehicle state.
        for index in range(60):
            frame = world.tick(30.0)
            snapshot = world.get_snapshot()
            if snapshot.frame != frame:
                raise RuntimeError("Snapshot mismatch; only one client should tick the world")
            if index >= 3:
                states[int(frame)] = read_state(snapshot, vehicle.id)
            try:
                accept(images.get(timeout=1.0))
            except queue.Empty:
                pass
            while len(records) < COUNT:
                try:
                    image = images.get_nowait()
                except queue.Empty:
                    break
                accept(image)
            if len(records) == COUNT:
                break
        if len(records) != COUNT:
            raise RuntimeError(f"Only {len(records)}/{COUNT} matched frames received")

        records.sort(key=lambda r: r["image_frame"])
        report = {
            "carla_client_version": client.get_client_version(),
            "carla_server_version": client.get_server_version(),
            "map": map_name,
            "scenario": "stationary_ego_sensor_integration",
            "fixed_delta_seconds": STEP,
            "camera": {
                "type": "sensor.camera.rgb",
                "width": WIDTH, "height": HEIGHT, "horizontal_fov_deg": FOV,
                "relative_location_m": {"x": 1.6, "y": 0.0, "z": 1.7},
            },
            "matched_frame_count": len(records),
            "frames": records,
        }
        metadata = output / "metadata.json"
        metadata.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
        print("Metadata:", metadata, flush=True)
        print("First image:", output / records[0]["image_file"], flush=True)

    finally:
        # Independently attempt every cleanup operation, even after an error.
        def cleanup(label, action):
            try:
                if action() is False:
                    raise RuntimeError("operation returned False")
            except Exception as exc:
                cleanup_errors.append(f"{label}: {exc}")

        if camera is not None:
            cleanup("stop camera", camera.stop)
            cleanup("destroy camera", camera.destroy)
        if vehicle is not None:
            cleanup("destroy vehicle", vehicle.destroy)
        if changed:
            cleanup("restore settings", lambda: world.apply_settings(original))
        if cleanup_errors:
            for error in cleanup_errors:
                print("CLEANUP WARNING:", error, flush=True)
        else:
            print("Cleanup complete; original world settings restored.", flush=True)

    if cleanup_errors:
        raise RuntimeError("Cleanup needs attention")
    print(f"PASS: CARLA RGB + vehicle state ({COUNT} matched frames)", flush=True)


if __name__ == "__main__":
    main()
