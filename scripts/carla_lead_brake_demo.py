"""CARLA 0.9.16 lead-vehicle braking scenario, simulation only.

Run from the project root: python -m scripts.carla_lead_brake_demo
Produces RGB frames + matched state metadata compatible with carla_analyze_capture.
The ego speed is externally set for a controlled experiment, not autonomous driving.
"""
import json
import math
import queue
from datetime import datetime
from pathlib import Path
import carla

STEP = 0.1
WIDTH, HEIGHT = 640, 480
INITIAL_SPEED = 7.0  # m/s
GAP = 18.0  # nominal initial center spacing in meters
PRE_TICKS, BRAKE_TICKS = 10, 15
COUNT = PRE_TICKS + BRAKE_TICKS


def vec(v):
    return {"x": float(v.x), "y": float(v.y), "z": float(v.z)}


def planar_speed(v):
    return math.hypot(v.x, v.y)


def yaw_diff(a, b):
    return abs((a - b + 180) % 360 - 180)


def find_pairs(roadmap):
    # Select a straight, non-junction lane with enough room for the test.
    for start in roadmap.get_spawn_points():
        wp = roadmap.get_waypoint(start.location, project_to_road=True,
                                  lane_type=carla.LaneType.Driving)
        if wp is None or wp.is_junction:
            continue
        chosen = {}
        for distance in (6.0, 12.0, GAP, 24.0, 30.0, 36.0):
            candidates = wp.next(distance)
            if len(candidates) != 1:
                break
            nxt = candidates[0]
            if (nxt.is_junction or nxt.road_id != wp.road_id
                    or nxt.lane_id != wp.lane_id
                    or yaw_diff(nxt.transform.rotation.yaw, wp.transform.rotation.yaw) > 5):
                break
            chosen[distance] = nxt
        if 36.0 not in chosen:
            continue
        lead = chosen[GAP].transform
        lead.location.z += 0.35
        yield start, lead


def velocity_along(vehicle, speed):
    forward = vehicle.get_transform().get_forward_vector()
    return carla.Vector3D(x=forward.x * speed, y=forward.y * speed, z=0.0)


def state_for(snapshot, ego, lead, phase, index):
    es, ls = snapshot.find(ego.id), snapshot.find(lead.id)
    if es is None or ls is None:
        raise RuntimeError("Missing vehicle snapshot")
    et, lt = es.get_transform(), ls.get_transform()
    ev, lv = es.get_velocity(), ls.get_velocity()
    # Project planar relative position and velocity onto ego heading.
    yaw = math.radians(et.rotation.yaw)
    fx, fy = math.cos(yaw), math.sin(yaw)
    dx, dy = lt.location.x - et.location.x, lt.location.y - et.location.y
    longitudinal = dx * fx + dy * fy
    lateral = -dx * fy + dy * fx
    closing = (ev.x - lv.x) * fx + (ev.y - lv.y) * fy
    return {
        "world_frame": int(snapshot.frame),
        "world_timestamp_s": float(snapshot.timestamp.elapsed_seconds),
        "ego_vehicle_id": int(ego.id),
        "ego_speed_kmh": 3.6 * planar_speed(ev),
        "location_m": vec(et.location),
        "velocity_mps": vec(ev),
        "rotation_deg": {"pitch": float(et.rotation.pitch),
                         "yaw": float(et.rotation.yaw), "roll": float(et.rotation.roll)},
        "scenario_step": index,
        "phase": phase,
        "lead_brake_command": 0.8 if phase == "lead_braking" else 0.0,
        "lead_vehicle_id": int(lead.id),
        "lead_speed_kmh": 3.6 * planar_speed(lv),
        "lead_location_m": vec(lt.location),
        "lead_velocity_mps": vec(lv),
        "longitudinal_origin_separation_m": longitudinal,
        "lateral_origin_offset_m": lateral,
        "closing_speed_mps": closing,
    }


def main():
    root = Path(__file__).resolve().parents[1]
    output = root / "evidence" / "carla_lead_brake" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=False)
    client = carla.Client("127.0.0.1", 2000)
    client.set_timeout(30.0)
    world = client.get_world()
    original = world.get_settings()
    if original.synchronous_mode:
        raise RuntimeError("World already synchronous; stop other tick clients or restart CARLA")
    if list(world.get_actors().filter("vehicle.*")) or list(world.get_actors().filter("walker.pedestrian.*")):
        raise RuntimeError("Use an empty simulation world; close other vehicle scripts or restart CARLA")
    roadmap = world.get_map()
    print("Map:", roadmap.name, flush=True)
    print("Output:", output, flush=True)
    ego = lead = camera = None
    changed = False
    images = queue.Queue()
    states, records, saved = {}, [], set()
    cleanup_errors = []

    try:
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = STEP
        settings.no_rendering_mode = False
        settings.substepping = True
        settings.max_substep_delta_time = 0.01
        settings.max_substeps = 10
        changed = True
        world.apply_settings(settings)

        library = world.get_blueprint_library()
        eb = library.find("vehicle.tesla.model3")
        eb.set_attribute("role_name", "ego")
        lb = library.find("vehicle.audi.a2")
        lb.set_attribute("role_name", "lead")
        for start, ahead in find_pairs(roadmap):
            ego = world.try_spawn_actor(eb, start)
            if ego is None:
                continue
            lead = world.try_spawn_actor(lb, ahead)
            if lead is not None:
                break
            if ego.destroy() is False:
                raise RuntimeError("Failed to remove unsuccessful spawn")
            ego = None
            world.tick(30.0)
        if ego is None or lead is None:
            raise RuntimeError("No suitable straight-lane spawn pair; try Town01")

        print("Ego:", ego.id, "Lead:", lead.id, flush=True)
        for vehicle in (ego, lead):
            vehicle.apply_control(carla.VehicleControl(brake=1.0, hand_brake=True))
        for _ in range(25):
            world.tick(30.0)

        bp = library.find("sensor.camera.rgb")
        for key, value in {"image_size_x": WIDTH, "image_size_y": HEIGHT,
                           "fov": 90, "sensor_tick": 0.0}.items():
            bp.set_attribute(key, str(value))
        camera = world.spawn_actor(bp, carla.Transform(carla.Location(x=1.6, z=1.7)),
                                   attach_to=ego, attachment_type=carla.AttachmentType.Rigid)
        camera.listen(images.put)
        for _ in range(5):
            world.tick(30.0)

        def accept(image):
            frame = int(image.frame)
            if frame not in states or frame in saved:
                return
            state = states[frame]
            dt = abs(float(image.timestamp) - state["world_timestamp_s"])
            if dt > 0.001 or (image.width, image.height) != (WIDTH, HEIGHT):
                raise RuntimeError(f"Image/state mismatch: {frame}")
            name = f"frame_{frame:08d}.png"
            dest = output / name
            image.save_to_disk(str(dest), carla.ColorConverter.Raw)
            if not dest.is_file() or not dest.stat().st_size:
                raise RuntimeError(f"Image save failed: {frame}")
            records.append({**state, "image_frame": frame,
                            "image_timestamp_s": float(image.timestamp),
                            "timestamp_difference_s": dt, "image_file": name,
                            "image_width": WIDTH, "image_height": HEIGHT})
            saved.add(frame)

        def drain(wait=0.0):
            if wait:
                try:
                    accept(images.get(timeout=wait))
                except queue.Empty:
                    pass
            while True:
                try:
                    image = images.get_nowait()
                except queue.Empty:
                    break
                accept(image)

        # Use externally commanded velocity to simplify ego motion;
        # after the pre-phase the lead car brakes using CARLA vehicle physics.
        neutral = carla.VehicleControl(throttle=0.0, brake=0.0, hand_brake=False)
        ego.apply_control(neutral)
        lead.apply_control(neutral)
        for index in range(COUNT):
            ego.set_target_velocity(velocity_along(ego, INITIAL_SPEED))
            if index < PRE_TICKS:
                phase = "before_braking"
                lead.set_target_velocity(velocity_along(lead, INITIAL_SPEED))
            else:
                phase = "lead_braking"
                lead.apply_control(carla.VehicleControl(brake=0.8))
            frame = world.tick(30.0)
            snap = world.get_snapshot()
            if frame != snap.frame:
                raise RuntimeError("Snapshot mismatch: only one client may tick")
            state = state_for(snap, ego, lead, phase, index)
            states[int(frame)] = state
            drain(0.5)
            print(f"{phase} | frame={frame} | ego={state['ego_speed_kmh']:.2f} km/h "
                  f"| lead={state['lead_speed_kmh']:.2f} km/h "
                  f"| separation={state['longitudinal_origin_separation_m']:.2f} m",
                  flush=True)
            if state["longitudinal_origin_separation_m"] < 7.0:
                raise RuntimeError("Scenario stopped: vehicles too close for this collection demo")

        # Advance a few extra ticks to flush delayed GPU images, without
        # adding them to the experiment. Both vehicles now brake.
        for vehicle in (ego, lead):
            vehicle.apply_control(carla.VehicleControl(brake=1.0))
        for _ in range(15):
            if len(records) == COUNT:
                break
            world.tick(30.0)
            drain(1.0)
        drain(1.0)
        if len(records) != COUNT:
            raise RuntimeError(f"Only {len(records)}/{COUNT} matched frames")

        records.sort(key=lambda r: r["scenario_step"])
        pre_speed = records[PRE_TICKS - 1]["lead_speed_kmh"]
        end_speed = records[-1]["lead_speed_kmh"]
        reduction = pre_speed - end_speed
        observed = reduction >= 5.0
        report = {
            "carla_client_version": client.get_client_version(),
            "carla_server_version": client.get_server_version(),
            "map": roadmap.name,
            "scenario": "lead_vehicle_braking",
            "fixed_delta_seconds": STEP,
            "camera": {"type": "sensor.camera.rgb", "width": WIDTH, "height": HEIGHT,
                       "horizontal_fov_deg": 90,
                       "relative_location_m": {"x": 1.6, "y": 0.0, "z": 1.7}},
            "matched_frame_count": len(records),
            "scenario_validation": {
                "lead_speed_before_brake_kmh": pre_speed,
                "lead_speed_at_end_kmh": end_speed,
                "lead_speed_reduction_kmh": reduction,
                "observed_reduction_at_least_5_kmh": observed,
            },
            "notes": [
                "Ego speed is externally commanded; not autonomous control.",
                "Position separation is between actor origins projected on ego heading, not bumper clearance.",
                "Horizontal speed from simulator snapshots; not image-derived.",
                "Brake phases describe commands; measured speed verifies the event.",
                "No independent risk labels or risk accuracy evaluation yet."
            ],
            "frames": records,
        }
        metadata = output / "metadata.json"
        metadata.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
        print("Metadata:", metadata, flush=True)
        print("First image:", output / records[0]["image_file"], flush=True)
        print(f"Lead speed reduction: {reduction:.2f} km/h", flush=True)
        if not observed:
            raise RuntimeError("Captured frames but lead speed reduction below 5 km/h; inspect the report")

    finally:
        def cleanup(label, action):
            try:
                if action() is False:
                    raise RuntimeError("operation returned False")
            except Exception as exc:
                cleanup_errors.append(f"{label}: {exc}")
        if camera is not None:
            cleanup("stop camera", camera.stop)
            cleanup("destroy camera", camera.destroy)
        if lead is not None:
            cleanup("destroy lead", lead.destroy)
        if ego is not None:
            cleanup("destroy ego", ego.destroy)
        if changed:
            cleanup("restore settings", lambda: world.apply_settings(original))
        if cleanup_errors:
            for error in cleanup_errors:
                print("CLEANUP WARNING:", error, flush=True)
        else:
            print("Cleanup complete; original world settings restored.", flush=True)

    if cleanup_errors:
        raise RuntimeError("Cleanup needs attention")
    print(f"PASS: CARLA lead braking ({COUNT} matched frames)", flush=True)


if __name__ == "__main__":
    main()
