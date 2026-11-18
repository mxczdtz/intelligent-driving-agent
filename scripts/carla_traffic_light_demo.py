"""Controlled CARLA 0.9.16 traffic-light change experiment (simulation only).

Run: python -m scripts.carla_traffic_light_demo
Produces matched RGB images and simulator state, compatible with
scripts.carla_analyze_capture. Signal labels are simulator state, not
verified image-visible labels or risk ground truth.
"""
import json
import math
import queue
from datetime import datetime
from pathlib import Path

import carla

STEP = 0.1
WIDTH, HEIGHT = 640, 480
EGO_SPEED = 2.5  # m/s
START_DISTANCE = 20.0
PER_PHASE = 10
SETTLE_TICKS = 3
COUNT = PER_PHASE * 3


def vec(v):
    return {"x": float(v.x), "y": float(v.y), "z": float(v.z)}


def angle_diff(a, b):
    return abs((a - b + 180) % 360 - 180)


def candidates(lights):
    """Rank straight approaches to actual traffic-light stop waypoints."""
    options = []
    for light in lights:
        for stop in light.get_stop_waypoints():
            if stop.lane_type != carla.LaneType.Driving:
                continue
            start_points = stop.previous(START_DISTANCE)
            for start in start_points:
                if start.is_junction or start.lane_id != stop.lane_id or start.road_id != stop.road_id:
                    continue
                heading = start.transform.rotation.yaw
                if angle_diff(heading, stop.transform.rotation.yaw) > 8:
                    continue
                valid = True
                for d in (5.0, 10.0, 15.0):
                    waypoints = start.next(d)
                    if len(waypoints) != 1:
                        valid = False
                        break
                    mid = waypoints[0]
                    if (mid.is_junction or mid.road_id != start.road_id
                            or mid.lane_id != start.lane_id
                            or angle_diff(mid.transform.rotation.yaw, heading) > 8):
                        valid = False
                        break
                if not valid:
                    continue
                origin = start.transform.location
                signal = light.get_location()
                yaw = math.radians(heading)
                fx, fy = math.cos(yaw), math.sin(yaw)
                dx, dy = signal.x - origin.x, signal.y - origin.y
                forward, lateral = dx * fx + dy * fy, -dx * fy + dy * fx
                # Pole origin is only a visibility proxy, not proof of visible lamps.
                if forward < 12.0 or forward > 60.0 or abs(lateral) > forward * 0.7:
                    continue
                transform = start.transform
                transform.location.z += 0.35
                score = abs(lateral) / forward + 0.01 * abs(forward - 25.0)
                options.append((score, light, stop, transform))
    options.sort(key=lambda item: item[0])
    return options


def main():
    root = Path(__file__).resolve().parents[1]
    output = root / "evidence" / "carla_traffic_light" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    client = carla.Client("127.0.0.1", 2000)
    client.set_timeout(30.0)
    world = client.get_world()
    original = world.get_settings()
    if original.synchronous_mode:
        raise RuntimeError("World already synchronous; restart CARLA if no other script is running")
    actors = world.get_actors()
    if list(actors.filter("vehicle.*")) or list(actors.filter("walker.pedestrian.*")):
        raise RuntimeError("Use an empty simulation world; stop other actor scripts or restart CARLA")
    lights = list(actors.filter("traffic.traffic_light"))
    if not lights:
        raise RuntimeError("No traffic lights found in current map")
    print("Map:", world.get_map().name, flush=True)
    print("Traffic light actors:", len(lights), flush=True)
    approaches = candidates(lights)
    print("Candidate approaches:", len(approaches), flush=True)
    if not approaches:
        raise RuntimeError("No suitable straight signal approach found; try Town01")
    output.mkdir(parents=True, exist_ok=False)
    print("Output:", output, flush=True)

    ego = camera = light = None
    changed = frozen_changed = False
    original_frozen = False
    previous_states = []
    images = queue.Queue()
    states, saved, records = {}, set(), []
    errors = []

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

        bp = world.get_blueprint_library().find("vehicle.tesla.model3")
        bp.set_attribute("role_name", "ego")
        for _, candidate, stop_wp, transform in approaches:
            ego = world.try_spawn_actor(bp, transform)
            if ego is not None:
                light = candidate
                break
        if ego is None or light is None:
            raise RuntimeError("Could not spawn vehicle at any candidate approach")
        ego.apply_control(carla.VehicleControl(brake=1.0, hand_brake=True))
        for _ in range(25):
            world.tick(30.0)

        original_frozen = bool(light.is_frozen())
        # Freeze is world-wide. Require a uniform original freeze flag.
        if any(bool(item.is_frozen()) != original_frozen for item in lights):
            raise RuntimeError("Mixed original traffic-light freeze flags; restart CARLA")
        frozen_changed = True
        world.freeze_all_traffic_lights(True)
        world.tick(30.0)
        group = {item.id: item for item in light.get_group_traffic_lights()}
        group[light.id] = light
        previous_states = [(item, item.get_state()) for item in group.values()]
        for item in group.values():
            if item.id != light.id:
                item.set_state(carla.TrafficLightState.Red)
        light.set_state(carla.TrafficLightState.Green)
        print("Ego:", ego.id, "Target traffic light:", light.id, flush=True)

        bp = world.get_blueprint_library().find("sensor.camera.rgb")
        for k, v in {"image_size_x": WIDTH, "image_size_y": HEIGHT,
                     "fov": 90, "sensor_tick": 0.0}.items():
            bp.set_attribute(k, str(v))
        camera = world.spawn_actor(bp, carla.Transform(carla.Location(x=1.6, z=1.7)),
                                   attach_to=ego, attachment_type=carla.AttachmentType.Rigid)
        camera.listen(images.put)
        for _ in range(5):
            world.tick(30.0)

        stop = stop_wp.transform.location
        yaw = math.radians(stop_wp.transform.rotation.yaw)
        fx, fy = math.cos(yaw), math.sin(yaw)

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

        def drain(timeout=0.0):
            if timeout:
                try:
                    accept(images.get(timeout=timeout))
                except queue.Empty:
                    pass
            while True:
                try:
                    image = images.get_nowait()
                except queue.Empty:
                    break
                accept(image)

        def advance():
            forward = ego.get_transform().get_forward_vector()
            ego.set_target_velocity(carla.Vector3D(x=forward.x * EGO_SPEED,
                                                   y=forward.y * EGO_SPEED, z=0.0))
            frame = world.tick(30.0)
            snap = world.get_snapshot()
            if snap.frame != frame:
                raise RuntimeError("Snapshot mismatch; only one client may tick")
            actor = snap.find(ego.id)
            if actor is None:
                raise RuntimeError("Missing ego snapshot")
            transform = actor.get_transform()
            loc = transform.location
            distance = (stop.x - loc.x) * fx + (stop.y - loc.y) * fy
            if distance < 5.0:
                raise RuntimeError("Ending test: ego approaching stop waypoint too closely")
            return snap, actor, distance

        ego.apply_control(carla.VehicleControl())
        phases = [("green", carla.TrafficLightState.Green),
                  ("yellow", carla.TrafficLightState.Yellow),
                  ("red", carla.TrafficLightState.Red)]
        step = 0
        for name, command in phases:
            light.set_state(command)
            # Allow rendering transition before recording this phase.
            for _ in range(SETTLE_TICKS):
                advance()
                drain()
            for _ in range(PER_PHASE):
                snap, actor, distance = advance()
                actual = light.get_state()
                if actual != command:
                    raise RuntimeError(f"Signal state mismatch at frame {snap.frame}: {actual}")
                transform = actor.get_transform()
                velocity = actor.get_velocity()
                state = {
                    "world_frame": int(snap.frame), "world_timestamp_s": float(snap.timestamp.elapsed_seconds),
                    "ego_vehicle_id": int(ego.id), "ego_speed_kmh": 3.6 * math.hypot(velocity.x, velocity.y),
                    "location_m": vec(transform.location), "velocity_mps": vec(velocity),
                    "rotation_deg": {"pitch": float(transform.rotation.pitch),
                        "yaw": float(transform.rotation.yaw), "roll": float(transform.rotation.roll)},
                    "scenario_step": step, "phase": name, "traffic_light_id": int(light.id),
                    "traffic_light_command": name, "traffic_light_state": name,
                    "stop_waypoint_longitudinal_distance_m": distance,
                    "stop_waypoint_location_m": vec(stop),
                    "stop_waypoint_road_id": int(stop_wp.road_id),
                    "stop_waypoint_lane_id": int(stop_wp.lane_id),
                }
                states[int(snap.frame)] = state
                drain(0.5)
                print(f"{name} | frame={snap.frame} | signal={name} "
                      f"| ego={state['ego_speed_kmh']:.2f} km/h | stop waypoint={distance:.2f} m", flush=True)
                step += 1

        ego.apply_control(carla.VehicleControl(brake=1.0))
        for _ in range(15):
            if len(records) == COUNT:
                break
            world.tick(30.0)
            drain(1.0)
        if len(records) != COUNT:
            raise RuntimeError(f"Only {len(records)}/{COUNT} matched frames")
        records.sort(key=lambda r: r["scenario_step"])
        actual_sequence = [r["traffic_light_state"] for r in records]
        expected_sequence = [name for name, _ in phases for _ in range(PER_PHASE)]
        if actual_sequence != expected_sequence:
            raise RuntimeError("Recorded signal sequence mismatch")
        report = {
            "carla_client_version": client.get_client_version(),
            "carla_server_version": client.get_server_version(),
            "map": world.get_map().name, "scenario": "traffic_light_green_yellow_red",
            "fixed_delta_seconds": STEP,
            "camera": {"type": "sensor.camera.rgb", "width": WIDTH, "height": HEIGHT,
                       "horizontal_fov_deg": 90, "relative_location_m": {"x": 1.6, "y": 0, "z": 1.7}},
            "matched_frame_count": len(records),
            "scenario_validation": {"target_traffic_light_id": int(light.id),
                "observed_sequence": ["green", "yellow", "red"],
                "frames_per_state": {name: PER_PHASE for name, _ in phases},
                "all_state_commands_verified": True,
                "min_stop_waypoint_distance_m": min(r["stop_waypoint_longitudinal_distance_m"] for r in records)},
            "notes": [
                "Externally commanded ego speed; no autonomous decisions.",
                "Signal state comes from simulator actor after each tick, not image classification.",
                "Stop waypoint from traffic-light trigger geometry, not necessarily a painted stop line.",
                "Pole origin used to rank approaches; image visibility and correct lamp association need visual review.",
                "Signal phase durations are scripted, not real traffic timing.",
                "No risk ground-truth labels. Signal cycle elapsed time cannot be restored exactly."
            ],
            "frames": records,
        }
        metadata = output / "metadata.json"
        metadata.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
        print("Metadata:", metadata, flush=True)
        for index in (0, PER_PHASE, 2 * PER_PHASE):
            print(records[index]["phase"], "image:", output / records[index]["image_file"], flush=True)
        print("Verified signal sequence: green -> yellow -> red", flush=True)

    finally:
        def cleanup(label, action):
            try:
                if action() is False:
                    raise RuntimeError("operation returned False")
            except Exception as exc:
                errors.append(f"{label}: {exc}")
        if camera is not None:
            cleanup("stop camera", camera.stop)
            cleanup("destroy camera", camera.destroy)
        if ego is not None:
            cleanup("destroy ego", ego.destroy)
        for item, state in previous_states:
            cleanup(f"restore signal {item.id}", lambda item=item, state=state: item.set_state(state))
        if frozen_changed:
            cleanup("restore traffic-light freeze flag",
                    lambda: world.freeze_all_traffic_lights(original_frozen))
        if changed:
            cleanup("restore world settings", lambda: world.apply_settings(original))
        if errors:
            for error in errors:
                print("CLEANUP WARNING:", error, flush=True)
        else:
            print("Cleanup complete; world settings and modified signal states restored.", flush=True)
    if errors:
        raise RuntimeError("Cleanup needs attention")
    print(f"PASS: CARLA traffic light change ({COUNT} matched frames)", flush=True)


if __name__ == "__main__":
    main()
