"""Controlled CARLA pedestrian crossing, offline simulation only.

Run from project root:
    python -m scripts.carla_pedestrian_demo

The ego vehicle follows a slow externally commanded speed, while a scripted
walker crosses its lane. Captures are compatible with carla_analyze_capture.
No autonomous driving, no risk ground-truth labels.
"""
import json
import math
import queue
from datetime import datetime
from pathlib import Path

import carla

STEP = 0.1
WIDTH, HEIGHT = 640, 480
EGO_SPEED = 3.0  # m/s
WALK_SPEED = 1.6  # m/s
WAIT_TICKS, CROSS_TICKS = 8, 40
COUNT = WAIT_TICKS + CROSS_TICKS
CROSS_DISTANCE = 25.0


def vec(v):
    return {"x": float(v.x), "y": float(v.y), "z": float(v.z)}


def angle_difference(a, b):
    return abs((a - b + 180) % 360 - 180)


def candidates(roadmap):
    """Find a straight non-junction road segment and right lane-edge start."""
    for point in roadmap.get_spawn_points():
        wp = roadmap.get_waypoint(point.location, project_to_road=True,
                                  lane_type=carla.LaneType.Driving)
        if wp is None or wp.is_junction:
            continue
        selected = {}
        for d in (5.0, 10.0, 15.0, 20.0, CROSS_DISTANCE, 30.0):
            next_wps = wp.next(d)
            if len(next_wps) != 1:
                break
            nxt = next_wps[0]
            if (nxt.is_junction or nxt.road_id != wp.road_id
                    or nxt.lane_id != wp.lane_id
                    or angle_difference(nxt.transform.rotation.yaw,
                                        wp.transform.rotation.yaw) > 5.0):
                break
            selected[d] = nxt
        if 30.0 not in selected:
            continue
        cross = selected[CROSS_DISTANCE]
        if not 2.5 <= cross.lane_width <= 4.5:
            continue
        t = cross.transform
        yaw = math.radians(t.rotation.yaw)
        fx, fy = math.cos(yaw), math.sin(yaw)
        rx, ry = -fy, fx
        half = cross.lane_width / 2.0
        # Start on the right side of the selected lane, not necessarily a sidewalk.
        offset = half + 0.35
        loc = carla.Location(x=t.location.x + rx * offset,
                             y=t.location.y + ry * offset,
                             z=t.location.z + 0.4)
        # Avoid spawning significantly above/below a neighboring mapped surface.
        neighbor = roadmap.get_waypoint(loc, project_to_road=True,
              lane_type=carla.LaneType.Driving | carla.LaneType.Sidewalk |
                        carla.LaneType.Shoulder | carla.LaneType.Parking)
        if neighbor is None:
            continue
        nl = neighbor.transform.location
        if math.hypot(nl.x - loc.x, nl.y - loc.y) > 3.0 or abs(nl.z - t.location.z) > 0.6:
            continue
        loc.z = max(t.location.z, nl.z) + 0.35
        crossing_yaw = math.degrees(math.atan2(-ry, -rx))
        yield point, carla.Transform(loc, carla.Rotation(yaw=crossing_yaw)), cross, (fx, fy, rx, ry)


def main():
    root = Path(__file__).resolve().parents[1]
    output = root / "evidence" / "carla_pedestrian" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    client = carla.Client("127.0.0.1", 2000)
    client.set_timeout(30.0)
    world = client.get_world()
    original = world.get_settings()
    if original.synchronous_mode:
        raise RuntimeError("World already synchronous; stop other clients or restart CARLA")
    actors = world.get_actors()
    if list(actors.filter("vehicle.*")) or list(actors.filter("walker.pedestrian.*")):
        raise RuntimeError("Use an empty simulation world; stop other scripts or restart CARLA")
    roadmap = world.get_map()
    output.mkdir(parents=True, exist_ok=False)
    print("Map:", roadmap.name, flush=True)
    print("Output:", output, flush=True)

    ego = walker = camera = None
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
        walker_bps = sorted(library.filter("walker.pedestrian.*"), key=lambda b: b.id)
        if not walker_bps:
            raise RuntimeError("No pedestrian blueprints")
        wb = walker_bps[0]
        for start, walker_start, cross_wp, axes in candidates(roadmap):
            ego = world.try_spawn_actor(eb, start)
            if ego is None:
                continue
            walker = world.try_spawn_actor(wb, walker_start)
            if walker is not None:
                break
            if ego.destroy() is False:
                raise RuntimeError("Failed to clean up unavailable spawn")
            ego = None
            world.tick(30.0)
        if ego is None or walker is None:
            raise RuntimeError("No suitable lane-edge spawn pair. Try Town01.")

        fx, fy, rx, ry = axes
        center = cross_wp.transform.location
        half = cross_wp.lane_width / 2.0
        direction = carla.Vector3D(x=-rx, y=-ry, z=0.0)

        def walk(speed):
            walker.apply_control(carla.WalkerControl(direction=direction, speed=speed, jump=False))

        ego.apply_control(carla.VehicleControl(brake=1.0, hand_brake=True))
        walk(0.0)
        for _ in range(25):
            world.tick(30.0)
        print("Ego:", ego.id, "Walker:", walker.id, "Lane width:", cross_wp.lane_width, flush=True)

        bp = library.find("sensor.camera.rgb")
        for k, v in {"image_size_x": WIDTH, "image_size_y": HEIGHT,
                     "fov": 90, "sensor_tick": 0.0}.items():
            bp.set_attribute(k, str(v))
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
            delta = abs(float(image.timestamp) - state["world_timestamp_s"])
            if delta > 0.001 or (image.width, image.height) != (WIDTH, HEIGHT):
                raise RuntimeError(f"Image/state mismatch: {frame}")
            name = f"frame_{frame:08d}.png"
            dest = output / name
            image.save_to_disk(str(dest), carla.ColorConverter.Raw)
            if not dest.is_file() or not dest.stat().st_size:
                raise RuntimeError(f"Image save failed: {frame}")
            records.append({**state, "image_frame": frame,
                            "image_timestamp_s": float(image.timestamp),
                            "timestamp_difference_s": delta, "image_file": name,
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

        ego.apply_control(carla.VehicleControl())
        for index in range(COUNT):
            phase = "waiting" if index < WAIT_TICKS else "crossing_command"
            walk(0.0 if index < WAIT_TICKS else WALK_SPEED)
            forward = ego.get_transform().get_forward_vector()
            ego.set_target_velocity(carla.Vector3D(x=forward.x * EGO_SPEED,
                                                   y=forward.y * EGO_SPEED, z=0.0))
            frame = world.tick(30.0)
            snap = world.get_snapshot()
            if snap.frame != frame:
                raise RuntimeError("Snapshot mismatch; only one tick-driving client allowed")
            es, ws = snap.find(ego.id), snap.find(walker.id)
            if es is None or ws is None:
                raise RuntimeError("Missing actor state")
            et, wt = es.get_transform(), ws.get_transform()
            ev, wv = es.get_velocity(), ws.get_velocity()
            wx, wy = wt.location.x - center.x, wt.location.y - center.y
            offset = wx * rx + wy * ry
            dx, dy = wt.location.x - et.location.x, wt.location.y - et.location.y
            longitudinal, lateral = dx * fx + dy * fy, dx * rx + dy * ry
            state = {
                "world_frame": int(frame),
                "world_timestamp_s": float(snap.timestamp.elapsed_seconds),
                "ego_vehicle_id": int(ego.id),
                "ego_speed_kmh": 3.6 * math.hypot(ev.x, ev.y),
                "location_m": vec(et.location),
                "velocity_mps": vec(ev),
                "rotation_deg": {"pitch": float(et.rotation.pitch),
                                 "yaw": float(et.rotation.yaw), "roll": float(et.rotation.roll)},
                "scenario_step": index, "phase": phase,
                "walker_id": int(walker.id), "walker_location_m": vec(wt.location),
                "walker_velocity_mps": vec(wv),
                "walker_speed_mps": math.hypot(wv.x, wv.y),
                "walker_lane_center_offset_m": offset,
                "walker_inside_lane_lateral_band": abs(offset) <= half,
                "walker_longitudinal_offset_from_crossing_m": wx * fx + wy * fy,
                "walker_relative_longitudinal_m": longitudinal,
                "walker_relative_lateral_m": lateral,
                "walker_planar_origin_distance_m": math.hypot(dx, dy),
            }
            states[int(frame)] = state
            drain(0.5)
            print(f"{phase} | frame={frame} | ego={state['ego_speed_kmh']:.2f} km/h "
                  f"| walker={state['walker_speed_mps']:.2f} m/s "
                  f"| lane offset={offset:.2f} m | forward={longitudinal:.2f} m", flush=True)
            if longitudinal < 6.0:
                raise RuntimeError("Scenario stopped: ego too close to crossing point")

        ego.apply_control(carla.VehicleControl(brake=1.0))
        walk(0.0)
        for _ in range(15):
            if len(records) == COUNT:
                break
            world.tick(30.0)
            drain(1.0)
        if len(records) != COUNT:
            raise RuntimeError(f"Only {len(records)}/{COUNT} matched frames")

        records.sort(key=lambda r: r["scenario_step"])
        offsets = [r["walker_lane_center_offset_m"] for r in records]
        # Verify measured motion across the lane, not only the commands.
        crossed = (offsets[0] >= half and offsets[-1] <= -half
                   and any(abs(v) <= half for v in offsets)
                   and max(abs(r["walker_longitudinal_offset_from_crossing_m"]) for r in records) <= 2.0)
        report = {
            "carla_client_version": client.get_client_version(),
            "carla_server_version": client.get_server_version(),
            "map": roadmap.name, "scenario": "pedestrian_lane_crossing",
            "fixed_delta_seconds": STEP,
            "camera": {"type": "sensor.camera.rgb", "width": WIDTH, "height": HEIGHT,
                       "horizontal_fov_deg": 90, "relative_location_m": {"x": 1.6, "y": 0, "z": 1.7}},
            "matched_frame_count": len(records),
            "scenario_validation": {"lane_width_m": cross_wp.lane_width,
                "start_lateral_offset_m": offsets[0], "end_lateral_offset_m": offsets[-1],
                "observed_full_lane_crossing": crossed},
            "notes": ["Scripted walking and externally commanded ego velocity, not autonomous control.",
                      "Lane crossing verified against a local straight-lane reference, not a marked crosswalk.",
                      "Distances are actor-origin geometry, not physical clearance.",
                      "No independent risk labels yet."],
            "frames": records,
        }
        metadata = output / "metadata.json"
        metadata.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
        print("Metadata:", metadata, flush=True)
        print("First image:", output / records[0]["image_file"], flush=True)
        print(f"Lane offset: {offsets[0]:.2f} -> {offsets[-1]:.2f} m", flush=True)
        if not crossed:
            raise RuntimeError("Capture saved, but measured full lane crossing not verified; inspect metadata.")

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
        if walker is not None:
            cleanup("destroy walker", walker.destroy)
        if ego is not None:
            cleanup("destroy ego", ego.destroy)
        if changed:
            cleanup("restore settings", lambda: world.apply_settings(original))
        if cleanup_errors:
            for err in cleanup_errors:
                print("CLEANUP WARNING:", err, flush=True)
        else:
            print("Cleanup complete; original world settings restored.", flush=True)
    if cleanup_errors:
        raise RuntimeError("Cleanup needs attention")
    print(f"PASS: CARLA pedestrian crossing ({COUNT} matched frames)", flush=True)


if __name__ == "__main__":
    main()
