
import carla
import time


def main():
    # 1. 连接 CARLA 仿真服务器
    client = carla.Client("127.0.0.1", 2000)
    client.set_timeout(20.0)

    world = client.get_world()
    print("Current map:", world.get_map().name)

    # 2. 获取车辆模型库
    blueprint_library = world.get_blueprint_library()
    vehicle_bp = blueprint_library.find(
        "vehicle.tesla.model3"
    )
    vehicle_bp.set_attribute("role_name", "ego")

    # 3. 获取地图上的车辆生成点
    spawn_points = world.get_map().get_spawn_points()
    print("Available spawn points:", len(spawn_points))

    vehicle = None
    spectator = world.get_spectator()
    original_view = spectator.get_transform()

    try:
        # 4. 寻找可以生成车辆的位置
        for spawn_point in spawn_points:
            vehicle = world.try_spawn_actor(
                vehicle_bp, spawn_point
            )

            if vehicle is not None:
                break

        if vehicle is None:
            raise RuntimeError("Failed to spawn vehicle")

        print("Vehicle spawned successfully")
        print("Vehicle ID:", vehicle.id)
        print("Vehicle type:", vehicle.type_id)

        # 保持车辆静止
        vehicle.apply_control(
            carla.VehicleControl(
                brake=1.0,
                hand_brake=True
            )
        )

        time.sleep(1)

        # 5. 获取车辆位置和朝向
        transform = vehicle.get_transform()
        location = transform.location

        print("Vehicle location:", location)

        forward = transform.get_forward_vector()

        # 6. 设置车辆后上方的观察视角
        spectator.set_transform(
            carla.Transform(
                location + carla.Location(
                    x=-8 * forward.x,
                    y=-8 * forward.y,
                    z=5
                ),
                carla.Rotation(
                    pitch=-25,
                    yaw=transform.rotation.yaw
                )
            )
        )

        print("Spectator camera updated")
        print("PASS: CARLA vehicle spawn")

        input("Press Enter to remove the vehicle...")

    finally:
        # 7. 清理生成的车辆，恢复原观察视角
        if vehicle is not None:
            vehicle.destroy()
            print("Vehicle destroyed")

        spectator.set_transform(original_view)


if __name__ == "__main__":
    main()