import carla
from src.simulators.carla.sensors import cameras
from src.simulators.carla.sensors import navigation


class SensorBuilderSettings:
    def __init__(self):
        self.use_rgb = True
        self.use_semantic_segmentation = True
        self.use_instance_segmentation = True
        self.use_depth = True
        self.use_gnss = True
        self.image_height = 300
        self.image_width = 400
        self.image_fov = 60
        self.peripheral_vision = True
        self.tick_interval = 0.0


class SensorBuilder:
    def __init__(
            self,
            blueprint_library: carla.BlueprintLibrary,
            root_directory: str,
            sensor_builder_settings: SensorBuilderSettings
    ):
        self.sensors = []

        tick_interval = sensor_builder_settings.tick_interval
        image_height = sensor_builder_settings.image_height
        image_width = sensor_builder_settings.image_width
        image_fov = sensor_builder_settings.image_fov

        if sensor_builder_settings.use_rgb:
            self.sensors.append(
                cameras.RGBCamera(
                    name='rgb_front',
                    root_directory=root_directory,
                    transform=carla.Transform(carla.Location(x=2.5, y=0, z=0.7)),
                    blueprint_library=blueprint_library,
                    tick_interval=tick_interval,
                    height=image_height,
                    width=image_width,
                    fov=image_fov,
                    enable_postprocess_effects=True
                )
            )
            if sensor_builder_settings.peripheral_vision:
                self.sensors.append(cameras.RGBCamera(
                    name='rgb_left',
                    root_directory=root_directory,
                    transform=carla.Transform(carla.Location(x=4.5, y=2.0, z=0.7), rotation=carla.Rotation(roll=0, pitch=0, yaw=-45)),
                    blueprint_library=blueprint_library,
                    tick_interval=tick_interval,
                    height=image_height,
                    width=image_width,
                    fov=image_fov,
                    enable_postprocess_effects=True
                ))
                self.sensors.append(
                    cameras.RGBCamera(
                        name='rgb_right',
                        root_directory=root_directory,
                        transform=carla.Transform(carla.Location(x=4.5, y=-2.0, z=0.7), rotation=carla.Rotation(roll=0, pitch=0, yaw=45)),
                        blueprint_library=blueprint_library,
                        tick_interval=tick_interval,
                        height=image_height,
                        width=image_width,
                        fov=image_fov,
                        enable_postprocess_effects=True
                    )
                )

        if sensor_builder_settings.use_semantic_segmentation:
            self.sensors.append(
                cameras.SemanticSegmentationCamera(
                    name='semantic_front',
                    root_directory=root_directory,
                    transform=carla.Transform(carla.Location(x=2.5, y=0, z=0.7)),
                    blueprint_library=blueprint_library,
                    tick_interval=tick_interval,
                    height=image_height,
                    width=image_width,
                    fov=image_fov
                )
            )
            if sensor_builder_settings.peripheral_vision:
                self.sensors.append(
                    cameras.SemanticSegmentationCamera(
                        name='semantic_left',
                        root_directory=root_directory,
                        transform=carla.Transform(carla.Location(x=4.5, y=2.0, z=0.7), rotation=carla.Rotation(roll=0, pitch=0, yaw=-45)),
                        blueprint_library=blueprint_library,
                        tick_interval=tick_interval,
                        height=image_height,
                        width=image_width,
                        fov=image_fov
                    )
                )
                self.sensors.append(
                    cameras.SemanticSegmentationCamera(
                        name='semantic_right',
                        root_directory=root_directory,
                        transform=carla.Transform(carla.Location(x=4.5, y=-2.0, z=0.7), rotation=carla.Rotation(roll=0, pitch=0, yaw=45)),
                        blueprint_library=blueprint_library,
                        tick_interval=tick_interval,
                        height=image_height,
                        width=image_width,
                        fov=image_fov
                    )
                )

        if sensor_builder_settings.use_instance_segmentation:
            self.sensors.append(
                cameras.InstanceSegmentationCamera(
                    name='instance_front',
                    root_directory=root_directory,
                    transform=carla.Transform(carla.Location(x=2.5, y=0, z=0.7)),
                    blueprint_library=blueprint_library,
                    tick_interval=tick_interval,
                    height=image_height,
                    width=image_width,
                    fov=image_fov
                )
            )
            if sensor_builder_settings.peripheral_vision:
                self.sensors.append(
                    cameras.InstanceSegmentationCamera(
                        name='instance_left',
                        root_directory=root_directory,
                        transform=carla.Transform(carla.Location(x=4.5, y=2.0, z=0.7), rotation=carla.Rotation(roll=0, pitch=0, yaw=-45)),
                        blueprint_library=blueprint_library,
                        tick_interval=tick_interval,
                        height=image_height,
                        width=image_width,
                        fov=image_fov
                    )
                )
                self.sensors.append(
                    cameras.InstanceSegmentationCamera(
                        name='instance_right',
                        root_directory=root_directory,
                        transform=carla.Transform(carla.Location(x=4.5, y=-2.0, z=0.7), rotation=carla.Rotation(roll=0, pitch=0, yaw=45)),
                        blueprint_library=blueprint_library,
                        tick_interval=tick_interval,
                        height=image_height,
                        width=image_width,
                        fov=image_fov
                    )
                )

        if sensor_builder_settings.use_depth:
            self.sensors.append(
                cameras.DepthCamera(
                    name='depth_front',
                    root_directory=root_directory,
                    transform=carla.Transform(carla.Location(x=2.5, y=0, z=0.7)),
                    blueprint_library=blueprint_library,
                    tick_interval=tick_interval,
                    height=image_height,
                    width=image_width,
                    fov=image_fov
                )
            )
            if sensor_builder_settings.peripheral_vision:
                self.sensors.append(
                    cameras.DepthCamera(
                        name='depth_left',
                        root_directory=root_directory,
                        transform=carla.Transform(carla.Location(x=4.5, y=2.0, z=0.7), rotation=carla.Rotation(roll=0, pitch=0, yaw=-45)),
                        blueprint_library=blueprint_library,
                        tick_interval=tick_interval,
                        height=image_height,
                        width=image_width,
                        fov=image_fov
                    )
                )
                self.sensors.append(
                    cameras.DepthCamera(
                        name='depth_right',
                        root_directory=root_directory,
                        transform=carla.Transform(carla.Location(x=4.5, y=-2.0, z=0.7), rotation=carla.Rotation(roll=0, pitch=0, yaw=45)),
                        blueprint_library=blueprint_library,
                        tick_interval=tick_interval,
                        height=image_height,
                        width=image_width,
                        fov=image_fov
                    )
                )

        if sensor_builder_settings.use_gnss:
            self.sensors.append(navigation.GNSS(blueprints_library=blueprint_library, tick_interval=tick_interval))
