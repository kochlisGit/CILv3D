import carla
from src.simulators.carla.sensors.cameras.camera import Camera


class DepthCamera(Camera):
    def __init__(
            self,
            name: str,
            root_directory: str,
            transform: carla.Transform,
            blueprint_library: carla.BlueprintLibrary,
            tick_interval: float = 0.0,
            fov: int = 60,
            height: int = 300,
            width: int = 400
    ):
        blueprint = blueprint_library.find('sensor.camera.depth')

        super().__init__(
            name=name,
            root_directory=root_directory,
            transform=transform,
            blueprint=blueprint,
            color_converter=carla.ColorConverter.LogarithmicDepth,
            tick_interval=tick_interval,
            fov=fov,
            height=height,
            width=width
        )
