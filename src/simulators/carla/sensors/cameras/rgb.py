import carla
from src.simulators.carla.sensors.cameras.camera import Camera


class RGBCamera(Camera):
    def __init__(
            self,
            name: str,
            root_directory: str,
            transform: carla.Transform,
            blueprint_library: carla.BlueprintLibrary,
            tick_interval: float = 0.0,
            fov: int = 60,
            height: int = 300,
            width: int = 400,
            enable_postprocess_effects: bool = True
    ):
        blueprint = blueprint_library.find('sensor.camera.rgb')
        blueprint.set_attribute('enable_postprocess_effects', f'{enable_postprocess_effects}')

        super().__init__(
            name=name,
            root_directory=root_directory,
            transform=transform,
            blueprint=blueprint,
            color_converter=carla.ColorConverter.Raw,
            tick_interval=tick_interval,
            fov=fov,
            height=height,
            width=width
        )
