import carla
import cv2
import numpy as np
from abc import ABC
from src.simulators.carla.sensors.sensor import Sensor


class Camera(Sensor, ABC):
    def __init__(
            self,
            name: str,
            root_directory: str,
            transform: carla.Transform,
            blueprint: carla.ActorBlueprint,
            color_converter: carla.ColorConverter,
            tick_interval: float = 0.0,
            fov: int = 60,
            height: int = 300,
            width: int = 400
    ):
        self._height = height
        self._width = width
        self._channels = 4
        self._color_converter = color_converter

        blueprint.set_attribute('fov', str(fov))
        blueprint.set_attribute('image_size_x', str(width)),
        blueprint.set_attribute('image_size_y', str(height)),

        super().__init__(
            name=name,
            root_directory=root_directory,
            transform=transform,
            blueprint=blueprint,
            tick_interval=tick_interval
        )

    def preprocess(self, sensor_data: carla.SensorData) -> np.ndarray:
        sensor_data.convert(color_converter=self._color_converter)
        image_array = np.reshape(a=sensor_data.raw_data, newshape=(self._height, self._width, 4)).astype(dtype=np.uint8)
        return image_array[:, :, :3]

    def save(self, data: np.ndarray, frame: int):
        filepath = f'{self.directory}/{frame}.jpg'
        cv2.imwrite(filename=filepath, img=data)
