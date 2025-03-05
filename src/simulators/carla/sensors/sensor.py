import os
import carla
import numpy as np
from abc import ABC, abstractmethod
from typing import Callable, Optional


class Sensor(ABC):
    def __init__(
            self,
            name: str,
            root_directory: Optional[str],
            transform: carla.Transform,
            blueprint: carla.ActorBlueprint,
            tick_interval: Optional[float] = 0.0
    ):
        self._name = name
        self._directory = None if root_directory is None else f'{root_directory}/sensors/{name}'
        self._transform = transform

        if tick_interval is not None:
            blueprint.set_attribute('sensor_tick', str(tick_interval))

        self._blueprint = blueprint

        if self._directory is not None:
            os.makedirs(name=self._directory, exist_ok=True)

        self.sensor = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def directory(self) -> str:
        return self._directory

    @abstractmethod
    def preprocess(self, sensor_data: carla.SensorData) -> np.ndarray:
        pass

    @abstractmethod
    def save(self, data: np.ndarray, frame: int):
        pass

    def spawn(self, world: carla.World, vehicle: carla.Vehicle, sensor_callback: Callable):
        self.sensor = world.spawn_actor(blueprint=self._blueprint, transform=self._transform, attach_to=vehicle)
        self.sensor.listen(lambda data: sensor_callback(sensor_data=data, name=self._name))

    def destroy(self):
        if self.sensor is not None:
            self.sensor.destroy()
            self.sensor = None
