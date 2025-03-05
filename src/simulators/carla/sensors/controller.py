import time
import carla
import weakref
import numpy as np
from queue import Queue, Empty
from typing import Dict, Optional, Tuple
from src.simulators.carla.sensors.builder import SensorBuilder, SensorBuilderSettings


class SensorController:
    def __init__(
            self,
            blueprint_library: carla.BlueprintLibrary,
            root_directory: str,
            sensor_builder_settings: SensorBuilderSettings = SensorBuilderSettings(),
            queue_timeout: float = 1.0
    ):
        self._queue_timeout = queue_timeout

        builder = SensorBuilder(
            blueprint_library=blueprint_library,
            root_directory=root_directory,
            sensor_builder_settings=sensor_builder_settings
        )
        self._sensor_dict = {sensor.name: sensor for sensor in builder.sensors}
        self._num_sensors = len(self._sensor_dict)
        self.queue = Queue()

    @staticmethod
    def _sensor_callback(weak_self, sensor_data: carla.SensorData, name: str):
        self = weak_self()

        if not self:
            return

        data = self._sensor_dict[name].preprocess(sensor_data=sensor_data)
        self.queue.put(item=(data, name, sensor_data.frame))

    def reset_queue(self):
        self.queue = Queue()

    def collect_sensor_data(self, save: bool, ignore_missing: bool) -> Tuple[Dict[str, np.ndarray], bool, Optional[int]]:
        collected_data = {}
        no_missing = True
        world_frame = None

        for i in range(self._num_sensors):
            try:
                data, name, frame = self.queue.get(block=True, timeout=self._queue_timeout)

                if i == 0:
                    world_frame = frame
                else:
                    if world_frame != frame:
                        no_missing = False

                collected_data[name] = data
            except Empty:
                no_missing = False
                break

        if save and (no_missing or ignore_missing):
            for name, data in collected_data.items():
                sensor = self._sensor_dict[name]

                if sensor.directory is not None:
                    sensor.save(data=data, frame=world_frame)

        return collected_data, no_missing, world_frame

    def spawn(self, world: carla.World, vehicle: carla.Vehicle):
        weak_self = weakref.ref(self)
        for sensor in self._sensor_dict.values():
            sensor.spawn(
                world=world,
                vehicle=vehicle,
                sensor_callback=lambda sensor_data, name: SensorController._sensor_callback(weak_self=weak_self, sensor_data=sensor_data, name=name)
            )
        time.sleep(0.5)
        world.tick()
        self.queue = Queue()

    def destroy(self, world: carla.World):
        for sensor in self._sensor_dict.values():
            sensor.destroy()
        time.sleep(0.5)
        world.tick()
        self.queue = Queue()
