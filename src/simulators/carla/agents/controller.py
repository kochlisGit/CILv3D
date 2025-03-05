import time
import carla
import numpy as np
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class AgentController(ABC):
    def __init__(self):
        self.vehicle = None
        self.spectator = None

    def spawn(
            self,
            world: carla.World,
            spawn_point: carla.Transform,
            blueprint: carla.ActorBlueprint,
            traffic_manager: carla.TrafficManager,
            spawn_spectator: bool
    ):
        if self.vehicle is not None:
            self.destroy()

        self.vehicle = world.spawn_actor(blueprint=blueprint, transform=spawn_point)

        if spawn_spectator:
            self.spectator = world.get_spectator()

        world.tick()
        time.sleep(0.5)

        if self.spectator is not None:
            self.spectator.set_transform(carla.Transform(spawn_point.location + carla.Location(z=30),carla.Rotation(pitch=-90)))

        self.reset()

    def destroy(self, world: carla.World):
        if self.vehicle is not None:
            self.vehicle.destroy()
            self.vehicle = None

        if self.spectator is not None:
            self.spectator.destroy()
            self.spectator = None

        time.sleep(0.5)
        world.tick()

    def get_vehicle_control(self) -> carla.VehicleControl:
        return self.vehicle.get_control()

    def step(self, control: Optional[carla.VehicleControl]):
        if control is not None:
            self.vehicle.apply_control(control=control)

        if self.spectator is not None:
            transform = carla.Transform(self.vehicle.get_transform().location + carla.Location(z=50),carla.Rotation(pitch=-90))
            self.spectator.set_transform(transform)

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def compute_vehicle_control(self, sensor_dict: Dict[str, np.ndarray], state_dict: Dict[str, Any]) -> Optional[carla.VehicleControl]:
        pass

