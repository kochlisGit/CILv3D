import random
import carla
import numpy as np
from typing import Any, Dict, Optional
from src.simulators.carla.agents.controller import AgentController


class AutopilotSettings:
    def __init__(self):
        self.max_speed_limit_percentage = 10.0
        self.change_left_lane_percentage = 0.0
        self.change_right_lane_percentage = 0.0


class Autopilot(AgentController):
    def __init__(self, autopilot_settings: AutopilotSettings = AutopilotSettings()):
        super().__init__()

        self._autopilot_settings = autopilot_settings

    def spawn(
            self,
            world: carla.World,
            spawn_point: carla.Transform,
            blueprint: carla.ActorBlueprint,
            traffic_manager: carla.TrafficManager,
            spawn_spectator: bool
    ):
        super().spawn(
            world=world,
            spawn_point=spawn_point,
            blueprint=blueprint,
            traffic_manager=traffic_manager,
            spawn_spectator=spawn_spectator
        )

        self.vehicle.set_autopilot(True)
        world.tick()

        speed_limit_perc = random.uniform(a=0.0, b=self._autopilot_settings.max_speed_limit_percentage)
        traffic_manager.vehicle_percentage_speed_difference(actor=self.vehicle, percentage=speed_limit_perc)
        traffic_manager.update_vehicle_lights(actor=self.vehicle, do_update=True)
        traffic_manager.random_left_lanechange_percentage(actor=self.vehicle, percentage=self._autopilot_settings.change_left_lane_percentage)
        traffic_manager.random_right_lanechange_percentage(actor=self.vehicle, percentage=self._autopilot_settings.change_right_lane_percentage)
        world.tick()

    def destroy(self, world: carla.World):
        self.vehicle.set_autopilot(False)
        world.tick()

        super().destroy(world=world)

    def reset(self):
        return

    def compute_vehicle_control(self, sensor_dict: Dict[str, np.ndarray], state_dict: Dict[str, Any]) -> Optional[carla.VehicleControl]:
        return None
