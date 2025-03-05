import carla
from typing import List, Optional
from src.simulators.carla.traffic.vehicles import VehicleController
from src.simulators.carla.traffic.pedestrians import PedestrianController


class TrafficSettings:
    def __init__(self):
        self.num_vehicles = 'auto'
        self.num_pedestrians = 40
        self.enable_lane_change = True
        self.safe_distance = 4.0
        self.max_speed_limit_percentage = 30.0
        self.ignore_traffic_rules_percentage = 5.0
        self.walkers_running_prob = 0.5


class TrafficController:
    def __init__(
            self,
            world: carla.World,
            blueprint_library: carla.BlueprintLibrary,
            traffic_manager: carla.TrafficManager,
            vehicle_spawn_points: Optional[List[carla.Transform]],
            synchronous_mode: bool,
            traffic_settings: TrafficSettings = TrafficSettings(),
            seed: Optional[int] = None
    ):
        self._traffic_manager = traffic_manager
        self._synchronous_mode = synchronous_mode

        num_vehicles = traffic_settings.num_vehicles
        self._num_vehicles = int(len(vehicle_spawn_points)*0.35) if num_vehicles == 'auto' else num_vehicles
        self._num_pedestrians = traffic_settings.num_pedestrians

        if seed:
            world.set_pedestrians_seed(seed=seed)

        traffic_manager.set_synchronous_mode(synchronous_mode)

        self._vehicle_controller = None if num_vehicles == 0 else VehicleController(
            blueprints_library=blueprint_library,
            spawn_points=vehicle_spawn_points,
            enable_lane_change=traffic_settings.enable_lane_change,
            safe_distance=traffic_settings.safe_distance,
            max_speed_limit_difference_percentage=traffic_settings.max_speed_limit_percentage,
            ignore_traffic_rules_percentage=traffic_settings.ignore_traffic_rules_percentage
        )
        self._pedestrian_controller = None if traffic_settings.num_pedestrians == 0 else PedestrianController(
            blueprints_library=blueprint_library,
            walkers_running_prob=traffic_settings.walkers_running_prob
        )

    def spawn(self, client: carla.Client, world: carla.World):
        self._traffic_manager.set_synchronous_mode(self._synchronous_mode)

        if self._vehicle_controller is not None:
            self._vehicle_controller.spawn(
                client=client,
                world=world,
                population_size=self._num_vehicles,
                traffic_manager=self._traffic_manager
            )

        if self._pedestrian_controller is not None:
            self._pedestrian_controller.spawn(client=client, world=world, population_size=self._num_pedestrians)

    def destroy(self, client: carla.Client, world: carla.World):
        if self._vehicle_controller is not None:
            self._vehicle_controller.destroy(client=client, world=world)

        if self._pedestrian_controller is not None:
            self._pedestrian_controller.destroy(client=client, world=world)

        if self._synchronous_mode:
            self._traffic_manager.set_synchronous_mode(False)
