import random
import time
import carla
from src.simulators.carla.traffic.npc import NPCController
from typing import List, Optional


class VehicleController(NPCController):
    def __init__(
            self,
            blueprints_library: carla.BlueprintLibrary,
            spawn_points: List[carla.Transform],
            enable_lane_change: bool,
            safe_distance: float,
            max_speed_limit_difference_percentage: float,
            ignore_traffic_rules_percentage: float
    ):
        super().__init__(blueprints=blueprints_library.filter('vehicle.*'))

        self._spawn_points = spawn_points
        self._enable_lane_change = enable_lane_change
        self._safe_distance = safe_distance
        self._max_speed_limit_difference_percentage = max_speed_limit_difference_percentage
        self._ignore_traffic_rules_percentage = ignore_traffic_rules_percentage
        self._num_spawn_points = len(spawn_points)
        self.vehicle_ids_list = []

    def _generate_random_blueprints(self, num_vehicles: int) -> List[carla.ActorBlueprint]:
        blueprints = random.choices(population=self.blueprints, k=num_vehicles)
        for blueprint in blueprints:
            blueprint.set_attribute('role_name', 'autopilot')

            if blueprint.has_attribute('color'):
                color = random.choice(blueprint.get_attribute('color').recommended_values)
                blueprint.set_attribute('color', color)
            if blueprint.has_attribute('driver_id'):
                driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
                blueprint.set_attribute('driver_id', driver_id)
        return blueprints

    def _adjust_vehicles_behavior(self, vehicle_list: List[carla.Vehicle], traffic_manager: carla.TrafficManager):
        traffic_manager.set_global_distance_to_leading_vehicle(distance=self._safe_distance)
        for vehicle in vehicle_list:
            traffic_manager.vehicle_percentage_speed_difference(
                actor=vehicle,
                percentage=random.uniform(a=-self._max_speed_limit_difference_percentage, b=10.0)
            )
            traffic_manager.auto_lane_change(actor=vehicle, enable=self._enable_lane_change)
            traffic_manager.update_vehicle_lights(actor=vehicle, do_update=True)

            if self._ignore_traffic_rules_percentage > 0.0:
                traffic_manager.ignore_lights_percentage(actor=vehicle, perc=self._ignore_traffic_rules_percentage)
                traffic_manager.ignore_signs_percentage(actor=vehicle, perc=self._ignore_traffic_rules_percentage)

    def spawn(
            self,
            client: carla.Client,
            world: carla.World,
            population_size: int,
            traffic_manager: Optional[carla.TrafficManager] = None,
            **kwargs
    ):
        if traffic_manager is None:
            traffic_manager = client.get_trafficmanager()

        SpawnActor = carla.command.SpawnActor
        SetAutopilot = carla.command.SetAutopilot
        FutureActor = carla.command.FutureActor

        num_vehicles = min(population_size, self._num_spawn_points)
        blueprints = self._generate_random_blueprints(num_vehicles=num_vehicles)
        spawn_points = random.sample(self._spawn_points, k=num_vehicles)

        port = traffic_manager.get_port()
        batch = [
            SpawnActor(blueprint=blueprint, transform=transform).then(SetAutopilot(actor_id=FutureActor, enabled=True, tm_port=port))
            for blueprint, transform in zip(blueprints, spawn_points)
        ]
        results = client.apply_batch_sync(commands=batch, do_tick=True)

        self.vehicle_ids_list = [result.actor_id for result in results if not result.error]
        self._adjust_vehicles_behavior(
            vehicle_list=world.get_actors(actor_ids=self.vehicle_ids_list),
            traffic_manager=traffic_manager
        )
        time.sleep(0.5)
        world.tick()

    def destroy(self, client: carla.Client, world: carla.World, **kwargs):
        if len(self.vehicle_ids_list) > 0:
            client.apply_batch_sync(
                commands=[carla.command.DestroyActor(actor_id) for actor_id in self.vehicle_ids_list],
                do_tick=True
            )
            time.sleep(0.5)
            world.tick()

            self.vehicle_ids_list = []
