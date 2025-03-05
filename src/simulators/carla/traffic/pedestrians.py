import random
import time
import carla
from src.simulators.carla.traffic.npc import NPCController
from typing import List, Tuple


class PedestrianController(NPCController):
    def __init__(self, blueprints_library: carla.BlueprintLibrary, walkers_running_prob: float = 0.5):
        super().__init__(blueprints=blueprints_library.filter('walker.pedestrian.*'))

        self._walker_running_prob = walkers_running_prob

        self._walker_controller_blueprint = blueprints_library.find('controller.ai.walker')
        self._walker_ids_list = []
        self._walker_controller_ids_list = []
        self._walker_controller_list = []
        self._walker_speeds = []

    @staticmethod
    def _generate_spawn_points(world: carla.World, num_walkers: int) -> List[carla.Transform]:
        spawn_points = []
        for i in range(num_walkers):
            random_location = world.get_random_location_from_navigation()
            if random_location is not None:
                spawn_point = carla.Transform(location=random_location)
                spawn_points.append(spawn_point)
        return spawn_points

    def _generate_random_blueprints_and_speeds(self, k: int) -> Tuple[List[carla.ActorBlueprint], List[float]]:
        blueprints = random.choices(self.blueprints, k=k)
        walker_speeds = []

        for blueprint in blueprints:
            if blueprint.has_attribute('is_invincible'):
                blueprint.set_attribute('is_invincible', 'false')
            if blueprint.has_attribute('speed'):
                if random.random() > self._walker_running_prob:
                    walker_speeds.append(blueprint.get_attribute('speed').recommended_values[1])
                else:
                    walker_speeds.append(blueprint.get_attribute('speed').recommended_values[2])
            else:
                walker_speeds.append(0.0)
        return blueprints, walker_speeds

    def _adjust_walkers_behavior(self, world: carla.World):
        assert len(self.walker_speeds) == len(self.walker_controller_list) == len(self._walker_controller_ids_list), \
            (f'Expected speeds == walker_controllers == controller ids, got {len(self.walker_speeds)} speeds vs '
             f'{len(self.walker_controller_list)} controllers and {len(self._walker_controller_ids_list)} ids')

        world.set_pedestrians_cross_factor(percentage=1.0)
        for i, walker_controller in enumerate(self.walker_controller_list):
            walker_controller.start()
            walker_controller.go_to_location(world.get_random_location_from_navigation())
            walker_controller.set_max_speed(float(self.walker_speeds[i]))

    def _spawn_walkers(self, client: carla.Client, world: carla.World, num_walkers: int):
        SpawnActor = carla.command.SpawnActor

        spawn_points = self._generate_spawn_points(world=world, num_walkers=num_walkers)
        blueprints, walker_speeds = self._generate_random_blueprints_and_speeds(k=len(spawn_points))
        batch = [SpawnActor(blueprint=blueprint, transform=transform) for blueprint, transform in zip(blueprints, spawn_points)]
        results = client.apply_batch_sync(batch, do_tick=True)
        time.sleep(0.5)

        for i, result in enumerate(results):
            if not result.error:
                self._walker_ids_list.append(result.actor_id)
                self._walker_speeds.append(walker_speeds[i])

    def _spawn_controllers(self, client: carla.Client, world: carla.World):
        SpawnActor = carla.command.SpawnActor

        batch = [
            SpawnActor(blueprint=self._walker_controller_blueprint, transform=carla.Transform(), parent_id=walker_id)
            for walker_id in self._walker_ids_list
        ]
        results = client.apply_batch_sync(batch, do_tick=True)
        time.sleep(0.5)

        speeds = []
        for i, result in enumerate(results):
            if not result.error:
                self._walker_controller_ids_list.append(result.actor_id)
                speeds.append(self._walker_speeds[i])

        self.walker_controller_list = world.get_actors(actor_ids=self._walker_controller_ids_list)
        self.walker_speeds = speeds
        world.tick()

    def spawn(self, client: carla.Client, world: carla.World, population_size: int, **kwargs):
        self._spawn_walkers(client=client, world=world, num_walkers=population_size)
        self._spawn_controllers(client=client, world=world)
        self._adjust_walkers_behavior(world=world)
        time.sleep(0.5)
        world.tick()

    def destroy(self, client: carla.Client, world: carla.World, **kwargs):
        for walker_controller in self.walker_controller_list:
            walker_controller.stop()

        client.apply_batch_sync(
            commands=[
                carla.command.DestroyActor(actor_id)
                for actor_id in self._walker_controller_ids_list + self._walker_ids_list
            ],
            do_tick=True
        )
        time.sleep(0.5)
        world.tick()

        self._walker_ids_list = []
        self._walker_controller_ids_list = []
        self._walker_controller_list = []
        self._walker_speeds = []
