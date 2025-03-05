import carla
from abc import ABC, abstractmethod
from typing import List


class NPCController(ABC):
    def __init__(self, blueprints: List[carla.ActorBlueprint]):
        self._blueprints = blueprints

    @property
    def blueprints(self) -> List[carla.ActorBlueprint]:
        return self._blueprints

    @abstractmethod
    def spawn(self, client: carla.Client, world: carla.World, population_size: int, **kwargs):
        pass

    @abstractmethod
    def destroy(self, client: carla.Client, world: carla.World, **kwargs):
        pass
