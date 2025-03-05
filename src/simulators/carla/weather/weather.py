from abc import ABC, abstractmethod


class Weather(ABC):
    @abstractmethod
    def tick(self, delta_seconds: float):
        pass

    @abstractmethod
    def __str__(self):
        pass
