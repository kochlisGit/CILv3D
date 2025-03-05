import math
from src.simulators.carla.weather.weather import Weather


class Sun(Weather):
    def __init__(self, azimuth: float, altitude: float):
        self.azimuth = azimuth
        self.altitude = altitude
        self._t = 0.0

    def tick(self, delta_seconds: float):
        self._t += 0.008*delta_seconds
        self._t %= 2.0 * math.pi

        self.azimuth += 0.25*delta_seconds
        self.azimuth %= 360.0
        self.altitude = (70 * math.sin(self._t)) - 20

    def __str__(self):
        return 'Sun(alt: %.2f, azm: %.2f)' % (self.altitude, self.azimuth)
