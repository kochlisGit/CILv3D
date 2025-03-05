from src.simulators.carla.weather.weather import Weather


class Storm(Weather):
    def __init__(self, precipitation: float):
        self._t = precipitation if precipitation > 0.0 else -50.0
        self._increasing = True
        self.clouds = 0.0
        self.rain = 0.0
        self.wetness = 0.0
        self.puddles = 0.0
        self.wind = 0.0
        self.fog = 0.0

    @staticmethod
    def _clamp(value: float, minimum: float = 0.0, maximum: float = 100.0):
        return max(minimum, min(value, maximum))

    def tick(self, delta_seconds: float):
        delta = (1.3 if self._increasing else -1.3)*delta_seconds
        delay = -10.0 if self._increasing else 90.0

        self._t = self._clamp(value=delta + self._t, minimum=-250.0, maximum=100.0)

        if self._t == -250.0:
            self._increasing = True
        if self._t == 100.0:
            self._increasing = False

        self.clouds = self._clamp(value=self._t + 40.0, minimum=0.0, maximum=90.0)
        self.rain = self._clamp(self._t, minimum=0.0, maximum=80.0)
        self.puddles = self._clamp(value=self._t + delay, minimum=0.0, maximum=85.0)
        self.wetness = self._clamp(value=self._t * 5, minimum=0.0, maximum=100.0)
        self.wind = 5.0 if self.clouds <= 20 else 90 if self.clouds >= 70 else 40
        self.fog = self._clamp(value=self._t - 10, minimum=0.0, maximum=30.0)

    def __str__(self):
        return 'Storm(clouds=%d%%, rain=%d%%, wind=%d%%)' % (self.clouds, self.rain, self.wind)
