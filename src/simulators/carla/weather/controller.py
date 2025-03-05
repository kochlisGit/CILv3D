import carla
from src.simulators.carla.weather.storm import Storm
from src.simulators.carla.weather.sun import Sun


class WeatherSettings:
    def __init__(self):
        self.speed_factor = 0.6
        self.update_sun = True,
        self.update_clouds = True,
        self.update_rain = True,
        self.update_wind = True,
        self.update_fog = True


class DynamicWeatherController:
    def __init__(
            self,
            initial_weather: carla.WeatherParameters,
            weather_settings: WeatherSettings = WeatherSettings()
    ):
        self._speed_factor = weather_settings.speed_factor
        self._weather = initial_weather
        self.weather_settings = weather_settings

        self._update_freq = 0.1/self._speed_factor
        self._sun = Sun(azimuth=self._weather.sun_azimuth_angle, altitude=self._weather.sun_altitude_angle)
        self._storm = Storm(precipitation=self._weather.precipitation)
        self._elapsed_time = 0.0

    def _tick(self, delta_seconds: float):
        self._sun.tick(delta_seconds)
        self._storm.tick(delta_seconds)

        if self.weather_settings.update_sun:
            self._weather.sun_azimuth_angle = self._sun.azimuth
            self._weather.sun_altitude_angle = self._sun.altitude

        if self.weather_settings.update_clouds:
            self._weather.cloudiness = self._storm.clouds

        if self.weather_settings.update_rain:
            self._weather.precipitation = self._storm.rain
            self._weather.precipitation_deposits = self._storm.puddles
            self._weather.wetness = self._storm.wetness

        if self.weather_settings.update_wind:
            self._weather.wind_intensity = self._storm.wind

        if self.weather_settings.update_fog:
            self._weather.fog_density = self._storm.fog

    def step(self, world: carla.World):
        self._elapsed_time += world.get_snapshot().timestamp.delta_seconds

        if self._elapsed_time > self._update_freq:
            self._tick(delta_seconds=self._speed_factor*self._elapsed_time)
            world.set_weather(self._weather)
            self._elapsed_time = 0.0

    def __str__(self):
        return '%s %s' % (self._sun, self._storm)
