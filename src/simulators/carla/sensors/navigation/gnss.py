import carla
import numpy as np
from src.simulators.carla.sensors.sensor import Sensor


class GNSS(Sensor):
    def __init__(
            self,
            blueprints_library: carla.BlueprintLibrary,
            tick_interval: float = 0.0,
            altitude_noise_std: float = 0.0,
            longitude_noise_std: float = 0.0,
            latitude_noise_std: float = 0.0,
            seed: int = 0
    ):
        blueprint = blueprints_library.find('sensor.other.gnss')
        blueprint.set_attribute('noise_alt_stddev', str(altitude_noise_std))
        blueprint.set_attribute('noise_lat_stddev', str(longitude_noise_std))
        blueprint.set_attribute('noise_lon_stddev', str(latitude_noise_std))
        blueprint.set_attribute('noise_seed', str(seed))
        super().__init__(
            name='gnss',
            root_directory=None,
            transform=carla.Transform(),
            blueprint=blueprint,
            tick_interval=tick_interval
        )

    def preprocess(self, sensor_data: carla.SensorData) -> np.ndarray:
        return np.float32([sensor_data.latitude, sensor_data.longitude, sensor_data.altitude])

    def save(self, data: np.ndarray, frame: int):
        return
