import logging
from src.data.carla.builder import CarlaDatasetBuilder
from src.simulators.carla.agents.autopilot import Autopilot, AutopilotSettings
from src.simulators.carla.weather.controller import WeatherSettings
from src.simulators.carla.sensors.controller import SensorBuilderSettings
from src.simulators.carla.simulation.manager import SimulationSettings
from src.simulators.carla.traffic.controller import TrafficSettings

root_directory = 'storage/datasets/carla'
town_name = 'Town10HD'
vehicle_model = 'vehicle.tesla.model3'
vehicle_color = '255,255,255'
num_episodes = 10
steps_per_episode = 1000
skips_per_step = 10
enable_dynamic_weather_after_episode = 3
sensor_tick_interval = 0.0
simulation_settings = SimulationSettings()
weather_settings = WeatherSettings()
autopilot_settings = AutopilotSettings()
sensor_builder_settings = SensorBuilderSettings()
traffic_settings = TrafficSettings()
spectate = False


def main():
    if spectate:
        simulation_settings.render_graphics = True

    agent = Autopilot(autopilot_settings=autopilot_settings)
    builder = CarlaDatasetBuilder(
        root_directory=root_directory,
        town_name=town_name,
        vehicle_model=vehicle_model,
        vehicle_color=vehicle_color,
        num_episodes=num_episodes,
        steps_per_episode=steps_per_episode,
        skips_per_step=skips_per_step,
        enable_dynamic_weather_after_episode=enable_dynamic_weather_after_episode,
        simulation_settings=simulation_settings,
        weather_settings=weather_settings,
        sensor_builder_settings=sensor_builder_settings,
        traffic_settings=traffic_settings,
        spectate=spectate,
        agent=agent
    )
    builder.build_dataset()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    main()
