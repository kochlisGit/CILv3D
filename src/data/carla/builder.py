import logging
import random
import pandas as pd
from tqdm import tqdm
from src.simulators.carla.scenarios.runner import RandomScenarioRunner
from src.simulators.carla import tools
from src.simulators.carla.agents.controller import AgentController
from src.simulators.carla.weather.controller import WeatherSettings
from src.simulators.carla.sensors.controller import SensorBuilderSettings
from src.simulators.carla.simulation.manager import SimulationSettings
from src.simulators.carla.traffic.controller import TrafficSettings


class CarlaDatasetBuilder:
    def __init__(
            self,
            root_directory: str,
            town_name: str,
            vehicle_model: str,
            vehicle_color: str,
            num_episodes: int,
            steps_per_episode: int,
            skips_per_step: int,
            spectate: bool,
            enable_dynamic_weather_after_episode: int,
            simulation_settings: SimulationSettings,
            weather_settings: WeatherSettings,
            sensor_builder_settings: SensorBuilderSettings,
            traffic_settings: TrafficSettings,
            agent: AgentController
    ):
        self._root_directory = root_directory
        self._town_name = town_name
        self._vehicle_model = vehicle_model
        self._vehicle_color = vehicle_color
        self._num_episodes = num_episodes
        self._steps_per_episode = steps_per_episode
        self._skips_per_step = skips_per_step
        self._enable_dynamic_weather_after_episode = enable_dynamic_weather_after_episode
        self._spectate = spectate
        self._simulation_settings = simulation_settings
        self._weather_settings = weather_settings
        self._sensor_builder_settings = sensor_builder_settings
        self._traffic_settings = traffic_settings
        self._agent = agent

        self._lon1 = self._lon2 = self._lon3 = 0.0
        self._lat1 = self._lat2 = self._lat3 = 0.0
        self._invaded_lane = False
        self._invaded_lane_road_id = None
        self._invaded_lane_command = None
        self._direction = None
        self._command = None

    def update_gps_location(self, longitude: float, latitude: float):
        self._lon1 = self._lon2
        self._lon2 = self._lon3
        self._lon3 = longitude
        self._lat1 = self._lat2
        self._lat2 = self._lat3
        self._lat3 = latitude

    def set_gps_location(self, longitude: float, latitude: float):
        self._lon1 = self._lon2 = self._lon3 = longitude
        self._lat1 = self._lat2 = self._lat3 = latitude

    def build_dataset(self):
        runner = RandomScenarioRunner(
            vehicle_model=self._vehicle_model,
            vehicle_color=self._vehicle_color,
            spectate=self._spectate,
            simulation_settings=self._simulation_settings,
            weather_settings=self._weather_settings,
            sensor_builder_settings=self._sensor_builder_settings,
            traffic_settings=self._traffic_settings,
            agent=self._agent
        )
        result = runner.start(town=self._town_name)

        logging.info(f'Result: {result}\n')

        if result:
            for episode in tqdm(iterable=range(self._num_episodes), desc='Episode'):
                episode_directory = f'{self._root_directory}/{self._town_name}/{episode}'
                enable_dynamic_weather = episode >= self._enable_dynamic_weather_after_episode
                self._lon1 = self._lon2 = self._lon3 = 0.0
                self._lat1 = self._lat2 = self._lat3 = 0.0
                self._invaded_lane = False
                self._invaded_lane_road_id = None
                self._invaded_lane_command = None
                self._direction = None
                self._command = None

                random.seed(episode)
                sensor_dict, state_dict, no_missing_data, waypoint = runner.restart(
                    root_directory=episode_directory,
                    seed=episode
                )
                self._direction = tools.RoadOptions.STRAIGHT
                self._command = tools.RoadOptions.LANE_FOLLOW

                with tqdm(total=self._steps_per_episode, desc='Step') as episode_progressbar:
                    states = []
                    step = 0
                    collected_steps = 0
                    no_missing_data = True

                    while collected_steps < self._steps_per_episode:
                        save_data = step % self._skips_per_step == 0
                        update_weather = enable_dynamic_weather and no_missing_data

                        state_dict['direction'] = self._direction
                        state_dict['command'] = self._command
                        control = self._agent.compute_vehicle_control(sensor_dict=sensor_dict, state_dict=state_dict)
                        sensor_dict, state_dict, no_missing_data, waypoint = runner.step(
                            control=control,
                            update_vehicle_control=save_data,
                            update_weather=update_weather,
                            save_sensor_data=save_data,
                            ignore_missing_sensor_data=False
                        )

                        if state_dict['invaded_lane']:
                            self._invaded_lane = True

                        if step == 0:
                            self.set_gps_location(longitude=state_dict['longitude'], latitude=state_dict['latitude'])

                        if no_missing_data:
                            step += 1

                            if save_data:
                                self.update_gps_location(longitude=state_dict['longitude'], latitude=state_dict['latitude'])

                                self._direction = tools.determine_direction(
                                    lon1=self._lon1,
                                    lon2=self._lon2,
                                    lon3=self._lon3,
                                    lat1=self._lat1,
                                    lat2=self._lat2,
                                    lat3=self._lat3,
                                    straight_line_threshold=3.0
                                )
                                self._command = tools.construct_autopilot_command(
                                    waypoint=waypoint,
                                    has_invaded_lane=self._invaded_lane,
                                    direction=self._direction
                                )
                                road_id = state_dict['road_id']

                                if self._invaded_lane:
                                    if self._invaded_lane_road_id == road_id:
                                        self._command = self._invaded_lane_command
                                    else:
                                        self._invaded_lane_road_id = road_id
                                        self._invaded_lane_command = self._command

                                state_dict['direction'] = self._direction
                                state_dict['command'] = self._command
                                state_dict['invaded_lane'] = self._invaded_lane
                                self._invaded_lane = False

                                states.append(state_dict)
                                collected_steps += 1
                                episode_progressbar.update(1)
                        else:
                            self.set_gps_location(longitude=state_dict['longitude'], latitude=state_dict['latitude'])
                            self._invaded_lane = False

                pd.DataFrame(states).to_csv(f'{episode_directory}/states.csv', index=False)
                runner.terminate()

            runner.shutdown()
