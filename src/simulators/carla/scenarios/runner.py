import logging
import random
import time
import weakref
import carla
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from src.simulators.carla.agents.controller import AgentController
from src.simulators.carla.weather.controller import DynamicWeatherController, WeatherSettings
from src.simulators.carla.sensors.controller import SensorController, SensorBuilderSettings
from src.simulators.carla.simulation.manager import SimulationManager, SimulationSettings
from src.simulators.carla.traffic.controller import TrafficController, TrafficSettings
from src.simulators.carla import tools


class RandomScenarioRunner:
    def __init__(
            self,
            vehicle_model: str,
            vehicle_color: str,
            simulation_settings: SimulationSettings(),
            weather_settings: WeatherSettings(),
            traffic_settings: TrafficSettings(),
            sensor_builder_settings: SensorBuilderSettings(),
            agent: AgentController,
            spectate: bool
    ):
        if spectate and not simulation_settings.render_graphics:
            spectate = False

            logging.info('Cannot use spectator mode without rendering.')

        self._vehicle_model = vehicle_model
        self._vehicle_color = vehicle_color

        self._simulation_settings = simulation_settings
        self._weather_settings = weather_settings
        self._traffic_settings = traffic_settings
        self._sensor_builder_settings = sensor_builder_settings
        self._agent = agent
        self._spectate = spectate

        self._client = None
        self._world = None
        self._map = None

        self._simulation_manager = None
        self._weather_controller = None
        self._traffic_controller = None
        self._sensor_controller = None

        self._lane_invasion_sensor = None
        self._invaded_lane = False

    @staticmethod
    def _lane_invasion_callback(weak_self):
        self = weak_self()

        if not self:
            return

        self._invaded_lane = True

    def _adjust_world_settings(self, sync: bool, delta_seconds: Optional[float]):
        settings = self._world.get_settings()
        settings.synchronous_mode = sync
        settings.fixed_delta_seconds = delta_seconds
        self._world.apply_settings(settings)

    def _initialize_weather(self):
        self._weather_controller = DynamicWeatherController(
            initial_weather=self._world.get_weather(),
            weather_settings=self._weather_settings
        )

    def _spawn_agent(
            self,
            blueprint_library: carla.BlueprintLibrary,
            spawn_point: carla.Transform,
            traffic_manager: carla.TrafficManager
    ):
        blueprint = blueprint_library.find(self._vehicle_model)
        blueprint.set_attribute('color', str(self._vehicle_color))
        self._agent.spawn(
            world=self._world,
            spawn_point=spawn_point,
            blueprint=blueprint,
            traffic_manager=traffic_manager,
            spawn_spectator=self._spectate
        )

    def _spawn_sensors(self, blueprint_library: carla.BlueprintLibrary, vehicle: carla.Vehicle, root_directory: str):
        lane_invasion_bp = blueprint_library.find('sensor.other.lane_invasion')
        self._lane_invasion_sensor = self._world.spawn_actor(
            blueprint=lane_invasion_bp,
            transform=carla.Transform(),
            attach_to=vehicle
        )
        self._lane_invasion_sensor.listen(lambda _: RandomScenarioRunner._lane_invasion_callback(weakref.ref(self)))
        self._sensor_controller = SensorController(
            blueprint_library=blueprint_library,
            root_directory=root_directory,
            sensor_builder_settings=self._sensor_builder_settings
        )
        self._sensor_controller.spawn(world=self._world, vehicle=vehicle)

    def _spawn_npcs(
            self,
            blueprint_library: carla.BlueprintLibrary,
            traffic_manager: carla.TrafficManager,
            vehicle_spawn_points: List[carla.Transform],
            seed: Optional[int] = None
    ):
        self._traffic_controller = TrafficController(
            world=self._world,
            blueprint_library=blueprint_library,
            traffic_manager=traffic_manager,
            vehicle_spawn_points=vehicle_spawn_points,
            synchronous_mode=self._simulation_settings.synchronous_mode,
            seed=seed
        )
        self._traffic_controller.spawn(client=self._client, world=self._world)

    def get_vehicle_state(
            self,
            vehicle: carla.Vehicle,
            sensor_dict: Dict[str, np.ndarray],
            frame: int
    ) -> Tuple[Dict[str, Any], carla.Waypoint]:
        controls = vehicle.get_control()
        throttle = controls.throttle
        steer = controls.steer
        brake = controls.brake
        speed = tools.get_vehicle_speed(velocity=vehicle.get_velocity())
        location = vehicle.get_location()
        waypoint = self._map.get_waypoint(location=location)
        next_location = waypoint.next(5.0)[0].transform.location
        latitude, longitude, altitude = sensor_dict.get('gnss', [0.0, 0.0, 0.0])

        state_dict = {
            'frame': frame,
            'throttle': throttle,
            'steer': steer,
            'brake': brake,
            'speed': speed,
            'x1': location.x,
            'y1': location.y,
            'z1': location.z,
            'x2': next_location.x,
            'y2': next_location.y,
            'z2': next_location.z,
            'latitude': latitude,
            'longitude': longitude,
            'altitude': altitude,
            'invaded_lane': self._invaded_lane,
            'road_id': waypoint.road_id,
            'lane_id': waypoint.lane_id
        }
        return state_dict, waypoint

    def start(self, town: str) -> bool:
        self._simulation_manager = SimulationManager(simulation_settings=self._simulation_settings)

        if self._simulation_manager.start():
            time.sleep(self._simulation_settings.initialization_time)

            try:
                self._client = carla.Client(host=self._simulation_settings.host, port=self._simulation_settings.port)
                self._client.set_timeout(self._simulation_settings.client_timeout)

                if town != 'Town10HD':
                    logging.info(f'Loading Town: {town}')

                    self._client.load_world(map_name=town, reset_settings=True)

                self._world = self._client.get_world()
                return True
            except Exception as e:
                logging.error(e)

                self._simulation_manager.shutdown()
                return False
        else:
            return False

    def restart(
            self,
            root_directory: str,
            seed: Optional[int] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any], bool, carla.Waypoint]:
        self._map = self._world.get_map()

        self._adjust_world_settings(
            sync=self._simulation_settings.synchronous_mode,
            delta_seconds=self._simulation_settings.delta_seconds
        )
        self._initialize_weather()

        traffic_manager = self._client.get_trafficmanager()
        traffic_manager.set_synchronous_mode(self._simulation_settings.synchronous_mode)

        if seed:
            traffic_manager.set_random_device_seed(seed)

        blueprint_library = self._world.get_blueprint_library()
        spawn_points = self._map.get_spawn_points()
        random_index = random.randint(a=0, b=len(spawn_points) - 1)
        spawn_point = spawn_points.pop(random_index)

        self._spawn_agent(blueprint_library=blueprint_library, spawn_point=spawn_point, traffic_manager=traffic_manager)
        self._spawn_sensors(blueprint_library=blueprint_library, vehicle=self._agent.vehicle, root_directory=root_directory)
        self._spawn_npcs(blueprint_library=blueprint_library, traffic_manager=traffic_manager, vehicle_spawn_points=spawn_points, seed=seed)
        return self.tick(update_weather=False, save_sensor_data=False, ignore_missing_sensor_data=False)

    def tick(
            self,
            update_weather: bool,
            save_sensor_data: bool,
            ignore_missing_sensor_data: bool
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any], bool, carla.Waypoint]:
        self._sensor_controller.reset_queue()
        self._invaded_lane = False

        if update_weather:
            self._weather_controller.step(world=self._world)

        self._world.tick()
        try:
            sensor_dict, no_missing_data, frame = self._sensor_controller.collect_sensor_data(
                save=save_sensor_data,
                ignore_missing=ignore_missing_sensor_data,
            )
        except Exception as e:
            logging.info(e)

            self.terminate()
            exit(-1)
        else:
            vehicle_state, waypoint = self.get_vehicle_state(vehicle=self._agent.vehicle, frame=frame, sensor_dict=sensor_dict)
            return sensor_dict, vehicle_state, no_missing_data, waypoint

    def step(
            self,
            control: Optional[carla.VehicleControl],
            update_vehicle_control: bool,
            update_weather: bool,
            save_sensor_data: bool,
            ignore_missing_sensor_data: bool
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any], bool, carla.Waypoint]:
        if update_vehicle_control:
            self._agent.step(control=control)

        return self.tick(update_weather=update_weather, save_sensor_data=save_sensor_data, ignore_missing_sensor_data=ignore_missing_sensor_data)

    def terminate(self):
        # Destroy all NPC actors
        self._traffic_controller.destroy(client=self._client, world=self._world)

        # Destroy Sensors
        self._lane_invasion_sensor.destroy()
        self._sensor_controller.destroy(world=self._world)

        # Destroy agent's vehicle
        try:
            self._agent.destroy(world=self._world)
        except Exception as e:
            logging.info(e)

        self._world.tick()
        self._adjust_world_settings(sync=False, delta_seconds=None)

    def shutdown(self):
        self._simulation_manager.shutdown()

        self._client = None
        self._world = None
        self._map = None
