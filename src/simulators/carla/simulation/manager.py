import logging
import platform
import subprocess
from typing import Optional


class SimulationSettings:
    host = '127.0.0.1'
    port = 2000
    quality_level = 'Epic'
    render_graphics = False
    initialization_time = 15.0
    client_timeout = 15.0
    server_timeout = 10000
    window_width = 1200
    window_height = 800
    carla_path = 'C:\\Users\\kochlis\\Documents\\Research\\CarlaUE4\\CarlaUE4.exe'
    synchronous_mode = True
    delta_seconds = 0.05
    town_maps = [
        'Town01',
        'Town02',
        'Town03',
        'Town04',
        'Town05',
        'Town06',
        'Town07',
        'Town10HD'
    ]


class SimulationManager:
    def __init__(self, simulation_settings: Optional[SimulationSettings] = SimulationSettings()):
        self._host = simulation_settings.host
        self._port = simulation_settings.port

        self._flags = [
            simulation_settings.carla_path,
            '-carla-server',
            f'-carla-host={simulation_settings.host}',
            f'-carla-port={simulation_settings.port}',
            f'-quality-level={simulation_settings.quality_level}',
            f'-ResX={simulation_settings.window_width}',
            f'-ResY={simulation_settings.window_height}'
            f'-carla-server-timeout={simulation_settings.server_timeout}ms'
        ]

        if not simulation_settings.render_graphics:
            self._flags.append('-RenderOffScreen')

        self._process = None

    def start(self) -> bool:
        try:
            self._process = subprocess.Popen(self._flags, shell=True)
        except Exception as e:
            logging.error(e)

            success = False
        else:
            logging.info(
                f'Opening new CARLA process: {self._process.pid} in '
                f'{self._host}:{self._port}'
            )

            success = True
        return success

    def _terminate_windows_process(self) -> bool:
        try:
            subprocess.run(['taskkill', '/F', '/T', '/PID', str(self._process.pid)], check=True)
            return True
        except subprocess.CalledProcessError as e:
            logging.error('Error terminating process:', e)

            return False

    def _terminate_linux_process(self) -> bool:
        try:
            subprocess.run(['pkill', '-P', str(self._process.pid)], check=True)
            subprocess.run(['kill', str(self._process.pid)], check=True)
            return True
        except subprocess.CalledProcessError as e:
            logging.error('Error terminating process:', e)

            return False

    def shutdown(self) -> bool:
        system_platform = platform.system()

        if system_platform == 'Windows':
            result = self._terminate_windows_process()
        elif system_platform == 'Linux':
            result = self._terminate_linux_process()
        else:
            logging.warning(f'Detected Unknown System Platform: {system_platform}')

            result = self._terminate_windows_process() or self._terminate_linux_process()

        self._process = None
        return result
