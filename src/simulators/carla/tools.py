import math
import carla
import numpy as np
import pandas as pd
from enum import Enum
from typing import Union


class RoadOptions(Enum):
    VOID = 'RoadOptions.VOID'
    CHANGE_LANE_LEFT = 'RoadOptions.CHANGE_LANE_LEFT'
    CHANGE_LANE_RIGHT = 'RoadOptions.CHANGE_LANE_RIGHT'
    LANE_FOLLOW = 'RoadOptions.LANE_FOLLOW'
    LEFT = 'RoadOptions.LEFT'
    RIGHT = 'RoadOptions.RIGHT'
    STRAIGHT = 'RoadOptions.STRAIGHT'


def to_radians(degrees: Union[float, np.ndarray, pd.Series]):
    return degrees * np.pi / 180


def to_degrees(radians: Union[float, np.ndarray, pd.Series]):
    return radians * 180 / np.pi


def get_vehicle_speed(velocity: carla.Transform) -> float:
    return math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)*3.6


def compute_bearing(
        lon1: Union[float, np.ndarray, pd.Series],
        lat1: Union[float, np.ndarray, pd.Series],
        lon2: Union[float, np.ndarray, pd.Series],
        lat2: Union[float, np.ndarray, pd.Series]
) -> Union[float, np.ndarray, pd.Series]:
    lon1 = to_radians(degrees=lon1)
    lat1 = to_radians(degrees=lat1)
    lon2 = to_radians(degrees=lon2)
    lat2 = to_radians(degrees=lat2)
    delta_lon = lon2 - lon1
    x = np.sin(delta_lon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - (np.sin(lat1) * np.cos(lat2) * np.cos(delta_lon))
    return np.arctan2(x, y)


def compute_compass_bearing(
        lon1: Union[float, np.ndarray, pd.Series],
        lat1: Union[float, np.ndarray, pd.Series],
        lon2: Union[float, np.ndarray, pd.Series],
        lat2: Union[float, np.ndarray, pd.Series]
) -> Union[float, np.ndarray, pd.Series]:
    initial_bearing = compute_bearing(lon1, lat1, lon2, lat2)
    initial_bearing = to_degrees(radians=initial_bearing)
    compass_bearing = (initial_bearing + 360) % 360
    return compass_bearing


def determine_direction(
        lon1: float,
        lat1: float,
        lon2: float,
        lat2: float,
        lon3: float,
        lat3: float,
        straight_line_threshold: float = 3.0
) -> RoadOptions:
    initial_bearing = compute_compass_bearing(lon1, lat1, lon2, lat2)
    new_bearing = compute_compass_bearing(lon2, lat2, lon3, lat3)

    delta_heading = new_bearing - initial_bearing
    if delta_heading > 180:
        delta_heading -= 360
    elif delta_heading < -180:
        delta_heading += 360

    if abs(delta_heading) <= straight_line_threshold:
        return RoadOptions.STRAIGHT
    elif delta_heading > 0:
        return RoadOptions.RIGHT
    else:
        return RoadOptions.LEFT


def construct_autopilot_command(waypoint: carla.Waypoint, has_invaded_lane: bool, direction: RoadOptions) -> RoadOptions:
    if waypoint.is_junction:
        return direction
    elif has_invaded_lane:
        lane_change = waypoint.lane_change
        if lane_change == carla.LaneChange.Left:
            return RoadOptions.LEFT
        elif lane_change == carla.LaneChange.Right:
            return RoadOptions.RIGHT
        else:
            return direction
    else:
        return RoadOptions.LANE_FOLLOW
