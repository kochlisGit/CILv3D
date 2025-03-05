import logging
import carla
import cv2
import numpy as np
import tensorflow as tf
from typing import Any, Dict, Optional, Tuple
from sklearn.preprocessing import OneHotEncoder
from src.simulators.carla.agents.controller import AgentController
from src.simulators.carla import tools


class CILPilot(AgentController):
    def __init__(
            self,
            model: tf.keras.Model,
            control_normalizer: Optional,
            image_size: Tuple[int, int],
            normalize_images: bool,
            use_imagenet_normalization: bool,
            debug: bool = False
    ):
        super().__init__()

        self._model = model
        self._scaler = control_normalizer
        self._normalize_images = normalize_images
        self._image_size = image_size
        self._use_imagenet_normalization = use_imagenet_normalization
        self._debug = debug

        self._mean = np.float32([0.485, 0.456, 0.406])
        self._std = np.float32([0.229, 0.224, 0.225])

        road_options = tools.RoadOptions
        self._direction_encoder = OneHotEncoder(
            categories=[[road_options.LEFT.value, road_options.STRAIGHT.value, road_options.RIGHT.value]],
            sparse_output=False
        )
        self._command_encoder = OneHotEncoder(
            categories=[[
                road_options.LANE_FOLLOW.value,
                road_options.CHANGE_LANE_LEFT.value,
                road_options.CHANGE_LANE_RIGHT.value,
                road_options.LEFT.value,
                road_options.STRAIGHT.value,
                road_options.RIGHT.value
            ]],
            sparse_output=False
        )

        self._is_calibrated = False
        self._lon = self._lat = None

    def _compute_compass_bearing(self, lon2: float, lat2: float) -> float:
        compass_bearing = tools.compute_bearing(lon1=self._lon, lon2=lon2, lat1=self._lat, lat2=lat2)
        self._lon = lon2
        self._lat = lat2
        return compass_bearing

    def _preprocess_inputs(self, images: Dict[str, np.ndarray], controls: np.ndarray) -> Dict[str, np.ndarray]:
        for image_name, img in images.items():
            if not (img.shape[0] == self._image_size[0] and img.shape[1] == self._image_size[1]):
                img = cv2.resize(src=img, dsize=(224, 224), interpolation=cv2.INTER_LINEAR)

            if self._normalize_images:
                img = (img/255.0 - self._mean)/self._std if self._use_imagenet_normalization else img/128.0 - 1.0

            images[image_name] = np.expand_dims(img, axis=0)

        if self._scaler is not None:
            controls[:, :3] = self._scaler.transform(controls[:, :3])

        return {**images, 'controls': controls}

    def _postprocess_outputs(self, y: np.ndarray) -> Tuple[float, float]:
        if self._scaler is not None:
            dummy_y = np.float32([[y[0, 0], y[0, 1], 1.0]])
            y = self._scaler.inverse_transform(dummy_y)

        clipped_action = np.clip(y, a_min=-1.0, a_max=1.0)
        return float(clipped_action[0, 0]), float(clipped_action[0, 1])

    def reset(self):
        self._is_calibrated = False
        self._lon = self._lat = None

    def compute_vehicle_control(self, sensor_dict: Dict[str, np.ndarray], state_dict: Dict[str, Any]) -> Optional[carla.VehicleControl]:
        try:
            if not self._is_calibrated:
                self._lon = state_dict['longitude']
                self._lat = state_dict['latitude']
                self._is_calibrated = True

            compass_bearing = self._compute_compass_bearing(lon2=state_dict['longitude'], lat2=state_dict['latitude'])
            acceleration = state_dict['throttle'] - state_dict['brake']

            images = {'rgb_left': sensor_dict['rgb_left'], 'rgb_front': sensor_dict['rgb_front'], 'rgb_right': sensor_dict['rgb_right']}
            direction_vector = self._direction_encoder.fit_transform(np.array([[state_dict['direction'].value]]))[0]
            command_vector = self._command_encoder.fit_transform(np.array([[state_dict['command'].value]]))[0]
            controls = np.float32([[
                acceleration,
                state_dict['steer'],
                state_dict['speed'],
                compass_bearing,
                *direction_vector,
                *command_vector
            ]])
        except Exception as e:
            if self._debug:
                logging.info(e)

            return None
        else:
            x = self._preprocess_inputs(images=images, controls=controls)
            y = self._model.predict(x)
            acceleration, steer = self._postprocess_outputs(y=y)

            if self._debug:
                logging.info(f'Computed acceleration/steer: {acceleration}/{steer}')

            vehicle_control = self.get_vehicle_control()
            vehicle_control.steer = steer

            if acceleration > 0.0:
                vehicle_control.brake = 0.0
                vehicle_control.throttle = acceleration
            else:
                vehicle_control.throttle = 0.0
                vehicle_control.brake = -acceleration

            return vehicle_control
