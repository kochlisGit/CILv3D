import numpy as np
import tensorflow as tf
from collections import deque
from typing import Dict, Optional, Tuple
from src.simulators.carla.agents.cil import CILPilot


class CIL3DPilot(CILPilot):
    def __init__(
            self,
            model: tf.keras.Model,
            sequence_size: int,
            image_size: Tuple[int, int],
            control_normalizer: Optional,
            normalize_images: bool,
            use_imagenet_normalization: bool,
            debug: bool = False
    ):
        super().__init__(
            model=model,
            control_normalizer=control_normalizer,
            image_size=image_size,
            normalize_images=normalize_images,
            use_imagenet_normalization=use_imagenet_normalization,
            debug=debug
        )

        self._sequence_size = sequence_size
        self._is_3d_calibrated = False

        self._buffer = deque(maxlen=sequence_size)

    def reset(self):
        super().reset()

        self._is_3d_calibrated = False
        self._buffer = deque(maxlen=self._sequence_size)

    def _preprocess_inputs(self, images: Dict[str, np.ndarray], controls: np.ndarray) -> Dict[str, np.ndarray]:
        x = super()._preprocess_inputs(images=images, controls=controls)

        if not self._is_3d_calibrated:
            for _ in range(self._sequence_size):
                self._buffer.append(x)

            self._is_3d_calibrated = True
        else:
            self._buffer.pop()
            self._buffer.append(x)

        input_dict = {'rgb_left': [], 'rgb_front': [], 'rgb_right': [], 'controls': []}
        for x_old in self._buffer:
            for key, val in x_old.items():
                input_dict[key].append(val)
        return {key: np.concatenate(val, axis=0) for key, val in input_dict.items()}
