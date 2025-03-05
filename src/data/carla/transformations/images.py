from enum import Enum


class ImageTransformation(Enum):
    RANDOM_HORIZONTAL_FLIP = 'random_horizontal_flip'
    RANDON_COLOR_JITTER = 'random_color_jitter'
    RANDOM_NOISE = 'random_noise'
