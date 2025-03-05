from enum import Enum


class ControlNormalization(Enum):
    MINMAX = 'minmax'
    MAXABS = 'maxabs'
    STANDARD = 'standard'
