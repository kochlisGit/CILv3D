from src.models.imitation.modules.cilv2.config import CILv2Config


class CILv23DConfig(CILv2Config):
    model_name = 'cilv2-3d'
    sequence_size = 4
