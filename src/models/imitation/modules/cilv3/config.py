from src.models.imitation.modules.cilv2.config import CILv2Config


class CILv3Config(CILv2Config):
    model_name = 'cilv3'
    use_imagenet_normalization = True
