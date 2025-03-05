from src.data.carla.transformations.controls import ControlNormalization


class CILv2Config:
    model_name = 'cilv2'
    projection_dim = 256
    num_heads = 4
    transformer_head_units = [projection_dim]
    transformer_layers = 4
    mlp_head_units = [512]
    encoder_dropout = 0.4
    base_model_trainable = False
    learning_rate = 0.001
    use_ema = False
    weight_decay = None
    clip_value = None
    loss = 'mae'
    metrics = ['mae']
    early_stopping_patience = 300
    lr_decay_factor = 0.2
    lr_decay_patience = 200
    batch_size = 64
    epochs = 500
    image_size = (224, 224)
    image_augmentations = None
    normalize_images = True
    use_imagenet_normalization = False
    control_normalizer = ControlNormalization.MINMAX
    control_noise = True
    checkpoint_dir = 'storage/checkpoints/imitation'
    tensorboard_dir = 'storage/tensorboard/imitation'
    seed = 42
