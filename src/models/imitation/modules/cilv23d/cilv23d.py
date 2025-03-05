import tensorflow as tf
from typing import Tuple
from src.models.imitation.modules.cilv2.cilv2 import positional_encoding, transformer_encoder
from src.models.imitation.modules.cilv23d.config import CILv23DConfig


def build_model(image_shape: Tuple, control_size: int, config: CILv23DConfig) -> tf.keras.Model:
    # Construct CIL inputs
    image_3d_input_shape = (config.sequence_size, image_shape[0], image_shape[1], image_shape[2])
    left_image_inputs = tf.keras.layers.Input(shape=image_3d_input_shape, name='rgb_left')
    front_image_inputs = tf.keras.layers.Input(shape=image_3d_input_shape, name='rgb_front')
    right_image_inputs = tf.keras.layers.Input(shape=image_3d_input_shape, name='rgb_right')
    control_inputs = tf.keras.layers.Input(shape=(config.sequence_size, control_size), name='controls')

    # Load ResNet50 Model
    resnet = tf.keras.applications.resnet_v2.ResNet50V2(
        weights='imagenet',
        include_top=False,
        input_shape=image_shape
    )

    # Freeze model weights if requested
    if not config.base_model_trainable:
        resnet.trainable = False

    base_model = tf.keras.Sequential(layers=[
        tf.keras.layers.Input(shape=image_3d_input_shape, name='td_resnet_input'),
        tf.keras.layers.TimeDistributed(resnet, name='td_resnet'),
        tf.keras.layers.Conv3D(filters=512, kernel_size=(config.sequence_size, 3, 3), activation='gelu', name='resnet_3dconv'),
        tf.keras.layers.LayerNormalization(name='resnet_layer_norm'),
        tf.keras.layers.Flatten(name='resnet_flatten')
    ], name='base-model')

    # Construct Control Embeddings Model
    control_embeddings = tf.keras.Sequential(
        layers=[
            control_inputs,
            tf.keras.layers.Conv1D(filters=32, kernel_size=config.sequence_size, activation='gelu'),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Dense(units=base_model.output_shape[-1])
        ]
    )

    # Construct & Flatten & Add image embeddings
    control_embeddings_out = control_embeddings(control_inputs)
    left_embeddings = tf.keras.layers.Add(name=f'add_left_embeddings')([base_model(left_image_inputs), control_embeddings_out])
    front_embeddings = tf.keras.layers.Add(name=f'add_front_embeddings')([base_model(front_image_inputs), control_embeddings_out])
    right_embeddings = tf.keras.layers.Add(name=f'add_right_embeddings')([base_model(right_image_inputs), control_embeddings_out])
    complete_embeddings = tf.concat(values=[left_embeddings, front_embeddings, right_embeddings], axis=1)

    positional_embeddings = positional_encoding(length=3, depth=complete_embeddings.shape[-1])
    complete_embeddings = tf.keras.layers.Add(name='add_positional_embeddings')([complete_embeddings, positional_embeddings])

    transformer_encoder_model = tf.keras.Model(
        inputs=complete_embeddings,
        outputs=transformer_encoder(inputs=complete_embeddings, config=config),
        name='transformer-encoder-model'
    )
    x = transformer_encoder_model(complete_embeddings)
    y = tf.keras.layers.Dense(units=2, activation=None, name='outputs')(x)
    return tf.keras.Model(
        inputs={
            'rgb_left': left_image_inputs,
            'rgb_front': front_image_inputs,
            'rgb_right': right_image_inputs,
            'controls': control_inputs
        },
        outputs=y,
        name='CILv23D'
    )
