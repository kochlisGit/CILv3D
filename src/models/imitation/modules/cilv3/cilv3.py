from keras_cv_attention_models import uniformer
from typing import Tuple
from src.models.imitation.modules.cilv2.cilv2 import positional_encoding, transformer_encoder
from src.models.imitation.modules.cilv3.config import CILv3Config

import tensorflow as tf


def build_model(image_shape: Tuple, control_size: int, config: CILv3Config) -> tf.keras.Model:
    # Construct CIL inputs
    left_image_inputs = tf.keras.layers.Input(shape=image_shape, name='rgb_left')
    front_image_inputs = tf.keras.layers.Input(shape=image_shape, name='rgb_front')
    right_image_inputs = tf.keras.layers.Input(shape=image_shape, name='rgb_right')
    control_inputs = tf.keras.layers.Input(shape=(control_size,), name='controls')

    # Load Uniformer Model
    uniformer_model = uniformer.UniformerSmallPlus64(
        input_shape=(224, 224, 3),
        num_classes=0,
        classifier_activation=None,
        pretrained='imagenet'
    )

    # Freeze model weights if requested
    if not config.base_model_trainable:
        uniformer_model.trainable = False

    base_model = tf.keras.Sequential(layers=[
        uniformer_model,
        tf.keras.layers.Conv2D(filters=512, kernel_size=3, activation='gelu', name='uniformer-out-conv'),
        tf.keras.layers.LayerNormalization(name='uniformer-out-layer_norm'),
        tf.keras.layers.Flatten(name='uniformer_flatten')
    ], name='base-model')

    # Construct Control Embeddings
    control_embeddings = tf.keras.layers.Dense(units=base_model.output_shape[1])(control_inputs)

    # Construct & Flatten & Add image embeddings
    left_embeddings = tf.keras.layers.Add(name=f'add_left_embeddings')([base_model(left_image_inputs), control_embeddings])
    front_embeddings = tf.keras.layers.Add(name=f'add_front_embeddings')([base_model(front_image_inputs), control_embeddings])
    right_embeddings = tf.keras.layers.Add(name=f'add_right_embeddings')([base_model(right_image_inputs), control_embeddings])
    complete_embeddings = tf.stack(values=[left_embeddings, front_embeddings, right_embeddings], axis=1)

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
        name='CILv3'
    )
