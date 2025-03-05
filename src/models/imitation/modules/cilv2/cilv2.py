import tensorflow as tf
from typing import List, Tuple
from src.models.imitation.modules.cilv2.config import CILv2Config


def positional_encoding(length: int, depth: int):
    depth = depth/2
    positions = tf.range(length, dtype=tf.float32)[:, tf.newaxis]               # (seq, 1)
    depths = tf.range(depth)[tf.newaxis, :]/depth                               # (1, depth)
    angle_rates = 1/(10000**depths)                                             # (1, depth)
    angle_rads = positions*angle_rates                                          # (pos, depth)
    pos_encoding = tf.concat([tf.sin(angle_rads), tf.cos(angle_rads)], axis=-1)
    return tf.expand_dims(tf.cast(pos_encoding, dtype=tf.float32), axis=0)


def mlp_block(x: tf.keras.layers.Layer, hidden_units: List[int], dropout_rate: float) -> tf.keras.layers.Layer:
    for i, units in enumerate(hidden_units):
        x = tf.keras.layers.Dense(units=units, activation='gelu')(x)
        x = tf.keras.layers.Dropout(rate=dropout_rate)(x)
    return x


def transformer_block(
        inputs: tf.keras.layers.Layer,
        num_heads: int,
        projection_dim: int,
        transformer_head_units: List[int],
) -> tf.keras.layers.Layer:
    # Layer normalization 1.
    x1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(inputs)

    # Create a multi-head attention layer.
    attention_output = tf.keras.layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=projection_dim,
        dropout=0.1
    )(x1, x1)

    # Skip connection 1.
    x2 = tf.keras.layers.Add()([attention_output, inputs])

    # Layer normalization 2.
    x3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x2)

    # MLP.
    x3 = mlp_block(x=x3, hidden_units=transformer_head_units, dropout_rate=0.1)

    # Skip connection 2.
    return tf.keras.layers.Add()([x3, x2])


def transformer_encoder(inputs, config: CILv2Config) -> tf.keras.layers.Layer:
    x = tf.keras.layers.Dense(config.projection_dim)(inputs)

    # Create multiple layers of the Transformer block.
    for i in range(config.transformer_layers):
        x = transformer_block(
            inputs=x,
            num_heads=config.num_heads,
            projection_dim=config.projection_dim,
            transformer_head_units=config.transformer_head_units
        )

    # Create a [batch_size, projection_dim] tensor.
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dropout(rate=config.encoder_dropout)(x)

    # Add MLP.
    features = mlp_block(x=x, hidden_units=config.mlp_head_units, dropout_rate=config.encoder_dropout)
    return features


def build_model(image_shape: Tuple, control_size: int, config: CILv2Config) -> tf.keras.Model:
    # Construct CIL inputs
    left_image_inputs = tf.keras.layers.Input(shape=image_shape, name='rgb_left')
    front_image_inputs = tf.keras.layers.Input(shape=image_shape, name='rgb_front')
    right_image_inputs = tf.keras.layers.Input(shape=image_shape, name='rgb_right')
    control_inputs = tf.keras.layers.Input(shape=(control_size,), name='controls')

    # Load ResNet50 Model
    resnet_input = tf.keras.layers.Input(shape=image_shape, name='resnet_input')
    resnet = tf.keras.applications.resnet_v2.ResNet50V2(
        weights='imagenet',
        include_top=False,
        input_shape=image_shape
    )

    # Freeze model weights if requested
    if not config.base_model_trainable:
        resnet.trainable = False

    resnet_conv = tf.keras.layers.Conv2D(
        filters=512,
        kernel_size=3,
        activation='gelu',
        input_shape=resnet.output_shape[1:],
        name='resnet-out-conv'
    )
    resnet_conv_norm = tf.keras.layers.LayerNormalization(name='resnet-out-layer_norm')
    resnet_flatten = tf.keras.layers.Flatten(name='resnet_flatten')
    base_model = tf.keras.Sequential(
        layers=[
            resnet_input,
            resnet,
            resnet_conv,
            resnet_conv_norm,
            resnet_flatten
        ],
        name='base-model'
    )

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
        name='CILv2'
    )
