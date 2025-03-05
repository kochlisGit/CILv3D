import os
import tensorflow as tf
from typing import List, Optional, Tuple, Union


class TFTrainer:
    def __init__(
            self,
            model_name: str,
            learning_rate: float,
            weight_decay: Optional[float],
            clip_value: Optional[float],
            use_ema: Optional[bool],
            loss: str,
            metrics: Union[str, List[str]],
            epochs: int,
            lr_decay_patience: Optional[int],
            lr_decay_factor: float,
            early_stopping_patience: Optional[int],
            verbose: bool,
            checkpoint_directory: str,
            tensorboard_directory: str
    ):
        self._learning_rate = learning_rate
        self._weight_decay = weight_decay
        self._clip_value = clip_value
        self._use_ema = use_ema if use_ema is not None else False
        self._loss = loss
        self._metrics = metrics
        self._epochs = epochs
        self._lr_decay_patience = lr_decay_patience
        self._lr_decay_factor = lr_decay_factor
        self._early_stopping_patience = early_stopping_patience
        self._verbose = verbose
        self._checkpoint_directory = f'{checkpoint_directory}/{model_name}'
        self._tensorboard_directory = f'{tensorboard_directory}/{model_name}'

        os.makedirs(name=self._checkpoint_directory, exist_ok=True)
        os.makedirs(name=self._tensorboard_directory, exist_ok=True)

    def _get_callbacks(self) -> List[tf.keras.callbacks.Callback]:
        callbacks = []

        if self._lr_decay_patience is not None:
            callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=self._lr_decay_factor,
                patience=self._lr_decay_patience,
                verbose=self._verbose,
                mode='min',
            ))

        if self._early_stopping_patience is not None:
            callbacks.append(tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=self._early_stopping_patience,
                verbose=self._verbose,
                mode='min',
                restore_best_weights=True
            ))

        callbacks.append(tf.keras.callbacks.ModelCheckpoint(
            filepath=f'{self._checkpoint_directory}/weights.h5',
            monitor='val_loss',
            verbose=self._verbose,
            save_best_only=True,
            save_weights_only=True,
            save_freq='epoch'
        ))
        callbacks.append(tf.keras.callbacks.TensorBoard(
            log_dir=self._tensorboard_directory,
            histogram_freq=0,
            write_graph=False,
            write_images=False,
            write_steps_per_second=False,
            update_freq='epoch',
        ))
        return callbacks

    def _compile(self, model: tf.keras.Model):
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=self._learning_rate,
            weight_decay=self._weight_decay,
            clipvalue=self._clip_value,
            use_ema=self._use_ema
        )
        model.compile(optimizer=optimizer, loss=self._loss, metrics=self._metrics)

    def fit(self, model: tf.keras.Model, train_dataset: tf.data.Dataset, eval_dataset: tf.data.Dataset) -> Tuple:
        self._compile(model=model)

        history = model.fit(
            train_dataset,
            batch_size=None,
            epochs=self._epochs,
            callbacks=self._get_callbacks(),
            validation_data=eval_dataset,
            verbose='auto' if self._verbose else 1
        )
        return model, history
