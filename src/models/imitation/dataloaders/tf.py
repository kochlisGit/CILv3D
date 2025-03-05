import cv2
import tensorflow as tf
from typing import Dict, List, Optional, Tuple, Union
from sklearn.base import TransformerMixin
from src.data.carla import utils
from src.data.carla.transformations.images import ImageTransformation
from src.data.carla.transformations.controls import ControlNormalization


class CarlaTFDataset:
    def __init__(
            self,
            root_directory: str,
            image_size: Tuple[int, int],
            sequence_size: Optional[int],
            normalize_images: bool,
            use_imagenet_normalization: bool,
            seed: int
    ):
        if sequence_size is not None and sequence_size < 2:
            raise ValueError(f'sequence_size should be greater than 1, got {sequence_size}')

        self._root_directory = root_directory
        self._image_size = image_size
        self._sequence_size = sequence_size
        self._normalize_images = normalize_images
        self._seed = seed
        self._use_imagenet_normalization = use_imagenet_normalization

        self._control_size = None
        self._control_normalizer = None
        self._image_augmentations = None

        self._imagenet_mean = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)
        self._imagenet_std = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)

    @property
    def control_size(self) -> Optional[int]:
        return self._control_size

    @property
    def control_normalizer(self) -> Optional[Union[ControlNormalization, TransformerMixin]]:
        return self._control_normalizer

    def disable_image_augmentations(self):
        self._image_augmentations = None

    @tf.function
    def _load_and_preprocess_image(self, filepath: str) -> tf.Tensor:
        def apply_data_augmentations(img: tf.Tensor) -> tf.Tensor:
            for augmentation in self._image_augmentations:
                if augmentation == ImageTransformation.RANDOM_HORIZONTAL_FLIP:
                    img = tf.image.random_flip_left_right(image=img, seed=self._seed)
                elif augmentation == ImageTransformation.RANDON_COLOR_JITTER:
                    img = tf.image.random_brightness(image=img, max_delta=0.06, seed=self._seed)
                    img = tf.image.random_contrast(image=img, lower=0.9, upper=1.1, seed=self._seed)
                    img = tf.image.random_saturation(image=img, lower=1, upper=1.5, seed=self._seed)
                    img = tf.image.random_hue(image=img, max_delta=0.1, seed=self._seed)
                elif augmentation == ImageTransformation.RANDOM_NOISE:
                    img = tf.image.random_jpeg_quality(image=img, min_jpeg_quality=65, max_jpeg_quality=95, seed=self._seed)
                else:
                    raise NotImplementedError(f'Image Augmentation method "{augmentation}" has not been defined.')

            return img

        image = tf.io.read_file(filename=filepath)
        image = tf.io.decode_jpeg(contents=image)
        image = tf.image.resize(image, size=self._image_size)

        if self._image_augmentations is not None:
            image = apply_data_augmentations(img=image)

        image = tf.image.convert_image_dtype(image, tf.float32)

        # Normalize images in range [-1.0, 1.0]
        if self._normalize_images:
            if self._use_imagenet_normalization:
                image = (image/255.0 - self._imagenet_mean)/self._imagenet_std
            else:
                image = image/128.0 - 1.0

        return image

    @tf.function
    def _apply_control_noise(self, controls: tf.Tensor) -> tf.Tensor:
        if self._sequence_size is None:
            noise = tf.random.uniform(shape=(4,), minval=0.95, maxval=1.05)
            return tf.concat([controls[:4]*noise, controls[4:]], axis=0)
        else:
            noise = tf.random.uniform(shape=(self._sequence_size, 4), minval=0.95, maxval=1.05)
            return tf.concat([controls[:, :4]*noise, controls[:, 4:]], axis=1)

    @staticmethod
    @tf.function
    def _construct_input_dict(
            left_image: tf.Tensor,
            front_image: tf.Tensor,
            right_image: tf.Tensor,
            controls: tf.Tensor,
            targets: tf.Tensor,
    ) -> Tuple[Dict[str, tf.Tensor], tf.Tensor]:
        inputs = {
            'rgb_left': left_image,
            'rgb_front': front_image,
            'rgb_right': right_image,
            'controls': controls
        }
        return inputs, targets

    def load_dataset(
            self,
            town_list: Optional[List[str]],
            image_augmentations: Optional[List[ImageTransformation]],
            control_normalizer: Optional[Union[ControlNormalization, TransformerMixin]],
            control_noise: bool,
            batch_size: int,
            shuffle: bool
    ):
        self._image_augmentations = image_augmentations

        # Load dataset in numpy arrays
        left_filepaths, front_filepaths, right_filepaths, controls_array, self._control_normalizer = utils.load_carla_dataset(
            root_directory=self._root_directory,
            town_list=town_list,
            control_normalizer=control_normalizer
        )
        self._control_size = controls_array.shape[-1]

        # Construct tensors. Map filepaths to images.
        left_dataset = tf.data.Dataset.from_tensor_slices(tensors=left_filepaths[: -1]).map(self._load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
        front_dataset = tf.data.Dataset.from_tensor_slices(tensors=front_filepaths[: -1]).map(self._load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
        right_dataset = tf.data.Dataset.from_tensor_slices(tensors=right_filepaths[: -1]).map(self._load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
        controls_dataset = tf.data.Dataset.from_tensor_slices(tensors=controls_array[: -1])

        # Construct targets
        targets_dataset = tf.data.Dataset.from_tensor_slices(tensors=controls_array[1:, 0: 2])

        # Convert inputs to sequences
        if self._sequence_size is not None:
            left_dataset = left_dataset.window(size=self._sequence_size, shift=1, drop_remainder=True).flat_map(lambda window: window.batch(self._sequence_size))
            front_dataset = front_dataset.window(size=self._sequence_size, shift=1, drop_remainder=True).flat_map(lambda window: window.batch(self._sequence_size))
            right_dataset = right_dataset.window(size=self._sequence_size, shift=1, drop_remainder=True).flat_map(lambda window: window.batch(self._sequence_size))
            controls_dataset = controls_dataset.window(size=self._sequence_size, shift=1, drop_remainder=True).flat_map(lambda window: window.batch(self._sequence_size))
            targets_dataset = targets_dataset.skip(count=self._sequence_size - 1)

        # Apply control noise
        if control_noise:
            controls_dataset = controls_dataset.map(self._apply_control_noise, num_parallel_calls=tf.data.AUTOTUNE)

        tf_dataset = tf.data.Dataset.zip(datasets=(left_dataset, front_dataset, right_dataset, controls_dataset, targets_dataset))

        if shuffle:
            tf_dataset = tf_dataset.shuffle(buffer_size=8*batch_size, seed=self._seed)

        tf_dataset = tf_dataset. \
            batch(batch_size=batch_size). \
            map(self._construct_input_dict, num_parallel_calls=tf.data.AUTOTUNE). \
            prefetch(buffer_size=tf.data.AUTOTUNE)
        return tf_dataset
