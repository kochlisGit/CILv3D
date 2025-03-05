import logging
import os
import torch
import torchvision.io as io
from torchvision.transforms import v2
from typing import List, Optional, Tuple, Union
from sklearn.base import TransformerMixin
from torch.utils.data import Dataset, DataLoader
from src.data.carla import utils
from src.data.carla.transformations.images import ImageTransformation
from src.data.carla.transformations.controls import ControlNormalization


def get_suggested_num_workers() -> int:
    if hasattr(os, 'sched_getaffinity'):
        try:
            return len(os.sched_getaffinity(0))
        except Exception as e:
            logging.warning(e)

    return os.cpu_count()


def generate_transform(
        image_size: [Tuple[int, int]],
        image_augmentations: Optional[List[ImageTransformation]],
        normalize: bool,
        use_imagenet_normalization: bool = False
) -> torch.nn.Module:
    def get_transform(augmentation: ImageTransformation) -> torch.nn.Module:
        if augmentation == ImageTransformation.RANDOM_HORIZONTAL_FLIP:
            return v2.RandomHorizontalFlip(p=0.5)
        elif augmentation == ImageTransformation.RANDON_COLOR_JITTER:
            return v2.ColorJitter(
                brightness=0.06,
                contrast=(0.9, 1.1),
                saturation=(1, 1.5),
                hue=0.1
            )
        elif augmentation == ImageTransformation.RANDOM_NOISE:
            return v2.GaussianBlur(kernel_size=5, sigma=(0.01, 0.1))
        else:
            raise NotImplementedError(f'Image Augmentation method "{augmentation}" has not been defined.')

    preprocess_transformations = [v2.Resize(size=image_size)]
    augmentations = [] if image_augmentations is None else [get_transform(augmentation=a) for a in image_augmentations]
    postprocess_transformations = [v2.ConvertImageDtype(dtype=torch.float32)]

    if normalize:
        if use_imagenet_normalization:
            postprocess_transformations.append(v2.Lambda(lambda x: x/255.0))
            postprocess_transformations.append(v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
        else:
            postprocess_transformations.append(v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))

    transforms = preprocess_transformations + augmentations + postprocess_transformations
    return torch.nn.Sequential(*transforms)


class TorchCarlaDataset(Dataset):
    def __init__(
            self,
            root_directory: str,
            town_list: Optional[List[str]],
            image_size: Tuple[int, int],
            sequence_size: Optional[int],
            image_augmentations: Optional[List[ImageTransformation]],
            normalize_images: bool,
            control_normalizer: Optional[Union[ControlNormalization, TransformerMixin]],
            control_noise: bool,
            use_imagenet_normalization: bool = False
    ):
        if sequence_size is not None and sequence_size < 2:
            raise ValueError(f'sequence_size should be greater than 1, got {sequence_size}')

        self._sequence_size = sequence_size
        self._control_noise = control_noise

        self._transform = generate_transform(
            image_size=image_size,
            image_augmentations=image_augmentations,
            normalize=normalize_images,
            use_imagenet_normalization=use_imagenet_normalization
        )

        left_filepaths, front_filepaths, right_filepaths, controls_array, self._control_normalizer = utils.load_carla_dataset(
            root_directory=root_directory,
            town_list=town_list,
            control_normalizer=control_normalizer
        )
        self._left_filepaths = left_filepaths[: -1]
        self._front_filepaths = front_filepaths[: -1]
        self._right_filepaths = right_filepaths[: -1]
        self._controls = torch.from_numpy(controls_array[: -1])
        self._targets = torch.from_numpy(controls_array[1:, 0: 2])
        self._dataset_size = self._controls.shape[0] if sequence_size is None else self._controls.shape[
                                                                                       0] - sequence_size

    @property
    def control_size(self) -> Optional[int]:
        return self._controls.shape[1] if self._sequence_size is None else self._controls.shape[1] * self._sequence_size

    @property
    def control_normalizer(self) -> Optional[Union[ControlNormalization, TransformerMixin]]:
        return self._control_normalizer

    @torch.no_grad()
    def _load_and_preprocess_image(self, filepath: str) -> torch.Tensor:
        image = io.read_image(path=filepath, mode=io.ImageReadMode.UNCHANGED)
        return self._transform(image)

    def __len__(self):
        return self._dataset_size

    def __getitem__(self, idx: int):
        if self._sequence_size is None:
            left_image = self._load_and_preprocess_image(filepath=self._left_filepaths[idx])
            front_image = self._load_and_preprocess_image(filepath=self._front_filepaths[idx])
            right_image = self._load_and_preprocess_image(filepath=self._right_filepaths[idx])
            controls = self._controls[idx]
            y = self._targets[idx]

            if self._control_noise:
                controls[:4] *= (torch.rand(size=(4,)) * 0.1 + 0.95)
        else:
            left_image = torch.stack([
                self._load_and_preprocess_image(filepath=self._left_filepaths[i])
                for i in range(idx, idx + self._sequence_size)
            ], dim=0)
            front_image = torch.stack([
                self._load_and_preprocess_image(filepath=self._front_filepaths[i])
                for i in range(idx, idx + self._sequence_size)
            ], dim=0)
            right_image = torch.stack([
                self._load_and_preprocess_image(filepath=self._right_filepaths[i])
                for i in range(idx, idx + self._sequence_size)
            ], dim=0)
            controls = self._controls[idx: idx + self._sequence_size]

            if self._control_noise:
                controls[:, :4] *= (torch.rand(size=(self._sequence_size, 4)) * 0.1 + 0.95)

            controls = controls.flatten()
            y = self._targets[idx + self._sequence_size]

        x = {'rgb_left': left_image, 'rgb_front': front_image, 'rgb_right': right_image, 'controls': controls}
        return x, y


def construct_dataloader(
        dataset: Dataset,
        batch_size: int,
        shuffle: bool,
        enable_multiprocessing: bool
) -> DataLoader:
    num_workers = get_suggested_num_workers() if enable_multiprocessing else 0

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True,
        num_workers=num_workers
    )
