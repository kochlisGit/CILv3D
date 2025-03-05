import json
import io
import os
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from itertools import islice
from PIL import Image
from tqdm import tqdm
from typing import Tuple
from src.data.waymo.config import WaymoConfig


def decode_semantic_and_instance_labels_from_panoptic_label(
        panoptic_label: np.ndarray,
        panoptic_label_divisor: int) -> Tuple[np.ndarray, np.ndarray]:
    """Converts a panoptic label into semantic and instance segmentation labels.

    Args:
        panoptic_label: A 2D array where each pixel is encoded as: semantic_label *
        panoptic_label_divisor + instance_label.
        panoptic_label_divisor: an int used to encode the panoptic labels.

    Returns:
        A tuple containing the semantic and instance labels, respectively.
    """

    if panoptic_label_divisor <= 0:
        raise ValueError('panoptic_label_divisor must be > 0.')

    return np.divmod(panoptic_label, panoptic_label_divisor)


def download_label_parquets(label_blobs, config: WaymoConfig) -> int:
    """Download label parquets from waymo google storage

    Args:
        label_blobs (google storage glob): Iterable object with blobs
        config (WaymoConfig): Inner class object containing all the configuration options for the dataset builder
    Returns: Number of downloaded label parquets.
    """

    def download_label(blob) -> bool:
        filename = blob.name.split('/')[-1]
        label_filepath = f'{config.LABEL_PARQUETS_DIRECTORY}/{filename}'

        if blob.name.endswith('.parquet') and not os.path.exists(path=label_filepath):
            blob.download_to_filename(label_filepath)
            return True

        return False

    if config.max_label_blobs > -1:
        label_blobs = islice(label_blobs, config.max_label_blobs)

    os.makedirs(name=config.LABEL_PARQUETS_DIRECTORY, exist_ok=True)

    # Using ThreadPoolExecutor to speed up the process
    with ThreadPoolExecutor() as executor:
        results = list(tqdm(iterable=executor.map(download_label, label_blobs), desc='Downloading Label Parquet'))

    return sum(results)


def download_image_parquets(bucket, config: WaymoConfig) -> int:
    """Download image parquets from waymo google storage

    Args:
        bucket (google storage bucket): The google storage bucket containing the data.
        config (WaymoConfig): Inner class object containing all the configuration options for the dataset builder
    Returns:
        Number of downloaded image parquets.
    """

    def download_image(filename: str) -> bool:
        label_filepath = f'{config.LABEL_PARQUETS_DIRECTORY}/{filename}'
        df = pd.read_parquet(label_filepath, engine='pyarrow')

        if not df.empty:
            image_filepath = f'{config.IMAGE_PARQUETS_DIRECTORY}/{filename}'

            if not os.path.exists(path=image_filepath):
                blob = bucket.blob(config.images_prefix + '/' + filename)
                blob.download_to_filename(image_filepath)
                return True
        else:
            os.remove(path=label_filepath)

        return False

    filenames = os.listdir(path=config.LABEL_PARQUETS_DIRECTORY)

    # Using ThreadPoolExecutor to speed up the process
    with ThreadPoolExecutor() as executor:
        results = list(tqdm(iterable=executor.map(download_image, filenames), desc='Downloading Image Parquet'))

    return sum(results)


def get_cityscapes_instance_ids_format_mapping(arr: np.ndarray) -> dict:
    """Convert waymo instance_id array to cityscapes instance_id format

    Args:
        arr (np.array): array corresponding to np.unique(labelIds * 1000 + instanceIds)

    Returns:
        dict: dictionary with the mapping, number -> number
    """
    arr = arr.flatten()
    mapping = {}
    class_id = 0
    i = 0
    while i < arr.shape[0]:
        if arr[i] % 1000 == 0:
            mapping[arr[i]] = int(arr[i]/1000)
            i += 1
        else:
            class_id = arr[i]//1000
            c = 1
            mapping[arr[i]] = class_id * 1000 + c
            i += 1
            c += 1
            while i < len(arr) and class_id == arr[i]//1000:
                mapping[arr[i]] = class_id * 1000 + c
                i += 1
                c += 1
    return mapping


def extract_labels(df_label: pd.DataFrame, config: WaymoConfig, mapper: np.vectorize):
    """Extract label ids from a DataFrame 

    Args:
        df_label (pd.DataFrame): pandas.DataFrame read from a parquet, contains the labels
        config (WaymoConfig): Inner class object containing all the configuration options for the dataset builder
        mapper (np.vectorize): numpy function that maps waymo label Ids to cityscapes label Ids
    """

    for _, row in df_label.iterrows():
        # Extract panoptic labels from dataframe
        panoptic_label_divisor = row['[CameraSegmentationLabelComponent].panoptic_label_divisor']
        label_key = f"{row['key.frame_timestamp_micros']}_{row['key.camera_name']}"
        label_string = row['[CameraSegmentationLabelComponent].panoptic_label']
        img = Image.open(io.BytesIO(label_string))
        arr = np.asarray(img)

        # Decode semantic and instance labels
        semantic_labels, instance_labels = decode_semantic_and_instance_labels_from_panoptic_label(arr, panoptic_label_divisor)

        # Map waymo to cityscapes
        semantic_labels_mapped = mapper(semantic_labels).astype('uint8')

        # waymo instanceIds to cityscapes format
        instance_ids = semantic_labels_mapped*1000 + instance_labels
        mapping = get_cityscapes_instance_ids_format_mapping(arr=np.unique(instance_ids))
        for k, v in mapping.items():
            instance_ids[instance_ids == k] = v

        # Save labelIds and instanceIds
        Image.fromarray(semantic_labels_mapped).save(f'{config.ANNOTATIONS_DIRECTORY}/{label_key}_gtFine_labelIds.png')
        Image.fromarray(instance_ids).save(f'{config.ANNOTATIONS_DIRECTORY}/{label_key}_gtFine_instanceIds.png')


def extract_images(df_img: pd.DataFrame, config: WaymoConfig):
    """Extract images from a DataFrame

    Args:
        df_img (pd.DataFrame): pandas.DataFrame read from a parquet, contains the images
        config (WaymoConfig): Inner class object containing all the configuration options for the dataset builder
    """

    for _, row in df_img.iterrows():
        img_key = f"{row['key.frame_timestamp_micros']}_{row['key.camera_name']}"
        image_string = row['[CameraImageComponent].image']
        Image.open(io.BytesIO(image_string)).save(f'{config.IMAGES_DIRECTORY}/{img_key}_leftImg8bit.png')


def get_mapper(config):
    """Function that generates a numpy pyfunc to map label arrays from waymo to cityscapes

    Args:
        config (WaymoConfig): Inner class object containing all the configuration options for the dataset builder

    Returns:
        np.vectorize: numpy pyfunc to do the mapping
    """

    with open(config.waymo_to_cityscapes_filepath, mode='r') as jf:
        waymo_to_cityscapes = json.load(jf)
    waymo_to_cityscapes = {int(k): v for k, v in waymo_to_cityscapes.items()}
    return np.vectorize(lambda x: waymo_to_cityscapes.get(x, x))


def process_parquet(p, config, mapper) -> bool:
    """Preprocess parquet files for images and labels

    Args:
        p (str): parquet file name
        config (WaymoConfig): Inner class object containing all the configuration options for the dataset builder
        mapper (np.vectorize): numpy pyfunc to transform waymo labels to cityscapes
    Returns:
        True if extract operation is sucessful else False
    """

    try:
        df_label = pd.read_parquet(os.path.join(config.LABEL_PARQUETS_DIRECTORY, p), engine='pyarrow')
        df_img = pd.read_parquet(os.path.join(config.IMAGE_PARQUETS_DIRECTORY, p), engine='pyarrow')
        df_img = df_img[df_img['key.frame_timestamp_micros'].isin(df_label['key.frame_timestamp_micros'].unique())]

        assert len(df_label) == len(df_img), f"df_label and df_img does not match for {p}"

        # Extracts from parquets
        extract_labels(df_label, config, mapper)
        extract_images(df_img, config)
        return True
    except Exception as e:
        print(f'An error occurred while extracting labels: {e}')

        return False


def extract_from_parquets(config: WaymoConfig):
    """Function to extract images and labels from parquets

    Args:
        config (WaymoConfig): Inner class object containing all the configuration options for the dataset builder
    Returns:
        Number of extracted image-label pairs
    """

    mapper = get_mapper(config=config)

    def process(p) -> bool:
        return process_parquet(p=p, config=config, mapper=mapper)

    with ThreadPoolExecutor() as executor:
        results = list(tqdm(iterable=executor.map(process, os.listdir(config.LABEL_PARQUETS_DIRECTORY)), desc='Extracting segmentation labels'))

    return sum(results)
