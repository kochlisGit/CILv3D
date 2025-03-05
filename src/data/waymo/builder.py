import os
from google.cloud import storage
from src.data.waymo.config import WaymoConfig
from src.data.waymo.utils import download_image_parquets, download_label_parquets, extract_from_parquets


class WaymoDatasetBuilder:
    def __init__(
            self,
            max_label_blobs: int,
            label_parquets_directory: str,
            image_parquets_directory: str,
            annotations_directory: str,
            images_directory: str,
            waymo_to_cityscapes_filepath: str
    ):
        self.config = WaymoConfig(
            max_label_blobs=max_label_blobs,
            label_parquets_directory=label_parquets_directory,
            image_parquets_directory=image_parquets_directory,
            annotations_directory=annotations_directory,
            images_directory=images_directory,
            waymo_to_cityscapes_filepath=waymo_to_cityscapes_filepath
        )

        os.makedirs(label_parquets_directory, exist_ok=True)
        os.makedirs(image_parquets_directory, exist_ok=True)
        os.makedirs(annotations_directory, exist_ok=True)
        os.makedirs(images_directory, exist_ok=True)

    def build_dataset(self):
        client = storage.Client(project=self.config.PROJECT_ID)
        bucket = client.bucket(bucket_name=self.config.bucket_name)
        label_blobs = bucket.list_blobs(prefix=self.config.labels_prefix)

        # Download images and labels parquets
        download_label_parquets(label_blobs=label_blobs, config=self.config)
        download_image_parquets(bucket=bucket, config=self.config)

        # Extract images and labels from parquet files
        extract_from_parquets(config=self.config)
