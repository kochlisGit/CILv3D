class WaymoConfig:
    def __init__(
            self,
            max_label_blobs: int,
            label_parquets_directory: str,
            image_parquets_directory: str,
            annotations_directory: str,
            images_directory: str,
            waymo_to_cityscapes_filepath: str
    ):
        # Local environment files and folders
        self.LABEL_PARQUETS_DIRECTORY = label_parquets_directory
        self.IMAGE_PARQUETS_DIRECTORY = image_parquets_directory
        self.ANNOTATIONS_DIRECTORY = annotations_directory
        self.IMAGES_DIRECTORY = images_directory
        self.waymo_to_cityscapes_filepath = waymo_to_cityscapes_filepath

        # Google cloud info
        self.PROJECT_ID = 'waymo'
        self.bucket_name = 'waymo_open_dataset_v_2_0_0'
        self.labels_prefix = 'training/camera_segmentation'
        self.images_prefix = 'training/camera_image'
        self.max_label_blobs = max_label_blobs
