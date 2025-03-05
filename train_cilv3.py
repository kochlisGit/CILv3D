import os
import logging
import pickle
import random
import numpy as np
import tensorflow as tf
from src.models.imitation.modules.cilv3 import cilv3
from src.models.imitation.modules.cilv3.config import CILv3Config
from src.models.imitation.dataloaders.tf import CarlaTFDataset
from src.trainers.tf import TFTrainer

config = CILv3Config()
carla_directory = 'storage/datasets/carla'
train_town_list = ['Town01', 'Town02', 'Town03', 'Town04', 'Town05', 'Town10HD']
test_town_list = ['Town06', 'Town07']
# train_town_list = ['Town10HD']
# test_town_list = ['Town10HD']
test_batch_size = 256
verbose = True


def main():
    random.seed(config.seed)
    np.random.seed(seed=config.seed)
    tf.random.set_seed(seed=config.seed)

    carla_dataset = CarlaTFDataset(
        root_directory=carla_directory,
        image_size=config.image_size,
        sequence_size=None,
        normalize_images=config.normalize_images,
        use_imagenet_normalization=config.use_imagenet_normalization,
        seed=config.seed
    )

    print('\n--- Loading Training Data ---\n')

    train_dataset = carla_dataset.load_dataset(
        town_list=train_town_list,
        image_augmentations=config.image_augmentations,
        control_normalizer=config.control_normalizer,
        control_noise=config.control_noise,
        batch_size=config.batch_size,
        shuffle=True
    )

    normalizer = carla_dataset.control_normalizer
    if normalizer is not None:
        os.makedirs(name=f'{config.checkpoint_dir}/{config.model_name}', exist_ok=True)
        with open(f'{config.checkpoint_dir}/{config.model_name}/normalizer.pkl', mode='wb') as pickle_file:
            pickle.dump(obj=normalizer, file=pickle_file)

    print('\n--- Loading Validation Data ---\n')

    eval_dataset = carla_dataset.load_dataset(
        town_list=train_town_list,
        image_augmentations=None,
        control_normalizer=normalizer,
        control_noise=False,
        batch_size=test_batch_size,
        shuffle=False
    )

    print('\n--- Constructing Model ---\n')

    model = cilv3.build_model(
        image_shape=(config.image_size[0], config.image_size[1], 3),
        control_size=carla_dataset.control_size,
        config=config
    )

    if verbose:
        model.summary(expand_nested=True)

    print('\n--- Training Model ---\n')

    trainer = TFTrainer(
        model_name=config.model_name,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        use_ema=config.use_ema,
        clip_value=config.clip_value,
        loss=config.loss,
        metrics=config.metrics,
        epochs=config.epochs,
        lr_decay_patience=config.lr_decay_patience,
        lr_decay_factor=config.lr_decay_factor,
        early_stopping_patience=config.early_stopping_patience,
        verbose=verbose,
        checkpoint_directory=config.checkpoint_dir,
        tensorboard_directory=config.tensorboard_dir
    )
    model, history = trainer.fit(model=model, train_dataset=train_dataset, eval_dataset=eval_dataset)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    main()
