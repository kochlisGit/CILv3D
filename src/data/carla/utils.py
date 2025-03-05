import os
import numpy as np
import pandas as pd
from typing import List, Optional, Tuple, Union
from sklearn.base import TransformerMixin
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, StandardScaler, OneHotEncoder
from tqdm import tqdm
from src.data.carla.transformations.controls import ControlNormalization
from src.simulators.carla import tools


def list_carla_directories(root_directory: str, town_list: Optional[List[str]]) -> List[str]:
    directories = []

    if town_list is None:
        town_list = os.listdir(path=root_directory)

    for town in town_list:
        if town[: 4] != 'Town':
            raise ValueError(f'Expected town name to start with "Town", got {town}')

        for episode in os.listdir(path=f'{root_directory}/{town}'):
            directories.append(f'{root_directory}/{town}/{episode}')
    return directories


def one_hot_categorical(df: pd.DataFrame, column: str, categories: List, drop_column: bool) -> pd.DataFrame:
    encoder = OneHotEncoder(categories=categories, sparse_output=False)
    one_hot_encoded = encoder.fit_transform(df[[column]])
    one_hot_df = pd.DataFrame(data=one_hot_encoded, columns=encoder.get_feature_names_out([column]))
    result_df = pd.concat([df, one_hot_df], axis=1)

    if drop_column:
        result_df.drop(columns=[column], inplace=True)

    return result_df


def preprocess_vehicle_states(state_df: pd.DataFrame) -> pd.DataFrame:
    state_df['acceleration'] = state_df['throttle'] - state_df['brake']
    bearing = pd.DataFrame(
        data=tools.compute_bearing(
            lon1=state_df['longitude'].values[: -1],
            lat1=state_df['latitude'].values[: -1],
            lon2=state_df['longitude'].values[1:],
            lat2=state_df['latitude'].values[1:]
        ),
        columns=['compass_bearing']
    )
    state_df = pd.concat((state_df, bearing), axis=1)
    state_df.loc[state_df.index[-1], 'compass_bearing'] = state_df.loc[state_df.index[-2], 'compass_bearing']
    state_df = state_df[['frame', 'acceleration', 'steer', 'speed', 'compass_bearing', 'direction', 'command']]

    road_options = tools.RoadOptions
    state_df = one_hot_categorical(
        df=state_df,
        column='direction',
        categories=[[road_options.LEFT.value, road_options.STRAIGHT.value, road_options.RIGHT.value]],
        drop_column=True
    )
    state_df = one_hot_categorical(
        df=state_df,
        column='command',
        categories=[[
            road_options.LANE_FOLLOW.value,
            road_options.CHANGE_LANE_LEFT.value,
            road_options.CHANGE_LANE_RIGHT.value,
            road_options.LEFT.value,
            road_options.STRAIGHT.value,
            road_options.RIGHT.value
        ]],
        drop_column=True
    )
    return state_df


def load_vehicle_states(filepath: str) -> Tuple[np.ndarray, List[int]]:
    state_df = pd.read_csv(filepath)
    state_df = preprocess_vehicle_states(state_df=state_df)
    controls = state_df.drop(columns=['frame']).to_numpy(dtype=np.float32)
    frame_ids = state_df['frame'].to_list()
    return controls, frame_ids


def normalize_controls(
        controls: np.ndarray,
        normalizer: Union[ControlNormalization, TransformerMixin]
) -> Tuple[np.ndarray, Optional[Union[ControlNormalization, TransformerMixin]]]:
    if isinstance(normalizer, ControlNormalization):
        if normalizer == ControlNormalization.MINMAX:
            normalizer = MinMaxScaler(feature_range=(-1.0, 1.0))
        elif normalizer == ControlNormalization.MAXABS:
            normalizer = MaxAbsScaler()
        elif normalizer == ControlNormalization.STANDARD:
            normalizer = StandardScaler()
        else:
            raise NotImplementedError(f'Controls Transformation method "{normalizer}" has not been defined.')

        normalizer.fit(controls[:, :3])
    else:
        if not isinstance(normalizer, TransformerMixin):
            raise TypeError(f'Expected controls_normalizer to be instance of "TransformerMixin", got {normalizer}')

    controls[:, :3] = normalizer.transform(controls[:, :3])
    return controls, normalizer


def load_carla_dataset(
        root_directory: str,
        town_list: Optional[List[str]],
        control_normalizer: Optional[Union[ControlNormalization, TransformerMixin]]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[Union[ControlNormalization, TransformerMixin]]]:
    left_filepaths = []
    front_filepaths = []
    right_filepaths = []
    controls = []

    carla_directory_list = list_carla_directories(root_directory=root_directory, town_list=town_list)
    for directory in tqdm(iterable=carla_directory_list, desc='Loading'):
        control_data, frame_ids = load_vehicle_states(filepath=f'{directory}/states.csv')
        controls.append(control_data)
        left_filepaths.append(np.array([f'{directory}/sensors/rgb_left/{frame}.jpg' for frame in frame_ids]))
        front_filepaths.append(np.array([f'{directory}/sensors/rgb_front/{frame}.jpg' for frame in frame_ids]))
        right_filepaths.append(np.array([f'{directory}/sensors/rgb_right/{frame}.jpg' for frame in frame_ids]))

    controls = np.vstack(controls)
    left_filepaths = np.concatenate(left_filepaths, axis=None)
    front_filepaths = np.concatenate(front_filepaths, axis=None)
    right_filepaths = np.concatenate(right_filepaths, axis=None)

    if control_normalizer is not None:
        controls, control_normalizer = normalize_controls(controls=controls, normalizer=control_normalizer)

    return left_filepaths, front_filepaths, right_filepaths, controls, control_normalizer
