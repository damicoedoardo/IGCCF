#!/usr/bin/env python
__author__ = "XXX"
__email__ = "XXX"

import logging
import math

import numpy as np
import pandas as pd

from constants import *

log = logging.getLogger(__name__)


def _split_pandas_data_with_ratios(data, ratios, seed=SEED, shuffle=False):
    """Helper function to split pandas DataFrame with given ratios

    Note:
        Implementation referenced from `this source
        <https://stackoverflow.com/questions/38250710/how-to-split-data-into-3-sets-train-validation-and-test>`_.

    Args:
        data (pd.DataFrame): Pandas data frame to be split.
        ratios (list of floats): list of ratios for split. The ratios have to sum to 1.
        seed (int): random seed.
        shuffle (bool): whether data will be shuffled when being split.

    Returns:
        list: List of pd.DataFrame split by the given specifications.
    """
    if math.fsum(ratios) != 1.0:
        raise ValueError("The ratios have to sum to 1")

    split_index = np.cumsum(ratios).tolist()[:-1]

    if shuffle:
        data = data.sample(frac=1, random_state=seed)

    splits = np.split(data, [round(x * len(data)) for x in split_index])

    # Add split index (this makes splitting by group more efficient).
    for i in range(len(ratios)):
        splits[i]["split_index"] = i

    return splits


def split_stratified(data, ratio, col_user=DEFAULT_USER_COL, seed=SEED, shuffle=True):
    """Pandas stratified splitter.

    For each user / item, the split function takes proportions of ratings which is
    specified by the split ratio(s). The split is stratified.

    Args:
        data (pd.DataFrame): Pandas DataFrame to be split.
        ratio (list): Ratio for splitting data. list of float numbers, the splitter splits data into several portions
         corresponding to the split ratios. If ratios are not summed to 1, they will be normalized.
        seed (int): Seed.
        shuffle (bool): Whether or not shuffle each user profile before splitting
        col_user (str): column name of user IDs.

    Returns:
        list: Splits of the input data as pd.DataFrame.
    """
    if col_user not in data.columns:
        raise ValueError("Schema of data not valid. Missing User Col")

    if math.fsum(ratio) != 1.0:
        logging.warning("ratios passed don't sum to 1, normalization is applied")
        ratio = [x / math.fsum(ratio) for x in ratio]

    df_grouped = data.groupby(col_user)

    # Split by each group and aggregate splits together.
    splits = []

    for name, group in df_grouped:
        group_splits = _split_pandas_data_with_ratios(
            df_grouped.get_group(name), ratio, shuffle=shuffle, seed=seed
        )

        # Concatenate the list of split dataframes.
        concat_group_splits = pd.concat(group_splits)

        splits.append(concat_group_splits)

    # Concatenate splits for all the groups together.
    splits_all = pd.concat(splits)

    # Take split by split_index
    splits_list = [
        splits_all[splits_all["split_index"] == x].drop("split_index", axis=1)
        for x in range(len(ratio))
    ]

    return splits_list


def split_hit_rate(data, col_user=DEFAULT_USER_COL, seed=SEED, shuffle=True):
    """
    Remove a single interaction for each user

    Args:
        data (pd.DataFrame): interactions dataframe, every row is an interaction between a user and an item
        col_user (str): name of the column containig user IDs
        seed (int): random seed
        shuffle (bool): whether to shuffle the user profile before sample the interaction to remove,
            if False the last one will be removed
    Returns:
        pd.DataFrame, pd.DataFrame: train and test interactions dataframes
    """
    df_grouped = data.groupby(col_user)
    train = []
    test = []

    for name, group in df_grouped:
        if shuffle:
            group = group.sample(frac=1, random_state=seed)
        sample = group.loc[[group.index[-1]]]
        group = group.drop(group.index[-1])

        train.append(group)
        test.append(sample)

    train_df = pd.concat(train)
    test_df = pd.concat(test)
    return train_df, test_df
