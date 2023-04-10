#!/usr/bin/env python
__author__ = "XXX"
__email__ = "XXX"

import math

from utils.graph_utils import *
import networkx as nx
from constants import *
import logging
import os
from utils.pandas_utils import remap_columns_consecutive

log = logging.getLogger(__name__)
logging.basicConfig(level=os.environ.get("LOGLEVEL"))


def _check_min_rating_filter(filter_by, min_rating, col_user, col_item):
    if not (filter_by == "user" or filter_by == "item"):
        raise ValueError("filter_by should be either 'user' or 'item'.")

    if min_rating < 1:
        raise ValueError("min_rating should be integer and larger than or equal to 1.")

    split_by_column = col_user if filter_by == "user" else col_item
    split_with_column = col_item if filter_by == "user" else col_user
    return split_by_column, split_with_column


def min_rating_filter_pandas(
    data,
    min_rating=1,
    filter_by="user",
    col_user=DEFAULT_USER_COL,
    col_item=DEFAULT_ITEM_COL,
):
    """Filter rating DataFrame for each user with minimum rating.

    Filter rating data frame with minimum number of ratings for user/item is usually useful to
    generate a new data frame with warm user/item. The warmth is defined by min_rating argument. For
    example, a user is called warm if he has rated at least 4 items.

    Args:
        data (pd.DataFrame): DataFrame of user-item tuples. Columns of user and item
            should be present in the DataFrame while other columns like rating,
            timestamp, etc. can be optional.
        min_rating (int): minimum number of ratings for user or item.
        filter_by (str): either "user" or "item", depending on which of the two is to
            filter with min_rating.
        col_user (str): column name of user ID.
        col_item (str): column name of item ID.

    Returns:
        pd.DataFrame: DataFrame with at least columns of user and item that has been
            filtered by the given specifications.
    """
    split_by_column, _ = _check_min_rating_filter(
        filter_by, min_rating, col_user, col_item
    )
    grp = data.groupby(split_by_column).size()
    users_to_remove = list(grp[grp < min_rating].index)
    rating_filtered = data[~data[split_by_column].isin(users_to_remove)]
    # rating_filtered = grp.filter(lambda x: len(x) >= min_rating)
    return rating_filtered


def kcore(df, k, col_user=DEFAULT_USER_COL, col_item=DEFAULT_ITEM_COL, remap_idxs=True):
    """Remove users and items with less than K interactions

    Filter users and items with less than K interactions from a user-item interaction dataframe

    Args:
        df (pd.Dataframe): DataFrame of user-item tuples. Columns of user and item
            should be present in the DataFrame while other columns like rating,
        k (int): minimum number of interactions for a user/item to keep
        col_user (str, optional): column name of user ID.
        col_item (str, optional): column name of item ID.
        remap_idxs (bool, optional): whether to remap users and items idxs into consecutive

    Returns:
         pd.DaraFrame: Dataframe with users and items with at least K interactions
    """
    unique_users = df[col_user].unique()
    unique_items = df[col_item].unique()

    while True:
        df = min_rating_filter_pandas(
            data=df,
            col_user=col_user,
            col_item=col_item,
            min_rating=k,
            filter_by="user",
        )
        df = min_rating_filter_pandas(
            data=df,
            col_user=col_user,
            col_item=col_item,
            min_rating=k,
            filter_by="item",
        )

        if len(unique_users) == len(df[col_user].unique()) and len(unique_items) == len(
            df[col_item].unique()
        ):
            break

        removed_users_num = len(unique_users) - len(df[col_user].unique())
        removed_items_num = len(unique_items) - len(df[col_item].unique())
        log.info(
            f"Removed users: {removed_users_num}\n"
            f"Removed items: {removed_items_num}\n"
            f"\n"
        )

        unique_users = df[col_user].unique()
        unique_items = df[col_item].unique()

    if remap_idxs:
        remap_columns_consecutive(df, columns_names=[col_user, col_item])
    log.info(f"dataset interactions: {len(df)}")
    return df
