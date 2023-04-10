#!/usr/bin/env python
__author__ = "XXX"
__email__ = "XXX"

import logging
import os

from datasets.dataset_loader import DatasetLoader
import pandas as pd

from utils.pandas_utils import remap_columns_consecutive
from constants import *

logger = logging.getLogger(__name__)
logging.basicConfig(level=os.environ.get("LOGLEVEL"))


class Movielens1M(
    DatasetLoader,
    name="MovieLens1M",
    directory_name="ml-1m",
    url="http://files.grouplens.org/datasets/movielens/ml-1m.zip",
    url_archive_format="zip",
    expected_files=["ratings.dat", "users.dat", "movies.dat"],
    description="MovieLens 1M movie ratings. Stable benchmark dataset. 1 million ratings from 6000 users on 4000 movies. Released 2/2003.",
    source="https://grouplens.org/datasets/movielens/1m/",
):
    pass

    def load(self):
        self.download()
        ratings, users, movies, *_ = [
            self._resolve_path(path) for path in self.expected_files
        ]

        data_df = pd.read_csv(
            ratings,
            sep="::",
            header=None,
            names=["userID", "itemID", "rating", "timestamp"],
            usecols=["userID", "itemID", "rating"],
        )

        # remove data_df associated to rating < 3
        df = data_df[data_df["rating"] >= 3]

        # remove unused columns and drop duplicates
        df = df.drop(["rating"], axis=1).drop_duplicates()
        # remap users and artists idxs
        df = df.rename(columns={"userID": DEFAULT_USER_COL, "itemID": DEFAULT_ITEM_COL})
        remap_columns_consecutive(df, columns_names=["userID", "itemID"])
        logging.info(f"dataset interactions: {len(df)}")
        return df
