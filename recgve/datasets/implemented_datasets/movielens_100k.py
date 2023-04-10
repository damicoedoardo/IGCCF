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


class Movielens100k(
    DatasetLoader,
    name="Movielens100k",
    directory_name="ml-100k",
    url="http://files.grouplens.org/datasets/movielens/ml-100k.zip",
    url_archive_format="zip",
    expected_files=["u.data", "u.user", "u.item", "u.genre", "u.occupation",],
    description="The MovieLens 100K dataset contains 100,000 ratings from 943 users on 1682 movies.",
    source="https://grouplens.org/datasets/movielens/100k/",
):
    pass

    def load(self):
        self.download()

        ratings, users, movies, *_ = [
            self._resolve_path(path) for path in self.expected_files
        ]

        edges = pd.read_csv(
            ratings,
            sep="\t",
            header=None,
            names=["user_id", "movie_id", "rating", "timestamp"],
            usecols=["user_id", "movie_id", "rating"],
        )

        # remove edges associated to rating < 3
        df = edges[edges["rating"] >= 3]

        # remove unused columns and drop duplicates
        df = df.drop(["rating"], axis=1).drop_duplicates()
        # remap users and artists idxs
        df = df.rename(columns={"user_id": DEFAULT_USER_COL, "movie_id": DEFAULT_ITEM_COL})
        remap_columns_consecutive(df, columns_names=["userID", "itemID"])
        logging.info(f"dataset interactions: {len(df)}")
        return df
