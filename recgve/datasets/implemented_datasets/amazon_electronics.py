#!/usr/bin/env python
__author__ = "XXX"
__email__ = "XXX"

import logging
import os

import pandas as pd

from constants import *
from datasets.dataset_loader import DatasetLoader
from utils.pandas_utils import remap_columns_consecutive

logger = logging.getLogger(__name__)
logging.basicConfig(level=os.environ.get("LOGLEVEL"))


class AmazonElectronics(
    DatasetLoader,
    name="Amazon Electronics",
    directory_name="Amazon_Electronics",
    url="http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/ratings_Electronics.csv",
    url_archive_format=None,
    expected_files=["ratings_Electronics.csv"],
    description="""
    This dataset contains product reviews and metadata from Amazon, including 142.8 million reviews spanning May 1996 - July 2014
                """,
    source="http://jmcauley.ucsd.edu/data/amazon/links.html",
):
    pass

    def load(self):
        self.download()
        data_path = [self._resolve_path(path) for path in self.expected_files]
        df = pd.read_csv(
            data_path[0], names=["userID", "itemID", "rating", "timestamp"]
        )
        # drop timestamp
        df = df.drop("timestamp", axis=1).drop_duplicates()
        # keep only rating >=3
        df = df[df["rating"] >= 3]
        df = df.rename(columns={"userID": DEFAULT_USER_COL, "itemID": DEFAULT_ITEM_COL})
        remap_columns_consecutive(df, columns_names=["userID", "itemID"])
        logging.info(f"dataset interactions: {len(df)}")
        return df
