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


class Gowalla(
    DatasetLoader,
    name="Gowalla",
    directory_name="gowalla",
    url="http://snap.stanford.edu/data/loc-gowalla_totalCheckins.txt.gz",
    url_archive_format=None,
    expected_files=["loc-gowalla_totalCheckins.txt.gz"],
    description="""Gowalla is a location-based social networking website where users
     share their locations by checking-in. The friendship network is undirected and was
     collected using their public API, and consists of 196,591 nodes and 950,327 edges.
     We have collected a total of 6,442,890 check-ins of these users over the period of Feb. 2009 - Oct. 2010.
                """,
    source="https://snap.stanford.edu/data/loc-Gowalla.html",
):
    pass

    def load(self):
        self.download()
        data_path = [self._resolve_path(path) for path in self.expected_files]
        df = pd.read_csv(
            data_path[0],
            sep="\t",
            names=["userID", "check-in time", "latitude", "longitude", "itemID"],
        )
        # remove unused columns and drop duplicates
        df = df.drop(["check-in time", "latitude", "longitude"], axis=1).drop_duplicates()
        # remap users and artists idxs
        remap_columns_consecutive(df, columns_names=["userID", "itemID"])
        df = df.rename(columns={"userID": DEFAULT_USER_COL, "itemID": DEFAULT_ITEM_COL})
        logging.info(f"dataset interactions: {len(df)}")
        return df
