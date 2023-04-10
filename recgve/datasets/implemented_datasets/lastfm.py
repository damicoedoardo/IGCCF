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


class LastFM(
    DatasetLoader,
    name="LastFM",
    directory_name="hetrec2011-lastfm-2k",
    url="http://files.grouplens.org/datasets/hetrec2011/hetrec2011-lastfm-2k.zip",
    url_archive_format="zip",
    expected_files=[
        "user_artists.dat",
        "tags.dat",
        "artists.dat",
        "user_taggedartists-timestamps.dat",
        "user_taggedartists.dat",
        "user_friends.dat",
    ],
    description="""92,800 artist listening records from 1892 users.
                   This dataset contains social networking, tagging, and music artist listening information 
                    from a set of 2K users from Last.fm online music system.
                    http://www.last.fm 

                    The dataset is released in the framework of the 2nd International Workshop on 
                    Information Heterogeneity and Fusion in Recommender Systems (HetRec 2011) 
                    http://ir.ii.uam.es/hetrec2011 
                    at the 5th ACM Conference on Recommender Systems (RecSys 2011)
                    http://recsys.acm.org/2011 
                """,
    source="https://grouplens.org/datasets/hetrec-2011/",
):
    pass

    def load(self):
        self.download()
        user_artists = self._resolve_path("user_artists.dat")
        with open(user_artists) as data:
            lines = data.readlines()
            # remove the first line which contains the columns name
            spl_lines = [line.replace("\n", "").split("\t") for line in lines]
            columns_name = spl_lines.pop(0)
        df = pd.DataFrame(spl_lines, columns=columns_name).astype(int)
        # remove unused columns and drop duplicates
        df = df.drop("weight", axis=1).drop_duplicates()
        # remap users and artists idxs
        remap_columns_consecutive(df, columns_names=["userID", "artistID"])
        df = df.rename(
            columns={"userID": DEFAULT_USER_COL, "artistID": DEFAULT_ITEM_COL}
        )
        logging.info(f"dataset interactions: {len(df)}")
        return df
