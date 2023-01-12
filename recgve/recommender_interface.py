#!/usr/bin/env python
__author__ = "XXX"
__email__ = "XXX"

from utils.general_utils import get_top_k_scored_items

"""
Base interface to implement a recommendation topk-k algorithm
"""

from abc import abstractmethod, ABC

import numpy as np
import pandas as pd

from constants import *


class Recommender(ABC):
    """ Interface for recommender system algorithms

    Attributes:
        train_data (pd.DataFrame): dataframe containing user-item interactions

    Note:
        For each user-item pair is expected a row in the dataframe, additional columns are allowed

    Example:
        >>> train_data = pd.DataFram({"userID":[0, 0, 1], "itemID":[0, 1, 2]})
    """

    def __init__(self, train_data):
        """
        Interface for recommender system algorithms

        Args:
            train_data (pd.DataFrame): dataframe containing user-item interactions
        """
        for c in [DEFAULT_USER_COL, DEFAULT_ITEM_COL]:
            assert c in train_data.columns, f"column {c} not present in train_data"
        self.train_data = train_data
        self.user_idxs = sorted(train_data["userID"].unique())
        self.item_idxs = sorted(train_data["itemID"].unique())
        self.items_after_users_idxs = self.item_idxs + self.user_idxs[-1] + 1
        self.user_count = len(self.user_idxs)
        self.item_count = len(self.item_idxs)

    @abstractmethod
    def compute_items_scores(self, *args, **kwargs):
        """Computes items scores

        Compute items scores for each user store them inside a pd.DataFrame indexed by userID

        Returns:
            pd.Dataframe: items scores for each user
        """
        pass

    def remove_seen_items(self, scores, user_data):
        """Methods to set scores of items used at training time to `-np.inf`

        Args:
            scores (pd.Dataframe): items scores for each user, indexed by user id
            user_data (dict): dictionary containing inside `interactions` the interactions of the users for which
                retrieve predictions stored inside a pd.DataFrame

        Returns:
            pd.DataFrame: dataframe of scores for each user indexed by user id
        """
        users_interactions = user_data["interactions"]

        user_list = users_interactions[DEFAULT_USER_COL].values
        item_list = users_interactions[DEFAULT_ITEM_COL].values

        scores_array = scores.values

        user_index = scores.index.values
        arange = np.arange(len(user_index))
        mapping_dict = dict(zip(user_index, arange))
        user_list_mapped = np.array([mapping_dict.get(u) for u in user_list])

        scores_array[user_list_mapped, item_list] = -np.inf
        scores = pd.DataFrame(scores_array, index=user_index)

        return scores

    def recommend(self, cutoff, user_data):
        """
        Give recommendations up to a given cutoff to users inside `user_idxs` list

        Args:
            cutoff (int): cutoff used to retrieve the recommendations
            user_data (dict): dictionary containing inside `interactions` the interactions of the users for which
                retrieve predictions stored inside a pd.DataFrame

        Returns:
            pd.DataFrame: DataFrame with predictions for users

        Note:
            predictions are in the following format | userID | itemID | prediction | item_rank
        """

        # compute scores
        scores = self.compute_items_scores(user_data)

        # set the score of the items used during the training to -inf
        scores_df = self.remove_seen_items(scores, user_data)

        array_scores = scores_df.to_numpy()
        user_ids = scores_df.index.values

        items, scores = get_top_k_scored_items(
            scores=array_scores, top_k=cutoff, sort_top_k=True
        )
        # create user array to match shape of retrievied items
        users = np.repeat(user_ids, cutoff).reshape(len(user_ids), -1)

        recs_df = pd.DataFrame(
            zip(users.flatten(), items.flatten(), scores.flatten()),
            columns=[DEFAULT_USER_COL, DEFAULT_ITEM_COL, DEFAULT_PREDICTION_COL],
        )

        # add item rank
        recs_df["item_rank"] = np.tile(np.arange(1, cutoff + 1), len(user_ids))
        return recs_df
