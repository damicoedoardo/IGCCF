#!/usr/bin/env python
__author__ = "XXX"
__email__ = "XXX"

from abc import abstractmethod, ABC
import numpy as np
from recommender_interface import Recommender, Recommender
import pandas as pd


class RepresentationsBasedRecommender(Recommender, ABC):
    """Representation based algorithm interface

    Interface for recommendation system algorithms which learn users and items embeddings to retrieve recommendation

    We use `pandas` dataframe to store the representations for both user and item, the dataframes have to be indexed by
    the user and item idxs

    Attributes:
        train_data (pd.DataFrame): dataframe containing user-item interactions
    """

    def __init__(self, train_data):
        """Representation based algorithm interface

        Interface for recommendation system algorithms which learn users and items embeddings to retrieve recommendation

        We use `pandas` dataframe to store the representations for both user and item, the dataframes have to be indexed by
        the user and item idxs

        Args:
            train_data (pd.DataFrame): dataframe containing user-item interactions
        """
        super().__init__(train_data=train_data)

    @abstractmethod
    def compute_representations(self, user_data):
        """Compute users and items representations

        Args:
            user_data (dict):dictionary containing inside `interactions` the interactions of the users for which
                retrieve predictions stored inside a pd.DataFrame

        Returns:
            pd.DataFrame, pd.DataFrame: user representations, item representations
        """
        pass

    def compute_items_scores(self, user_data):
        """Compute items scores as dot product between users and items representations

        Args:
            user_data (dict):dictionary containing inside `interactions` the interactions of the users for which
                retrieve predictions stored inside a pd.DataFrame

        Returns:
            pd.DataFrame: items scores for each user
        """
        users_repr_df, items_repr_df = self.compute_representations(user_data)

        assert isinstance(users_repr_df, pd.DataFrame) and isinstance(
            items_repr_df, pd.DataFrame
        ), "Representations have to be stored inside pd.DataFrane objects!\n user: {}, item: {}".format(
            type(users_repr_df), type(items_repr_df)
        )
        assert (
            users_repr_df.shape[1] == items_repr_df.shape[1]
        ), "Users and Items representations have not the same shape!\n user: {}, item: {}".format(
            users_repr_df.shape[1], items_repr_df.shape[1]
        )

        # sort items representations
        items_repr_df.sort_index(inplace=True)

        # compute the scores as dot product between users and items representations
        arr_scores = users_repr_df.to_numpy().dot(items_repr_df.to_numpy().T)
        scores = pd.DataFrame(arr_scores, index=users_repr_df.index)
        return scores
