#!/usr/bin/env python
__author__ = "XXX"
__email__ = "XXX"

from utils.general_utils import *
import numpy as np
from constants import *
from utils import pandas_utils
import tensorflow as tf


class TripletsBPRGenerator:
    """
    Creates batches of triplets of (user, positive_item, negative_item)

    Create batches of triplets required to train an algorithm with BPR loss function

    Attributes:
        train_data (pd.DataFrame): DataFrame containing user-item interactions
        batch_size (int): size of the batch to be generated
        items_after_users_idxs (bool): whether or not make the ids of items start after the last user id
        seed (int): random seed used to generate samples
    """

    def __init__(
        self,
        train_data,
        batch_size,
        items_after_users_idxs=False,
        seed=SEED,
        full_data=None,
    ):
        """
        Creates batches of triplets of (user, positive_item, negative_item)

        Create batches of triplets required to train an algorithm with BPR loss function

        Args:
            train_data (pd.DataFrame): DataFrame containing user-item interactions
            batch_size (int): size of the batch to be generated
            items_after_users_idxs (bool): whether or not make the ids of items start after the last user id
            seed (int): random seed used to generate samples
            full_data (pd.DataFrame): DataFrame containing all user-item interactions of the dataset
        """
        assert pandas_utils.has_columns(
            df=train_data, columns=[DEFAULT_USER_COL, DEFAULT_ITEM_COL]
        )

        # set random seed
        np.random.seed(seed)

        # drop unnecessary columns
        train_data = train_data[[DEFAULT_USER_COL, DEFAULT_ITEM_COL]]

        if items_after_users_idxs:
            if full_data is not None:
                full_data = full_data[[DEFAULT_USER_COL, DEFAULT_ITEM_COL]]
                last_user = max(full_data[DEFAULT_USER_COL].unique())
                full_data[DEFAULT_ITEM_COL] = (
                    full_data[DEFAULT_ITEM_COL] + last_user + 1
                )

            # item ids have to start from last user + 1
            last_user = max(train_data[DEFAULT_USER_COL].unique())
            train_data[DEFAULT_ITEM_COL] = train_data[DEFAULT_ITEM_COL] + last_user + 1

        if full_data is not None:
            self.user_idxs = list(sorted(full_data[DEFAULT_USER_COL].unique()))
            self.item_idxs = list(sorted(full_data[DEFAULT_ITEM_COL].unique()))
        else:
            self.user_idxs = list(sorted(train_data[DEFAULT_USER_COL].unique()))
            self.item_idxs = list(sorted(train_data[DEFAULT_ITEM_COL].unique()))

        self.train_data = train_data
        # number of rows of the dataframe
        self.num_samples = train_data.shape[0]
        self.batch_size = batch_size

        interaction_list = np.array(list(train_data.itertuples(index=False, name=None)))
        self.interactions_list = interaction_list

        # create user-item dict {user_idx: { item_idx: rating, ..., }, ...}
        # useful for negative sampling
        train_data_grouped = (
            train_data.groupby(DEFAULT_USER_COL)[DEFAULT_ITEM_COL]
            .apply(list)
            .reset_index()
        )
        self.user_item_dict = dict(
            map(
                lambda x: (x[0], dict(zip(x[1], np.ones(len(x[1]))))),
                train_data_grouped.values,
            )
        )

    def _negative_sample(self, u):
        def _get_random_key(list):
            L = len(list)
            i = np.random.randint(0, L)
            return list[i]

        # sample negative sample
        j = _get_random_key(self.item_idxs)
        while j in self.user_item_dict[u]:
            j = _get_random_key(self.item_idxs)
        return j

    def sample(self):
        """Create batch of triplets to optimize BPR loss

        The data are provided as following: [[user_0, ... ,], [pos_item_0, ... ,] [neg_item_0], ... ,]]

        Returns:
            np.array: batch of triplets
        """
        u_list = []
        i_list = []
        j_list = []

        pos_sample_idx = np.random.random_integers(
            low=0, high=round(self.num_samples-1), size=self.batch_size
        )
        pos_sample = self.interactions_list[pos_sample_idx]
        u, i = list(zip(*pos_sample))
        u_list.extend(u)
        i_list.extend(i)
        for u in u_list:
            j_list.append(self._negative_sample(u))
        return np.array([u_list, i_list, j_list])

class MRFTripletsBPRGenerator:
    """
    Creates batches of triplets of (user, positive_item, negative_item)

    Create batches of triplets required to train an algorithm with BPR loss function

    Attributes:
        train_data (pd.DataFrame): DataFrame containing user-item interactions
        batch_size (int): size of the batch to be generated
        items_after_users_idxs (bool): whether or not make the ids of items start after the last user id
        seed (int): random seed used to generate samples
    """

    def __init__(
        self,
        train_data,
        batch_size,
        propagation_matrix,
        items_after_users_idxs=False,
        seed=SEED,
        full_data=None,
    ):
        """
        Creates batches of triplets of (user, positive_item, negative_item)

        Create batches of triplets required to train an algorithm with BPR loss function

        Args:
            train_data (pd.DataFrame): DataFrame containing user-item interactions
            batch_size (int): size of the batch to be generated
            items_after_users_idxs (bool): whether or not make the ids of items start after the last user id
            seed (int): random seed used to generate samples
            full_data (pd.DataFrame): DataFrame containing all user-item interactions of the dataset
        """
        assert pandas_utils.has_columns(
            df=train_data, columns=[DEFAULT_USER_COL, DEFAULT_ITEM_COL]
        )

        # set random seed
        np.random.seed(seed)

        # drop unnecessary columns
        train_data = train_data[[DEFAULT_USER_COL, DEFAULT_ITEM_COL]]

        if items_after_users_idxs:
            if full_data is not None:
                full_data = full_data[[DEFAULT_USER_COL, DEFAULT_ITEM_COL]]
                last_user = max(full_data[DEFAULT_USER_COL].unique())
                full_data[DEFAULT_ITEM_COL] = (
                    full_data[DEFAULT_ITEM_COL] + last_user + 1
                )

            # item ids have to start from last user + 1
            last_user = max(train_data[DEFAULT_USER_COL].unique())
            train_data[DEFAULT_ITEM_COL] = train_data[DEFAULT_ITEM_COL] + last_user + 1

        if full_data is not None:
            self.user_idxs = list(sorted(full_data[DEFAULT_USER_COL].unique()))
            self.item_idxs = list(sorted(full_data[DEFAULT_ITEM_COL].unique()))
        else:
            self.user_idxs = list(sorted(train_data[DEFAULT_USER_COL].unique()))
            self.item_idxs = list(sorted(train_data[DEFAULT_ITEM_COL].unique()))

        self.train_data = train_data
        # number of rows of the dataframe
        self.num_samples = train_data.shape[0]
        self.batch_size = batch_size

        interaction_list = np.array(list(train_data.itertuples(index=False, name=None)))
        self.interactions_list = interaction_list

        # create user-item dict {user_idx: { item_idx: rating, ..., }, ...}
        # useful for negative sampling
        train_data_grouped = (
            train_data.groupby(DEFAULT_USER_COL)[DEFAULT_ITEM_COL]
            .apply(list)
            .reset_index()
        )
        self.user_item_dict = dict(
            map(
                lambda x: (x[0], dict(zip(x[1], np.ones(len(x[1]))))),
                train_data_grouped.values,
            )
        )

    def _negative_sample(self, u):
        def _get_random_key(list):
            L = len(list)
            i = np.random.randint(0, L)
            return list[i]

        # sample negative sample
        j = _get_random_key(self.item_idxs)
        while j in self.user_item_dict[u]:
            j = _get_random_key(self.item_idxs)
        return j

    def sample(self):
        """Create batch of triplets to optimize BPR loss

        The data are provided as following: [[user_0, ... ,], [pos_item_0, ... ,] [neg_item_0], ... ,]]

        Returns:
            np.array: batch of triplets
        """
        u_list = []
        i_list = []
        j_list = []

        pos_sample_idx = np.random.random_integers(
            low=0, high=self.num_samples - 1, size=self.batch_size
        )
        pos_sample = self.interactions_list[pos_sample_idx]
        u, i = list(zip(*pos_sample))
        u_list.extend(u)
        i_list.extend(i)
        for u in u_list:
            j_list.append(self._negative_sample(u))
        return np.array([u_list, i_list, j_list])

class LeastSquareTupleGenerator:
    def __init__(
        self,
        train_data,
        batch_size,
        negative_perc,
        items_after_users_idxs=False,
        seed=SEED,
        full_data=None,
    ):
        assert pandas_utils.has_columns(
            df=train_data, columns=[DEFAULT_USER_COL, DEFAULT_ITEM_COL]
        )

        # set random seed
        np.random.seed(seed)

        # drop unnecessary columns
        train_data = train_data[[DEFAULT_USER_COL, DEFAULT_ITEM_COL]]

        if items_after_users_idxs:
            if full_data is not None:
                full_data = full_data[[DEFAULT_USER_COL, DEFAULT_ITEM_COL]]
                last_user = max(full_data[DEFAULT_USER_COL].unique())
                full_data[DEFAULT_ITEM_COL] = (
                        full_data[DEFAULT_ITEM_COL] + last_user + 1
                )

            # item ids have to start from last user + 1
            last_user = max(train_data[DEFAULT_USER_COL].unique())
            train_data[DEFAULT_ITEM_COL] = train_data[DEFAULT_ITEM_COL] + last_user + 1

        if full_data is not None:
            self.user_idxs = list(sorted(full_data[DEFAULT_USER_COL].unique()))
            self.item_idxs = list(sorted(full_data[DEFAULT_ITEM_COL].unique()))
        else:
            self.user_idxs = list(sorted(train_data[DEFAULT_USER_COL].unique()))
            self.item_idxs = list(sorted(train_data[DEFAULT_ITEM_COL].unique()))

        self.train_data = train_data
        # number of rows of the dataframe
        self.num_samples = train_data.shape[0]
        self.batch_size = batch_size
        self.negative_perc = negative_perc

        interaction_list = np.array(list(train_data.itertuples(index=False, name=None)))
        self.interactions_list = interaction_list

        # create user-item dict {user_idx: { item_idx: rating, ..., }, ...}
        # useful for negative sampling
        train_data_grouped = (
            train_data.groupby(DEFAULT_USER_COL)[DEFAULT_ITEM_COL]
            .apply(list)
            .reset_index()
        )
        self.user_item_dict = dict(
            map(
                lambda x: (x[0], dict(zip(x[1], np.ones(len(x[1]))))),
                train_data_grouped.values,
            )
        )

    def _negative_sample(self, u):
        def _get_random_key(list):
            L = len(list)
            i = np.random.randint(0, L)
            return list[i]

        # sample negative sample
        j = _get_random_key(self.item_idxs)
        while j in self.user_item_dict[u]:
            j = _get_random_key(self.item_idxs)
        return j

    def sample(self):
        """
        """
        u_list = []
        i_list = []
        labels = []

        pos_sample_idx = np.random.random_integers(
            low=0, high=self.num_samples - 1, size=round(self.batch_size*(1-self.negative_perc))
        )
        pos_sample = self.interactions_list[pos_sample_idx]
        u, i = list(zip(*pos_sample))
        pos_labels = np.ones(len(u))
        u_list.extend(u)
        i_list.extend(i)
        labels.extend(pos_labels)

        neg_sample_idx = np.random.random_integers(
            low=0, high=len(self.user_idxs) - 1, size=round(self.batch_size*(self.negative_perc))
        )
        neg_users = np.array(self.user_idxs)[neg_sample_idx]
        for u in neg_users:
            u_list.append(u)
            i_list.append(self._negative_sample(u))
            labels.append(0)
        data_tuples = list(zip(u_list, i_list, labels))
        np.random.shuffle(data_tuples)
        u_list, i_list, labels = list(zip(*data_tuples))
        #u_list = np.array(u_list).astype(int)
        #i_list = np.array(i_list).astype(int)
        return np.array([u_list, i_list, labels])




