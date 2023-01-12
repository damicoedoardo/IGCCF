#!/usr/bin/env python
__author__ = "XXX"
__email__ = "XXX"

import logging

from tensorflow import keras

from representations_based_recommender import (
    RepresentationsBasedRecommender,
)
import pandas as pd
import tensorflow as tf
import os
from constants import *
logger = logging.getLogger(__name__)
logging.basicConfig(level=os.environ.get("LOGLEVEL"))


class MatrixFactorizationBPR(keras.Model, RepresentationsBasedRecommender):
    """Matrix factorization BPR

    Note:
        paper: https://arxiv.org/abs/1205.2618

    Attributes:
        train_data (pd.DataFrame): dataframe containing user-item interactions
        embeddings_size (int): dimension of user-item embeddings
    """

    def __init__(self, train_data, embeddings_size):
        """Matrix factorization BPR

        Note:
            paper: https://arxiv.org/abs/1205.2618

        Args:
            train_data (pd.DataFrame): dataframe containing user-item interactions
            embeddings_size (int): dimension of user-item embeddings
        """
        keras.Model.__init__(self)
        RepresentationsBasedRecommender.__init__(self, train_data)

        self.embeddings_size = embeddings_size

        # create embeddings
        initializer = tf.initializers.GlorotUniform()
        self.embeddings = tf.Variable(
            initializer(shape=[self.user_count + self.item_count, embeddings_size]),
            trainable=True,
        )

    def __call__(self):
        """Return users and items embeddings

        Returns:
            tf.Variable: embeddings of users and items
        """
        return self.embeddings

    def compute_representations(self, user_data):
        user_interactions = user_data["interactions"]
        user_id = user_interactions[DEFAULT_USER_COL].unique()
        logger.info("Computing representations")
        embeddings = self()
        users_emb = tf.gather(embeddings, tf.constant(user_id)).numpy()
        items_emb = tf.gather(
            embeddings, tf.constant(self.items_after_users_idxs)
        ).numpy()
        users_repr_df = pd.DataFrame(users_emb, index=user_id)
        items_repr_df = pd.DataFrame(items_emb, index=self.item_idxs)
        logger.info("Representation computed")
        return users_repr_df, items_repr_df

    def compute_seen_representations(self, user_data):
        """
        Compute representations for SEEN users and items

        SEEN means users and items should be present during the training procedure
        """
        pass

    def compute_unseen_representations(self, user_data):
        """
        Compute representations for UNSEEN users and items

        UNSEEN means users and items were not present at training time
        """

        # 1) gather item embeddings

        # 2) gather lambda

        # 3) loop through users and compute their representations

        # 4) set users and items representation df
        pass

