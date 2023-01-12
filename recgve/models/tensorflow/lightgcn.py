#!/usr/bin/env python
__author__ = "XXX"
__email__ = "XXX"

from tensorflow import keras

import logging
import os
import tensorflow as tf
from constants import *
from representations_based_recommender import (
    RepresentationsBasedRecommender,
)
from utils.graph_utils import (
    nxgraph_from_user_item_interaction_df,
    symmetric_normalized_laplacian_matrix,
)
from utils.tensorflow_utils import to_tf_sparse_tensor
import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(level=os.environ.get("LOGLEVEL"))


class LightGCN(keras.Model, RepresentationsBasedRecommender):
    """LightGCN

    Note:
        paper: https://arxiv.org/abs/2002.02126

    Attributes:
        train_data (pd.DataFrame): dataframe containing user-item interactions
        embeddings_size (int): dimension of user-item embeddings
        convolution_depth (int): number of convolution step to perform
    """

    def __init__(self, train_data, embeddings_size, convolution_depth):
        """LightGCN

        Note:
            paper: https://arxiv.org/abs/2002.02126

        Args:
            train_data (pd.DataFrame): dataframe containing user-item interactions
            embeddings_size (int): dimension of user-item embeddings
            convolution_depth (int): number of convolution step to perform
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

        self.k = convolution_depth

        # Compute propagation matrix
        graph = nxgraph_from_user_item_interaction_df(
            train_data, user_col=DEFAULT_USER_COL, item_col=DEFAULT_ITEM_COL
        )
        S = symmetric_normalized_laplacian_matrix(graph, self_loop=False)
        self.S = to_tf_sparse_tensor(S)

    def __call__(self):
        """Return users and items embeddings

        Returns:
            tf.Variable: embeddings of users and items
        """
        x = self.embeddings
        depth_embeddings = [x]

        # propagation step
        for i in range(self.k):
            x = tf.sparse.sparse_dense_matmul(self.S, x)
            depth_embeddings.append(x)

        stackked_emb = tf.stack(depth_embeddings, axis=1)
        final_emb = tf.reduce_mean(stackked_emb, axis=1)
        return final_emb

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
