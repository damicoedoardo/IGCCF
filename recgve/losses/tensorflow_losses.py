#!/usr/bin/env python
__author__ = "XXX"
__email__ = "XXX"

import tensorflow as tf


@tf.function
def bpr_loss(x_u, x_i, x_j):
    """ Create BPR loss for a batch of samples

    Args:
        x_u (tf.Tensor): tensor containing user representations
        x_i (tf.Tensor): tensor containing positive item representations
        x_j (tf.Tensor): tensor containing negative item representation

    Returns:
        loss

    Paper: BPR: Bayesian Personalized Ranking from Implicit Feedback
    https://arxiv.org/pdf/1205.2618.pdf
    """
    pos_scores = tf.reduce_sum(tf.multiply(x_u, x_i), axis=1)
    neg_scores = tf.reduce_sum(tf.multiply(x_u, x_j), axis=1)
    xuij = tf.math.log_sigmoid(pos_scores - neg_scores)
    loss = tf.negative(tf.reduce_sum(xuij))
    return loss


@tf.function
def rmse_loss(x_u, x_i, labels):
    scores = tf.reduce_sum(tf.multiply(x_u, x_i), axis=1)
    loss = tf.reduce_sum((labels - scores) ** 2)
    return loss


@tf.function
def l2_reg(model, alpha):
    """
    Create l2 loss for the model variables

    Args:
        model: model for which compute l2 reg
        alpha (float): l2 regularization coefficient
    Returns:
        float: l2 loss
    """
    l2_loss = 0
    for v in model.trainable_variables:
        l2_loss += tf.nn.l2_loss(v) * alpha
    return l2_loss
