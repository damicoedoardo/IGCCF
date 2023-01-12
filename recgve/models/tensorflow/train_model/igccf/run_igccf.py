#!/usr/bin/env python
__author__ = "XXX"
__email__ = "XXX"

import os

from datasets.implemented_datasets import *
from data.tensorflow_data import TripletsBPRGenerator
from evaluation.topk_evaluator import Evaluator
import tensorflow as tf
import time
from tqdm import tqdm
from losses.tensorflow_losses import bpr_loss, l2_reg

# select free gpu if available
from models.tensorflow.igccf import IGCCF
from utils import gpu_utils

if __name__ == "__main__":
    if gpu_utils.list_available_gpus() is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_utils.pick_gpu_lowest_memory())

    DATASET = "AmazonElectronics"
    SPLIT_NAME = "kcore10_stratified"

    BATCH_SIZE = 1024
    EMBEDDING_SIZE = 64
    CONVOLUTION_DEPTH = 2
    LR = 1e-3
    EPOCHS = 1000
    VAL_EVERY = 1
    CUTOFF = [5, 20]
    L2_REG = 0
    USER_PROFILE_DROPOUT = 0.5
    TOP_K = 5

    dataset_dict = eval(DATASET)().load_split(SPLIT_NAME)

    train_df = dataset_dict["train"]
    user_data = {"interactions": train_df}
    val_df = dataset_dict["val"]

    data_gen = TripletsBPRGenerator(
        train_data=train_df, batch_size=BATCH_SIZE, items_after_users_idxs=False
    )

    val_evaluator = Evaluator(
        cutoff_list=CUTOFF, metrics=["Recall", "NDCG"], test_data=val_df
    )

    model = IGCCF(
        train_df,
        embeddings_size=EMBEDDING_SIZE,
        convolution_depth=CONVOLUTION_DEPTH,
        user_profile_dropout=USER_PROFILE_DROPOUT,
        top_k=TOP_K,
    )

    # Initialize Adam optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=LR)

    num_batches = data_gen.num_samples // BATCH_SIZE

    @tf.function
    def train_step(idxs):
        with tf.GradientTape() as tape:
            #urm_filtered = tf.gather(model.urm, idxs[0])
            #user_emb, item_emb = model(urm_filtered)
            user_emb, item_emb = model(model.urm)
            x_u = tf.gather(user_emb, idxs[0])
            #x_u = user_emb
            x_i = tf.gather(item_emb, idxs[1])
            x_j = tf.gather(item_emb, idxs[2])
            loss = bpr_loss(x_u, x_i, x_j)
            loss += l2_reg(model, alpha=L2_REG)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss

    for epoch in range(1, EPOCHS):
        cum_loss = 0
        t1 = time.time()
        for batch in tqdm(range(num_batches)):
            idxs = tf.constant(data_gen.sample())
            loss = train_step(idxs)
            cum_loss += loss

        cum_loss /= num_batches
        log = "Epoch: {:03d}, Loss: {:.4f}, Time: {:.4f}s"
        print(log.format(epoch, cum_loss, time.time() - t1))

        if epoch % VAL_EVERY == 0:
            val_evaluator.evaluate_recommender(model, user_data)
            val_evaluator.print_evaluation_results()
