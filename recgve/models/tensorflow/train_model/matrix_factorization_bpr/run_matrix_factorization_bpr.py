#!/usr/bin/env python
__author__ = "XXX"
__email__ = "XXX"

import os

from datasets.dataset_statistics import get_dataset_stats
from datasets.implemented_datasets import *
from data.tensorflow_data import TripletsBPRGenerator
from evaluation.topk_evaluator import Evaluator
from models.tensorflow.matrix_factorization_bpr import MatrixFactorizationBPR
import tensorflow as tf
import time
from tqdm import tqdm
from losses.tensorflow_losses import bpr_loss, l2_reg

# select free gpu if available
from utils import gpu_utils
from utils.general_utils import print_dict

if __name__ == '__main__':
    if gpu_utils.list_available_gpus() is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_utils.pick_gpu_lowest_memory())

    DATASET = "Movielens100k"
    SPLIT_NAME = "kcore10_stratified"

    BATCH_SIZE = 1024
    EMBEDDING_SIZE = 64
    LR = 1e-3
    EPOCHS = 100
    VAL_EVERY = 10
    CUTOFF = [5, 20]
    L2_REG = 1e-5

    dataset_dict = eval(DATASET)().load_split(SPLIT_NAME)

    train_df = dataset_dict["train"]
    user_data = {"interactions": train_df}
    val_df = dataset_dict["val"]

    data_gen = TripletsBPRGenerator(
        train_data=train_df, batch_size=BATCH_SIZE, items_after_users_idxs=True, full_data=train_df
    )

    val_evaluator = Evaluator(
        cutoff_list=CUTOFF, metrics=["Recall", "NDCG"], test_data=val_df
    )

    model = MatrixFactorizationBPR(train_df, embeddings_size=EMBEDDING_SIZE)

    # Initialize Adam optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=LR)

    num_batches = data_gen.num_samples // BATCH_SIZE


    @tf.function
    def train_step(idxs):
        with tf.GradientTape() as tape:
            x = model()
            x_u = tf.gather(x, idxs[0])
            x_i = tf.gather(x, idxs[1])
            x_j = tf.gather(x, idxs[2])
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
