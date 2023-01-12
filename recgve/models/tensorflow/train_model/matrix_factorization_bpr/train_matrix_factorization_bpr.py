#!/usr/bin/env python
__author__ = "XXX"
__email__ = "XXX"

from datasets.implemented_datasets import *

import argparse
import os
import time

import tensorflow as tf
import wandb

from data.tensorflow_data import TripletsBPRGenerator
from evaluation.topk_evaluator import Evaluator
from losses.tensorflow_losses import bpr_loss, l2_reg
from models.tensorflow.matrix_factorization_bpr import MatrixFactorizationBPR
from utils.early_stopping import EarlyStoppingHandlerTensorFlow
from utils import gpu_utils

if __name__ == "__main__":
    parser = argparse.ArgumentParser("train BPR matrix factorization")

    # Model parameters
    parser.add_argument("--embedding_size", type=int, default=64)

    # Train parameters
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--l2_reg", type=float, default=0)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--cutoff", type=list, default=[5, 20])

    # Define the Early stopping parameters
    parser.add_argument("--val_every", type=int, default=10)
    parser.add_argument("--early_stopping", type=bool, default=True)
    parser.add_argument("--es_patience", type=int, default=5)
    parser.add_argument("--es_metric", type=str, default="Recall@20")
    parser.add_argument("--models_to_save", type=int, default=5)

    # Dataset parameters
    parser.add_argument("--dataset", type=str, default="LastFM")
    parser.add_argument("--dataset_split", type=str, default="kcore5_stratified")

    # WANDB
    parser.add_argument("--wandb", type=bool, default=True)

    parser.add_argument("--verbose", type=bool, default=True)

    args = vars(parser.parse_args())

    # select free gpu if available
    if gpu_utils.list_available_gpus() is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_utils.pick_gpu_lowest_memory())

    dataset_dict = eval(args["dataset"])().load_split(args["dataset_split"])

    train_df = dataset_dict["train"]
    user_data = {"interactions": train_df}
    val_df = dataset_dict["val"]

    data_gen = TripletsBPRGenerator(
        train_data=train_df, batch_size=args["batch_size"], items_after_users_idxs=True
    )

    val_evaluator = Evaluator(
        cutoff_list=args["cutoff"], metrics=["Recall", "NDCG"], test_data=val_df
    )

    model = MatrixFactorizationBPR(train_df, embeddings_size=args["embedding_size"])

    # add the model name inside args
    args.update({"recommender_name": model.__class__.__name__})
    # initialize wandb
    if args["wandb"]:
        wandb.init(config=args)

    if args["early_stopping"]:
        es_handler = EarlyStoppingHandlerTensorFlow(
            patience=args["es_patience"],
            save_path=os.path.join(wandb.run.dir, "best_models"),
        )

    # Initialize Adam optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=args["learning_rate"])

    num_batches = data_gen.num_samples // args["batch_size"]

    @tf.function
    def train_step(idxs):
        with tf.GradientTape() as tape:
            x = model()
            x_u = tf.gather(x, idxs[0])
            x_i = tf.gather(x, idxs[1])
            x_j = tf.gather(x, idxs[2])
            loss = bpr_loss(x_u, x_i, x_j)
            loss += l2_reg(model, alpha=args["l2_reg"])
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss

    for epoch in range(1, args["epochs"]):
        cum_loss = 0
        t1 = time.time()
        for batch in range(num_batches):
            idxs = tf.constant(data_gen.sample())
            loss = train_step(idxs)
            cum_loss += loss

        cum_loss /= num_batches
        log = "Epoch: {:03d}, Loss: {:.4f}, Time: {:.4f}s"

        if args["verbose"]:
            print(log.format(epoch, cum_loss, time.time() - t1))

        if epoch % args["val_every"] == 0:
            val_evaluator.evaluate_recommender(model, user_data=user_data)
            val_evaluator.print_evaluation_results()

            # wandb log validation metric
            if args["wandb"]:
                res_dict = val_evaluator.result_dict
                wandb.log(res_dict, step=epoch)

            if args["early_stopping"]:
                es_metric = val_evaluator.result_dict[args["es_metric"]]
                es_handler.update(epoch, es_metric, args["es_metric"], model)
                if es_handler.stop_training():
                    break

    # log best result score
    if args["early_stopping"]:
        wandb.log(es_handler.best_result_dict)
