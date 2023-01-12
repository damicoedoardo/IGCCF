#!/usr/bin/env python
__author__ = "XXX"
__email__ = "XXX"

import argparse
from datasets.implemented_datasets import *
import wandb
import pandas as pd
import time
from data.tensorflow_data import TripletsBPRGenerator
from evaluation.topk_evaluator import Evaluator
from igccf_experiments.best_models import get_wandb_project_dict
from models.tensorflow.igccf import IGCCF
import tensorflow as tf
from losses.tensorflow_losses import bpr_loss, l2_reg
from utils import gpu_utils
import os

from utils.early_stopping import EarlyStoppingHandlerTensorFlow

PROJECTS = ["lastfm_dat", "Amaz_dat", "ml1m_dat", "gowalla_dat"]
USER_PROFILE_DROPOUT = [0.0, 0.2, 0.4, 0.6, 0.8]

if __name__ == '__main__':
    # select free gpu if available
    if gpu_utils.list_available_gpus() is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_utils.pick_gpu_lowest_memory())

    for project in PROJECTS:
        wandb_project_dict = get_wandb_project_dict(project)

        ##########################################
        # Retrieve run parameters
        ##########################################
        api = wandb.Api()
        run_identifier = "XXXXXX/{}/{}".format(
            project, wandb_project_dict["igccf"]
        )
        run_object = api.run(run_identifier)
        run_parameters_dict = run_object.config
        summary = run_object.summary

        ##########################################
        # Load dataset
        ##########################################
        dataset_dict = eval(wandb_project_dict["dataset"])().load_split(
            wandb_project_dict["split_name"]
        )
        train_df = dataset_dict["train"]
        val_df = dataset_dict["val"]
        train_df = pd.concat([train_df, val_df])
        user_data = {"interactions": train_df}
        test_df = dataset_dict["test"]

        test_evaluator = Evaluator(cutoff_list=[5, 20], metrics=["Recall", "NDCG"], test_data=test_df)

        for user_profile_dropout in USER_PROFILE_DROPOUT:
            ##########################################
            # Setup sampler and optimizer
            ##########################################
            data_gen = TripletsBPRGenerator(
                train_data=train_df, batch_size=run_parameters_dict["batch_size"], items_after_users_idxs=False
            )
            # Initialize Adam optimizer
            optimizer = tf.keras.optimizers.Adam(learning_rate=run_parameters_dict["learning_rate"])
            num_batches = data_gen.num_samples // run_parameters_dict["batch_size"]

            @tf.function
            def train_step(idxs):
                with tf.GradientTape() as tape:
                    user_emb, item_emb = model(model.urm)
                    x_u = tf.gather(user_emb, idxs[0])
                    x_i = tf.gather(item_emb, idxs[1])
                    x_j = tf.gather(item_emb, idxs[2])
                    loss = bpr_loss(x_u, x_i, x_j)
                    loss += l2_reg(model, alpha=run_parameters_dict["l2_reg"])
                grads = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))
                return loss

            run = wandb.init(project=f"ablation_{project}", config=run_parameters_dict)
            run.name = f"igccf_user_profile_dropout_{user_profile_dropout}"
            model = IGCCF(
                train_df,
                embeddings_size=run_parameters_dict["embedding_size"],
                convolution_depth=run_parameters_dict["convolution_depth"],
                user_profile_dropout=user_profile_dropout,
                top_k=run_parameters_dict["top_k"],
            )

            # user early stopping handler to save models
            es_handler = EarlyStoppingHandlerTensorFlow(
                patience=100,
                save_path=os.path.join(run.dir, "best_models"),
            )

            for epoch in range(1, summary["epoch_best_result"]):
                cum_loss = 0
                t1 = time.time()
                for batch in range(num_batches):
                    idxs = tf.constant(data_gen.sample())
                    loss = train_step(idxs)
                    cum_loss += loss

                cum_loss /= num_batches
                log = "Epoch: {:03d}, Loss: {:.4f}, Time: {:.4f}s"
                print(log.format(epoch, cum_loss, time.time() - t1))

            test_evaluator.evaluate_recommender(model, user_data=user_data)
            test_evaluator.print_evaluation_results()

            wandb.log(test_evaluator.result_dict)
            es_handler.update(summary["epoch_best_result"], test_evaluator.result_dict["Recall@20"], "Recall@20", model)
            run.finish()



