#!/usr/bin/env python
__author__ = "XXX"
__email__ = "XXX"

import argparse
import shutil

import tensorflow as tf
import wandb

from datasets.implemented_datasets import *
from evaluation.topk_evaluator import Evaluator
from igccf_experiments.best_models import get_wandb_project_dict

# Define static variables
from models.tensorflow.ngcf import NGCF

PROJECT = "Amaz_dat"

if __name__ == "__main__":

    parser = argparse.ArgumentParser("Eval NGCF")
    ##########################################
    # identifier of WANDB run
    ##########################################
    parser.add_argument("--wandb_project", type=str, default=PROJECT)
    parser.add_argument("--cutoff", type=list, default=[5, 20])

    args = vars(parser.parse_args())
    wandb_project_dict = get_wandb_project_dict(args["wandb_project"])

    ##########################################
    # Retrieve run parameters
    ##########################################
    api = wandb.Api()
    run_identifier = "XXXXXX/{}/{}".format(
        args["wandb_project"], wandb_project_dict["ngcf"]
    )
    run_object = api.run(run_identifier)

    for f in run_object.files():
        if "best_models" in str(f):
            f.download(replace=True)

    run_parameters_dict = run_object.config

    ##########################################
    # Load dataset
    ##########################################
    dataset_dict = eval(wandb_project_dict["dataset"])().load_split(
        wandb_project_dict["split_name"]
    )
    train_df = dataset_dict["train"]
    user_data = {"interactions": train_df}
    val_df = dataset_dict["val"]
    test_df = dataset_dict["test"]

    ##########################################
    # Setting up val and test evaluator
    ##########################################
    val_evaluator = Evaluator(
        cutoff_list=run_parameters_dict["cutoff"],
        metrics=["Recall", "NDCG"],
        test_data=val_df,
    )
    test_evaluator = Evaluator(
        cutoff_list=run_parameters_dict["cutoff"],
        metrics=["Recall", "NDCG"],
        test_data=test_df,
    )

    ##########################################
    # Load model
    ##########################################
    model = NGCF(
        train_df,
        embeddings_size=run_parameters_dict["embedding_size"],
        convolution_depth=run_parameters_dict["convolution_depth"],
        mess_dropout=run_parameters_dict["mess_dropout"],
        node_dropout=run_parameters_dict["node_dropout"],
    )

    weights_path = "best_models"
    latest = tf.train.latest_checkpoint(weights_path)
    model.load_weights(latest)

    # delete the downloaded weights files
    print("Deleting restored files from wandb")
    shutil.rmtree("best_models")

    ##########################################
    # Evaluate model
    ##########################################
    val_evaluator.evaluate_recommender(model, user_data=user_data)
    val_evaluator.print_evaluation_results()

    test_evaluator.evaluate_recommender(model, user_data=user_data)
    test_evaluator.print_evaluation_results()

    ##########################################
    # Log results on WANDB
    ##########################################
    test_result_dict = {}
    for k, v in test_evaluator.result_dict.items():
        new_key = "test_{}".format(k)
        test_result_dict[new_key] = test_evaluator.result_dict[k]
    run_object.summary.update(test_result_dict)
