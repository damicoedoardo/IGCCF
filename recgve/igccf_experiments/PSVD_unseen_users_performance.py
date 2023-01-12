#!/usr/bin/env python
__author__ = "XXX"
__email__ = "XXX"

import argparse
from datasets.implemented_datasets import *
import wandb
import pandas as pd
import numpy as np
from constants import *
import time
from data.tensorflow_data import TripletsBPRGenerator
from evaluation.topk_evaluator import Evaluator
from igccf_experiments.best_models import get_wandb_project_dict
from models.puresvd import PureSVD
from models.tensorflow.fism import FISM
import tensorflow as tf
from losses.tensorflow_losses import bpr_loss, l2_reg
from models.tensorflow.igccf import IGCCF
from utils import gpu_utils
import os

from utils.early_stopping import EarlyStoppingHandlerTensorFlow
from utils.pandas_utils import remap_column_consecutive, remap_columns_consecutive

PROJECT = "gowalla_dat"
SEEN_USERS_PERCENTAGE = [0.5, 0.6, 0.7, 0.8, 0.9]
ALGORITHM = "PureSVD"
SEEDS = [27, 83, 96, 14]


def best_param():
    return {
        "lastfm_dat": 25,
        "ml1m_dat": 50,
        "Amaz_dat": 25,
        "gowalla_dat": 2000,
    }


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    ##########################################
    # identifier of WANDB run
    ##########################################
    parser.add_argument("--wandb_project", type=str, default=PROJECT)
    parser.add_argument("--alg", type=str, default=ALGORITHM)

    args = vars(parser.parse_args())
    alg = args["alg"]
    wandb_project_dict = get_wandb_project_dict(args["wandb_project"])

    ##########################################
    # Retrieve run parameters
    ##########################################
    api = wandb.Api()
    run_identifier = "XXXXXX/{}/{}".format(
        args["wandb_project"], wandb_project_dict["igccf"]
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
    # use both train and validation data
    train_df = pd.concat([train_df, val_df])
    user_data = {"interactions": train_df}
    test_df = dataset_dict["test"]
    full_data = pd.concat([train_df, val_df, test_df])

    test_evaluator = Evaluator(
        cutoff_list=[5, 20], metrics=["Recall", "NDCG"], test_data=test_df
    )


    for seed in SEEDS:
        ##########################################
        # Split Users
        ##########################################
        # set random seed
        np.random.seed(seed)
        users = train_df[DEFAULT_USER_COL].unique()
        num_items = len(train_df[DEFAULT_ITEM_COL].unique())
        np.random.shuffle(users)

        for perc in SEEN_USERS_PERCENTAGE:

            idx = round(len(users) * perc)
            users_partitions_list = np.split(users, [idx])

            # split train and test based on user id
            train_users = users_partitions_list[0]
            unseen_users = users_partitions_list[1]

            seen_user_train_data = train_df[train_df[DEFAULT_USER_COL].isin(train_users)]
            # I need not remapped users id for predictions
            #seen_user_train_data_copy = seen_user_train_data.copy()
            unseen_user_train_data = train_df[train_df[DEFAULT_USER_COL].isin(unseen_users)]

            # remap_column_consecutive(
            #     seen_user_train_data, DEFAULT_USER_COL, mapping_dict=False
            # )

            run = wandb.init(
                project="{}_inductive_performance".format(args["wandb_project"], seed),
                config=run_parameters_dict,
            )
            run.name = f"{alg}_seen_users_percentage_{perc}_seed_{seed}"

            n_comp = best_param()[args["wandb_project"]]
            model = PureSVD(seen_user_train_data, n_components=n_comp, full_data=full_data)

            test_evaluator.evaluate_recommender(
                model, user_data={"interactions": seen_user_train_data}
            )
            print("SEEN users performance \n")
            test_evaluator.print_evaluation_results()

            renamed_dict = {}
            for k, v in test_evaluator.result_dict.items():
                renamed_dict[f"{k}_seen_users"] = v
            wandb.log(renamed_dict)

            test_evaluator.evaluate_recommender(
                model, user_data={"interactions": unseen_user_train_data}
            )
            print("UNSEEN users performance \n")
            test_evaluator.print_evaluation_results()

            renamed_dict = {}
            for k, v in test_evaluator.result_dict.items():
                renamed_dict[f"{k}_unseen_users"] = v
            wandb.log(renamed_dict)

            run.finish()
