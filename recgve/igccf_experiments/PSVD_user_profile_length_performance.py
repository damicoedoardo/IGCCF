#!/usr/bin/env python
__author__ = "XXX"
__email__ = "XXX"

# Define static variables
import shutil
from models.puresvd import PureSVD
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import wandb
import tensorflow as tf
from datasets.implemented_datasets import *
import argparse
from constants import *
from evaluation.topk_evaluator import Evaluator
from igccf_experiments.best_models import get_wandb_project_dict, restore_models
import numpy as np
import os

METRICS = ["Recall", "NDCG"]
CUTOFFS = [5, 20]
N = 4
PROJECT = "ml1m_dat"


def profile_length_splitter(df, n):
    df_ulen = (
        df.groupby(DEFAULT_USER_COL)
        .size()
        .reset_index(name="count")
        .sort_values("count")
    )

    user_id = df_ulen[DEFAULT_USER_COL].values
    count = df_ulen["count"].values

    cum_sum = np.cumsum(count)
    partition_dim = cum_sum[-1] / n

    split_ids = []
    split_interactions = []
    for i in range(1, n):
        split_dim = partition_dim * i
        idx = (np.abs(cum_sum - split_dim)).argmin()
        split_ids.append(idx)
        split_interactions.append(count[idx])

    split_interactions.append(count[-1])
    splits = np.split(user_id, split_ids)

    split_user_count = []
    train_dfs = []
    for split in splits:
        split_user_count.append(len(split))
        train_dfs.append(df[df[DEFAULT_USER_COL].isin(split)])

    return train_dfs, split_interactions, split_user_count


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

    args = vars(parser.parse_args())
    wandb_project_dict = get_wandb_project_dict(args["wandb_project"])

    ##########################################
    # Load dataset
    ##########################################
    dataset_dict = eval(wandb_project_dict["dataset"])().load_split(
        wandb_project_dict["split_name"]
    )
    train_df = dataset_dict["train"]
    val_df = dataset_dict["val"]
    train_df = pd.concat([train_df, val_df])

    test_df = dataset_dict["test"]

    test_evaluator = Evaluator(cutoff_list=CUTOFFS, metrics=METRICS, test_data=test_df,)

    # TRAIN PURESVD
    n_comp = best_param()[args["wandb_project"]]
    model = PureSVD(train_df, n_components=n_comp)

    train_dfs, split_interactions, split_user_count = profile_length_splitter(
        train_df, n=N
    )

    algorithm = []
    score = []
    metric = []
    interactions = []

    for train_df, interactions_num in zip(train_dfs, split_interactions):
        user_data = {"interactions": train_df}
        test_evaluator.evaluate_recommender(model, user_data)
        test_evaluator.print_evaluation_results()

        for k, v in test_evaluator.result_dict.items():
            interactions.append(interactions_num)
            algorithm.append(model.__class__.__name__)
            score.append(v)
            metric.append(k)

    interactions = list(map(lambda x: "< {}".format(x), interactions))
    res_df = pd.DataFrame(
        zip(algorithm, score, metric, interactions),
        columns=["algorithm", "score", "metric", "User Group"],
    )

    int_partitions = list(map(lambda x: "< {}".format(x), split_interactions))
    df_bar = pd.DataFrame(
        zip(int_partitions, split_user_count), columns=["User Group", "# Users"]
    )

    SAVE_PATH = f"result_data/{PROJECT}/"
    res_df.to_csv(SAVE_PATH+"PSVD.csv", index=False)
    df_bar.to_csv(SAVE_PATH + "partitions.csv", index=False)

