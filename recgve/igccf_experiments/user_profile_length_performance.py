#!/usr/bin/env python
__author__ = "XXX"
__email__ = "XXX"

# Define static variables
import shutil
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

from utils.plot_utils import setup_plot

PROJECT = "ml1m_dat"
MODELS = ["igccf", "lightgcn", "bprmf", "ngcf", "fism"]
#MODELS = ["igcmf", "fism"]
METRICS = ["Recall", "NDCG"]
CUTOFFS = [5, 20]
N = 4


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


if __name__ == "__main__":
    setup_plot(241, style_sheet="upl_style", fig_ratio=0.9)

    parser = argparse.ArgumentParser()
    ##########################################
    # identifier of WANDB run
    ##########################################
    parser.add_argument("--wandb_project", type=str, default=PROJECT)

    args = vars(parser.parse_args())
    wandb_project_dict = get_wandb_project_dict(args["wandb_project"])

    ##########################################
    # Retrieve run parameters
    ##########################################
    # api = wandb.Api()
    # run_identifier = "XXXXXX/{}/{}".format(
    #     args["wandb_project"], wandb_project_dict["igccf"]
    # )
    # run_object = api.run(run_identifier)
    #
    # for f in run_object.files():
    #     if "best_models" in str(f):
    #         f.download(replace=True)
    #
    # run_parameters_dict = run_object.config

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

    ##########################################
    # Load models
    ##########################################
    restored_models = restore_models(MODELS, PROJECT, train_df)

    train_dfs, split_interactions, split_user_count = profile_length_splitter(
        train_df, n=N
    )

    ##########################################
    # retrieve Results
    ##########################################
    algorithm = []
    score = []
    metric = []
    interactions = []

    for model in restored_models:
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
    res_df.to_csv(f"result_data/{PROJECT}/all.csv", index=False)

    colors = {
        "LightGCN": "orange",
        "MatrixFactorizationBPR": "green",
        "NGCF": "red",
        "FISM": "black",
        "IGCCF":"purple"
    }

    for m in METRICS:
        for c in CUTOFFS:
            # make BARPLOT
            int_partitions = list(map(lambda x: "< {}".format(x), split_interactions))
            df_bar = pd.DataFrame(
                zip(int_partitions, split_user_count), columns=["User Group", "# Users"]
            )

            fix, ax = plt.subplots()
            ax.xaxis.grid(False)
            ax1 = ax.twinx()
            ax1.grid(False)
            ax1.set_xlabel("User Group")
            ax1.set_ylabel("#Users/1e3")

            grp_df = res_df.groupby("metric").get_group(f"{m}@{c}")
            sns.lineplot(
                data=grp_df,
                x="User Group",
                y="score",
                hue="algorithm",
                style="algorithm",
                palette=colors,
                ax=ax1,
                sort=False,
                markers=True,
            )

            df_bar["# Users"]/=1000
            sns.barplot(
                x="User Group",
                y="# Users",
                data=df_bar,
                color="royalblue",
                alpha=0.3,
                ax=ax,
            )

            if PROJECT != "lastfm_dat":
                ax.set_ylabel("")
                ax1.set_ylabel("")
                ax1.get_legend().remove()
            else:
                ax1.legend_.set_title(None)
                for t in ax1.legend_.texts:
                    if t.get_text() == "MatrixFactorizationBPR":
                        t.set_text("BPRMF")
                        t.set_text(t.get_text().replace("_", " ").replace("valid", ""))
                    elif t.get_text() == "UIGCCF":
                        t.set_text("IGCCF")
                #ax1.set_ylabel(f"{m}@{c}")

            ax1.tick_params(direction="in")
            ax.tick_params(direction="in")




            ax.set_ylabel("#Users/1e3") if PROJECT == "lastfm_dat" else plt.gca().set_ylabel("")
            ax1.set_ylabel(f"{m}@{c}") if PROJECT == "gowalla_dat" else plt.gca().set_ylabel("")


            plt.tight_layout(pad=0.05)
            TITLE = f"{m}@{c}_User_Profile_length_performance"
            plt.savefig(
                "{}/Desktop/{}_{}.pdf".format(os.environ["HOME"], PROJECT, TITLE)
            )
            plt.show()
