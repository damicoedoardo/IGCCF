#!/usr/bin/env python
__author__ = "XXX"
__email__ = "XXX"

import wandb
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

from utils.plot_utils import setup_plot

BASE = "seen_users_percentage_"
PROJECTS = ["lastfm_dat", "ml1m_dat", "Amaz_dat", "gowalla_dat"]
#PROJECTS = ["gowalla_dat"]
METRICS = ["NDCG", "Recall"]
CUTOFFS = [20]

if __name__ == "__main__":

    lastfm = []
    ml1m = []
    amaz = []
    gowalla = []

    for p in PROJECTS:
        api = wandb.Api()
        project = f"{p}_inductive_performance"
        runs_path = f"XXXXXX/{project}"
        runs = api.runs(runs_path)

        alg = []
        kind = []
        seen_users_percentage = []
        metric = []
        score = []
        for r in runs:
            if (
                BASE in r.name
                and "val" not in r.name
                and "uigccf" not in r.name
                and "0.9" in r.name
            ):
                run_name = r.name.split("_")
                for k, v in r.summary.items():
                    if "@" in k:
                        alg.append(run_name[0])
                        m_split = k.split("_")
                        kind.append(str.join("_", m_split[1:]))
                        seen_users_percentage.append(run_name[-1])
                        metric.append(m_split[0])
                        score.append(v)

        res_df = pd.DataFrame(
            zip(kind, map(float, seen_users_percentage), metric, score, alg),
            columns=["kind", "seen_users_percentage", "metric", "score", "alg"],
        )

        for c in CUTOFFS:
            for m in METRICS:
                # rec_grp_df = res_df.groupby("metric").get_group(f"Recall@{c}")
                # ndcg_grp_df = res_df.groupby("metric").get_group(f"NDCG@{c}")
                # grp_df = pd.concat([rec_grp_df, ndcg_grp_df])
                grp_df = res_df.groupby("metric").get_group(f"{m}@{c}")
                for alg in ["igccf"]:
                    print("\n\n\n")
                    print(f"Project: {p}\n")
                    print(f"Algorithm: {alg}\n")
                    print(f"Metric: {m}@{c}\n")
                    print(grp_df[grp_df["alg"]==alg].sort_values("kind"))
