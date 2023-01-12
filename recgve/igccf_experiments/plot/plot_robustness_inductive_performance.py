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
# PROJECTS = ["gowalla_dat"]
METRICS = ["Recall", "NDCG"]
CUTOFFS = [5, 20]

if __name__ == "__main__":
    setup_plot(241, fig_ratio=0.8, style_sheet="inductive_performance")

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
                and "fism" not in r.name
                and "PureSVD" not in r.name
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

        line_style = {
            "seen_users": [1, 0],
            "unseen_users": [1, 0.7],
        }

        colors = {
            "seen_users": "darkblue",
            "unseen_users": "orange",
        }

        markers = {
            "seen_users": "o",
            "unseen_users": "^",
        }

        for c in CUTOFFS:
            # rec_grp_df = res_df.groupby("metric").get_group(f"Recall@{c}")
            # ndcg_grp_df = res_df.groupby("metric").get_group(f"NDCG@{c}")
            # grp_df = pd.concat([rec_grp_df, ndcg_grp_df])
            grp_df = res_df.groupby("metric").get_group(f"NDCG@{c}")
            fix, ax = plt.subplots()

            # sns.set_style("whitegrid")
            sns.lineplot(
                data=grp_df,
                x="seen_users_percentage",
                y="score",
                hue="kind",
                sort=True,
                marker="o",
                dashes=line_style,
                style="kind",
                palette=colors,
                markers=markers,
            )
            # ax.set_ylabel(f"{m}@{c}")
            ax.tick_params(direction="in")
            ax.set_ylabel(f"NDCG@{c}")
            ax.set_xlabel("%Seen users")

            ax.legend().set_title("")
            for t in ax.legend_.texts:
                t.set_text(t.get_text().replace("_", " "))

            # handles, labels = ax.get_legend_handles_labels()
            # ax.legend(
            #     handles=handles[1:],
            #     labels=labels[1:],
            # )
            plt.xticks([0.1, 0.3, 0.5, 0.7, 0.9])
            plt.tight_layout(pad=0.05)
            plt.gca().xaxis.grid(False)

            if p != "lastfm_dat":
                ax.set_ylabel("")
                ax.get_legend().remove()

            TITLE = f"unseen_users_performance@{c}"
            print(TITLE)
            plt.savefig(
                "{}/Desktop/{}_{}.pdf".format(os.environ["HOME"], p, TITLE),
                # bbox_inches="tight",
            )
            plt.show()
