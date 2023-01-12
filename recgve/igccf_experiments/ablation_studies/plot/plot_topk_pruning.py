#!/usr/bin/env python
__author__ = "XXX"
__email__ = "XXX"

import wandb
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import matplotlib.ticker as mticker
from utils.plot_utils import setup_plot

BASE_KP = "igccf_top_k_"
#PROJECTS = ["Amaz_dat"]
PROJECTS = ["lastfm_dat", "ml1m_dat", "Amaz_dat", "gowalla_dat"]
METRICS = ["Recall", "NDCG"]
CUTOFFS = [5, 20]

if __name__ == "__main__":
    for p in PROJECTS:
        setup_plot(241, fig_ratio=0.8, style_sheet="ablation_topk")
        api = wandb.Api()
        project = f"ablation_{p}"
        runs_path = f"XXXXXX/{project}"
        runs = api.runs(runs_path)

        kind = []
        topk = []
        metric = []
        score = []
        runtimes = []
        for r in runs:
            if BASE_KP in r.name:
                run_name = r.name.split("_")
                for k, v in r.summary.items():
                    if "@" in k:
                        kind.append(str.join("_", run_name[:-1]))
                        topk.append(run_name[-1])
                        metric.append(k)
                        score.append(v)
                        runtimes.append(r.summary["_runtime"])


        res_df = pd.DataFrame(
            zip(kind, topk, metric, score, runtimes),
            columns=["kind", "top_k", "metric", "score", "run time"],
        )

        for m in METRICS:
            for c in CUTOFFS:
                grp_df = res_df.groupby("metric").get_group(f"{m}@{c}")

                #sns.set_style("whitegrid", {"axes.grid": False, "axes.axisbelow": True})
                fix, ax = plt.subplots()
                ax2 = ax
                ax = ax.twinx()

                # if p != "lastfm_dat":
                #     f = mticker.ScalarFormatter(useOffset=False, useMathText=True)
                #     g = lambda x, pos: "${}$".format(f._formatSciNotation('%1.10e' % x))
                #     plt.gca().yaxis.set_major_formatter(mticker.FuncFormatter(g))

                grp_df["run time"] /= 100
                sns.barplot(
                    x="top_k",
                    y="run time",
                    data=grp_df,
                    color="royalblue",
                    alpha=0.3,
                    ax=ax,
                    order = ["5", "20", "50", "100"]
                    #order = [5, 15, 30, 50, 100, 250]
                    #sort=True,
                )
                plt.gca().yaxis.grid(False)
                #plt.show()
                #sns.set_style("whitegrid")
                #ax2 = ax

                colors = {
                    "igccf_top_k": "darkblue",
                    "NDCG@20": "orange",
                    "Recall@5": "darkblue",
                    "NDCG@5": "orange",
                }

                sorter = ["5", "20", "50", "100"]
                grp_df.top_k = grp_df.top_k.astype("category")
                grp_df.top_k.cat.set_categories(sorter, inplace=True)
                sns.lineplot(
                    data=grp_df,
                    x="top_k",
                    y="score",
                    hue="kind",
                    sort=False,
                    marker="o",
                    ax=ax2,
                    palette=colors
                    #order = ["5", "15", "30", "50", "100", "250"]
                    #dashes=line_style,
                    #style="kind",
                    #palette=colors
                )
                ax.tick_params(direction="in")
                ax2.tick_params(direction="in")
                ax2.set_ylabel(f"{m}@{c}")
                ax2.set_xlabel("Top-K")
                ax2.legend_.remove()
                ax2.xaxis.grid(False)

                if p != "lastfm_dat" and p != "Amaz_dat":
                    ax.set_ylabel("")
                    ax2.set_ylabel("")

                ax.set_ylabel("Run time s/1e3") if p == "gowalla_dat" else plt.gca().set_ylabel("")
                plt.tight_layout(pad=0.8)
                TITLE = f"ablation_topk_{m}@{c}"
                print(TITLE)
                plt.savefig("{}/Desktop/{}_{}.pdf".format(os.environ["HOME"], p, TITLE))
                plt.show()
