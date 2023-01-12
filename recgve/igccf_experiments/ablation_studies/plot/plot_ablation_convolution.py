#!/usr/bin/env python
__author__ = "XXX XXX"
__email__ = "XXX"

import wandb
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from matplotlib import cm
from matplotlib.ticker import FormatStrFormatter

from utils.plot_utils import setup_plot

BASE_CD = "igccf_conv_depth_"
METRICS = ["NDCG"]
CUTOFFS = [20]
PROJECTS = ["lastfm_dat", "ml1m_dat", "Amaz_dat", "gowalla_dat"]

if __name__ == "__main__":
    setup_plot(241, fig_ratio=0.6, style_sheet="ablation_convolution")
    for p in PROJECTS:
        api = wandb.Api()
        project = f"ablation_{p}"
        runs_path = f"XXXXXX/{project}"
        runs = api.runs(runs_path)

        kind = []
        convolution_depth = []
        metric = []
        score = []
        for r in runs:
            if BASE_CD in r.name:
                run_name = r.name.split("_")
                for k, v in r.summary.items():
                    if "@" in k:
                        kind.append(str.join("_", run_name[:-1]))
                        convolution_depth.append(run_name[-1])
                        metric.append(k)
                        score.append(v)

        res_df = pd.DataFrame(
            zip(kind, convolution_depth, metric, score),
            columns=["kind", "convolution_depth", "metric", "score"],
        )

        colors = {
            "valid_conv_depth": "orange",
        }

        for m in METRICS:
            for c in CUTOFFS:
                grp_df = res_df.groupby("metric").get_group(f"{m}@{c}")
                fix, ax = plt.subplots()

                palette = sns.color_palette("Blues_d", n_colors=4)
                #palette = sns.color_palette(palette="flare")
                palette.reverse()

                sns.barplot(
                    data=grp_df,
                    x="convolution_depth",
                    y="score",
                    hue="convolution_depth",
                    order=["0", "1", "2", "3"],
                    dodge=False,
                    #dashes=line_style,
                    #palette=sns.color_palette(palette="flare"),
                    palette=palette,
                    alpha=1
                )

                score_conv_0 = grp_df[grp_df["convolution_depth"] == "0"]["score"].values[0]
                score_conv_1 = grp_df[grp_df["convolution_depth"] == "1"]["score"].values[0]

                print(p)
                print(f"{m}@{c}")
                print(1-(score_conv_0/score_conv_1))
                # x_coordinates = [0, 3]
                # y_coordinates = [score_conv_0, score_conv_0]
                # plt.plot(x_coordinates, y_coordinates)

                ax.axhline(score_conv_0, ls="--", color="black", linewidth=1.5)

                if p not in ["lastfm_dat", "Amaz_dat"]:
                     ax.set_ylabel("")

                # if p not in ["Amaz_dat", "gowalla_dat"]:
                #     ax.set_xlabel("")

                # yaxis fomatter
                #plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

                if p == "lastfm_dat":
                    plt.ylim(0.25, 0.325)
                    # plt.vlines(1, score_conv_0, score_conv_1, colors="yellow")
                    # label = plt.text(1.1, score_conv_1, "+X%")
                    # label.set_bbox(dict(facecolor='yellow', alpha=0.5, edgecolor='yellow'))

                elif p == "ml1m_dat":
                    plt.ylim(0.31, 0.35)
                elif p == "Amaz_dat":
                    plt.ylim(0.03, 0.054)
                elif p == "gowalla_dat":
                    plt.ylim(0.08, 0.14)

                plt.tight_layout(pad=0.05)
                plt.gca().xaxis.grid(False)
                #plt.gca().yaxis.grid(False)
                ax.tick_params(direction="in")
                ax.set_ylabel(f"{m}@{c}")
                ax.set_xlabel("Convolution depth")
                ax.legend_.remove()
                TITLE = f"ablation_convolution_depth_{m}@{c}"
                print(TITLE)
                plt.savefig("{}/Desktop/{}_{}.pdf".format(os.environ["HOME"], p, TITLE))
                plt.show()
