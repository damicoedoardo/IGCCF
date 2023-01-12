
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from utils.plot_utils import setup_plot

PROJECT = "ml1m_dat"
METRICS = ["Recall", "NDCG"]
CUTOFFS = [5, 20]
colors = {
        "LightGCN": "orange",
        "MatrixFactorizationBPR": "green",
        "NGCF": "red",
        "FISM": "black",
        "IGCCF":"purple",
        "PureSVD":"blue",
    }
LOAD_PATH = f"../result_data/{PROJECT}/"

if __name__ == '__main__':
    setup_plot(241, style_sheet="upl_style", fig_ratio=0.9)
    res_df_all = pd.read_csv(LOAD_PATH+"all.csv")
    res_df_SVD = pd.read_csv(LOAD_PATH+"PSVD.csv")
    res_df = pd.concat([res_df_all, res_df_SVD], axis=0)

    df_bar = pd.read_csv(LOAD_PATH+"partitions.csv")
    df_bar["# Users"] /= 1000

    for m in METRICS:
        for c in CUTOFFS:

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
                hue_order=["LightGCN", "MatrixFactorizationBPR", "NGCF", "FISM", "PureSVD", "IGCCF"],
                style="algorithm",
                palette=colors,
                ax=ax1,
                sort=False,
                markers=True,
            )

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
                    elif t.get_text() == "IGCCF":
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