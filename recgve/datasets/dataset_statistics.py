#!/usr/bin/env python
__author__ = "XXX"
__email__ = "XXX"

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
from constants import *

# https://github.com/statisticianinstilettos/recmetrics
from utils.pandas_utils import has_columns


def long_tail_plot(
    df,
    item_id_column=DEFAULT_ITEM_COL,
    interaction_type="interaction",
    percentage=0.33,
    x_labels=False,
):
    """Plots the long tail for a user-item interaction dataset.

    Args:
        df (pd.DataFrame):
        item_id_column (str): column name identifying the item ids in the dataframe
        interaction_type (str): type of user-item interactions, i.e. 'purchases', 'ratings' 'interactions', or 'clicks'
        percentage (float): percent of volume to consider as the head (percent as a decimal)
        x_labels (bool): if True, plot x-axis tick labels, if False, no x-axis tick lavels will be plotted.
    """
    # calculate cumulative volumes
    volume_df = df[item_id_column].value_counts().reset_index()
    volume_df.columns = [item_id_column, "volume"]
    volume_df[item_id_column] = volume_df[item_id_column].astype(str)
    volume_df["cumulative_volume"] = volume_df["volume"].cumsum()
    volume_df["percent_of_total_volume"] = (
        volume_df["cumulative_volume"] / volume_df["volume"].sum()
    )

    # line plot of cumulative volume
    x = range(0, len(volume_df))
    ax = sns.lineplot(x, y="volume", data=volume_df, color="black")
    plt.xticks(x)

    # set labels
    ax.set_title("Long Tail Plot")
    ax.set_ylabel("# of " + interaction_type)
    ax.set_xlabel(item_id_column)

    if percentage != None:
        # plot vertical line at the tail location
        head = volume_df[volume_df["percent_of_total_volume"] <= percentage]
        tail = volume_df[volume_df["percent_of_total_volume"] > percentage]
        items_in_head = len(head)
        items_in_tail = len(tail)
        plt.axvline(x=items_in_head, color="red", linestyle="--")

        # fill area under plot
        head = head.append(tail.head(1))
        x1 = head.index.values
        y1 = head["volume"]
        x2 = tail.index.values
        y2 = tail["volume"]
        ax.fill_between(x1, y1, color="blue", alpha=0.2)
        ax.fill_between(x2, y2, color="orange", alpha=0.2)

        # create legend
        legend_elements = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label=str(items_in_head) + ": items in the head",
                markerfacecolor="blue",
                markersize=5,
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label=str(items_in_tail) + ": items in the tail",
                markerfacecolor="orange",
                markersize=5,
            ),
        ]
        ax.legend(handles=legend_elements, loc=1)

    else:
        x1 = volume_df[item_id_column]
        y1 = volume_df["volume"]
        ax.fill_between(x1, y1, color="blue", alpha=0.3)
    if x_labels == False:
        plt.xticks([], [])
        ax.set(xticklabels=[])
    else:
        ax.set_xticklabels(labels=volume_df[item_id_column], rotation=45, ha="right")

    plt.show()


def get_dataset_concentration(df, N=0.33):
    """Interactions associated to the top-N most popular item

    Args:
        df: user-item interaction df
        N (float): percentage of top popular items to consider when computing concentration
    Returns:
        float: concentration of the dataset
    """
    item_count = df.groupby('itemID').count().sort_values('userID', ascending=False)
    n_items = len(item_count)
    top_n_items_num = round(n_items*N)
    concentration_interactions = item_count.head(top_n_items_num)['userID'].values.sum()
    total_interactions = len(df)
    concentration = concentration_interactions/total_interactions
    return concentration


def get_dataset_stats(df):
    """Print general statistics of the dataset

    Args:
        df (pd.DataFrame): dataset

    Returns:
        dict: dictionary with dataset statistics
    """
    assert has_columns(df, columns=[DEFAULT_USER_COL, DEFAULT_ITEM_COL])

    stats = {}
    users = df[DEFAULT_USER_COL].unique()
    items = df[DEFAULT_ITEM_COL].unique()
    stats["# users"] = len(users)
    stats["# items"] = len(items)
    stats["# interactions"] = len(df)
    stats["density"] = round(len(df) / (len(users) * len(items)), 4)

    return stats
