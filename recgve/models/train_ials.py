#!/usr/bin/env python
__author__ = "Edoardo D'Amico"
__email__ = "edoardo.d'amico@insight-centre.org"

import sys

sys.path.append("recgve")
import argparse

import implicit
import numpy as np
import pandas as pd
import tensorflow as tf
from constants import *
from datasets.implemented_datasets import *
from evaluation.topk_evaluator import Evaluator
from representations_based_recommender import RepresentationsBasedRecommender
from scipy.sparse import coo_matrix
from tqdm import tqdm

import wandb


class IALS(RepresentationsBasedRecommender):
    # do something like run puresvd for ials
    # wrapper for the ials from implicit library
    def __init__(
        self, train_data: pd.DataFrame, embeddings_size: int, reg: float, alpha: float
    ):
        RepresentationsBasedRecommender.__init__(self, train_data)
        self.embedding_size = embeddings_size
        self.reg = reg
        self.alpha = alpha

    def fit(self, iterations: int = 500):
        model = implicit.als.AlternatingLeastSquares(
            factors=self.embedding_size,
            regularization=self.reg,
            num_threads=16,
            iterations=iterations,
            alpha=self.alpha,
        )
        row, col, data = (
            self.train_data["userID"],
            self.train_data["itemID"],
            np.ones(len(self.train_data)),
        )
        sps_train = coo_matrix((data, (row, col))).tocsr()

        model.fit(sps_train, show_progress=True)

        self.model = model

    def compute_representations(self, user_data):
        user_interactions = user_data["interactions"]
        user_id = user_interactions[DEFAULT_USER_COL].unique()

        users_emb = self.model.user_factors
        items_emb = self.model.item_factors

        users_repr_df = pd.DataFrame(users_emb, index=user_id)
        items_repr_df = pd.DataFrame(items_emb, index=self.item_idxs)
        return users_repr_df, items_repr_df


if __name__ == "__main__":

    parser = argparse.ArgumentParser("train ials")

    # Model parameters
    parser.add_argument("--embedding_size", type=int, default=64)
    parser.add_argument("--l2_reg", type=float, default=0.0)

    # Train parameters
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--cutoff", type=list, default=[5, 20])

    # Define the Early stopping parameters
    parser.add_argument("--val_every", type=int, default=10)
    parser.add_argument("--early_stopping", type=bool, default=True)
    parser.add_argument("--es_patience", type=int, default=5)
    parser.add_argument("--es_metric", type=str, default="Recall@20")

    # Dataset parameters
    parser.add_argument("--dataset_split", type=str, default="kcore10_stratified")

    # WANDB
    parser.add_argument("--wandb", type=bool, default=False)

    parser.add_argument("--verbose", type=bool, default=True)

    args = vars(parser.parse_args())

    MAX_ITER = 100
    # DATASETS = ["LastFM", "Movielens1M", "AmazonElectronics", "Gowalla"]
    DATASETS = ["Movielens1M"]
    for dataset in DATASETS:
        print(f"Evaluating dataset:{dataset}\n")
        dataset = eval(dataset)()
        dataset_dict = dataset.load_split(args["dataset_split"])

        train_df = dataset_dict["train"]
        val_df = dataset_dict["val"]
        test_df = dataset_dict["test"]

        val_evaluator = Evaluator(
            cutoff_list=args["cutoff"], metrics=["Recall", "NDCG"], test_data=val_df
        )

        # add the model name inside args
        args.update({"recommender_name": "ials"})
        # initialize wandb
        if args["wandb"]:
            wandb.init(config=args)

        best_res = 0
        best_alpha = 0
        best_reg = 0

        # iALS gowalla
        REGS = [0.01, 0.001, 0.0001, 1, 10, 50]
        ALPHAS = [10, 50, 100, 200, 300]

        max_user = train_df["userID"].max()
        max_item = train_df["itemID"].max()

        # pos_int = len(train_df)
        # neg_int = max_user * max_item - len(train_df)
        # neg_weight = pos_int / neg_int

        user_data = {"interactions": train_df}
        for reg in REGS:
            for alpha in ALPHAS:
                # reg = reg * neg_weight
                print(f"Training with reg: {reg} and alpha: {alpha}")
                model = IALS(
                    train_data=train_df, embeddings_size=64, reg=reg, alpha=alpha
                )
                model.fit(iterations=MAX_ITER)
                val_evaluator.evaluate_recommender(model, user_data)
                val_evaluator.print_evaluation_results()
                if val_evaluator.result_dict["Recall@20"] > best_res:
                    best_res = val_evaluator.result_dict["Recall@20"]
                    best_reg = reg
                    best_alpha = alpha

        test_evaluator = Evaluator(
            cutoff_list=args["cutoff"], metrics=["Recall", "NDCG"], test_data=test_df
        )

        train_val = pd.concat([train_df, val_df])
        user_data["interactions"] = train_val
        model = IALS(
            train_data=train_val, embeddings_size=64, reg=best_reg, alpha=best_alpha
        )
        model.fit(iterations=MAX_ITER)

        test_evaluator.evaluate_recommender(model, user_data)
        test_evaluator.print_evaluation_results()
