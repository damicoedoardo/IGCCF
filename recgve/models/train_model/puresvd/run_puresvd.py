#!/usr/bin/env python
__author__ = "XXX"
__email__ = "XXX"

from datasets.dataset_statistics import get_dataset_concentration
from evaluation.topk_evaluator import Evaluator
from models.ease import EASE
from models.puresvd import PureSVD
from similarity_matrix_recommender import ItemKnn
from datasets.implemented_datasets import *
import pandas as pd

if __name__ == '__main__':
    #@DATASET = "Gowalla"
    DATASETS = ['LastFM', 'Movielens1M', 'AmazonElectronics', "Gowalla"]
    SPLIT_NAME = "kcore10_stratified"
    CUTOFF = [5, 20]

    for d in DATASETS:
        dataset_dict = eval(d)().load_split(SPLIT_NAME)

        train_df = dataset_dict["train"]
        user_data = {"interactions": train_df}
        val_df = dataset_dict["val"]
        test_df = dataset_dict["test"]

        # compute dataset statistics
        full_data = pd.concat([train_df, val_df, test_df])
        print(get_dataset_concentration(full_data))

        # val_evaluator = Evaluator(
        #     cutoff_list=CUTOFF, metrics=["Recall", "NDCG"], test_data=val_df
        # )
        #
        # test_evaluator = Evaluator(
        #     cutoff_list=CUTOFF, metrics=["Recall", "NDCG"], test_data=test_df
        # )
        #
        # best_components = 0
        # best_score = 0
        # for n_components in [5, 25, 50, 100]:
        #     model = PureSVD(train_df, n_components=n_components)
        #     print(f"n_components:{n_components}")
        #     print(f"val")
        #     val_evaluator.evaluate_recommender(model, user_data)
        #     val_evaluator.print_evaluation_results()
        #
        #     if val_evaluator.result_dict["Recall@20"] > best_score:
        #         best_score = val_evaluator.result_dict["Recall@20"]
        #         best_components = n_components
        #
        # train_df = pd.concat([train_df, val_df])
        # user_data = {"interactions": train_df}
        # model = PureSVD(train_df, n_components=best_components)
        # test_evaluator.evaluate_recommender(model, user_data)
        # print(f"dataset:{DATASET}, n_components:{best_components}")
        # test_evaluator.print_evaluation_results()
