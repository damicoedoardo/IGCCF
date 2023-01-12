#!/usr/bin/env python
__author__ = "XXX"
__email__ = "XXX"

from timeit import timeit
from typing import Union
import numpy as np
from recommender_interface import Recommender
from constants import *
import os
import logging
from evaluation.python_evaluation import *
from utils.decorator_utils import timing

logger = logging.getLogger(__name__)
logging.basicConfig(level=os.environ.get("LOGLEVEL"))


class Evaluator:
    """
    Top-K recommendation task Evaluator

    Evaluate a recommender algorithm according to specified metrics and cutoff

    Attributes:
        cutoff_list (list): list of cutoff used to retrieve the recommendations
        metrics (list): list of metrics to evaluate
        test_data (pd.DataFrame): ground truth data
    """

    def __init__(self, cutoff_list, metrics, test_data):
        """
        Top-K recommendation task Evaluator

        Evaluate a recommender algorithm according to specified metrics and cutoff

        Attributes:
            cutoff_list (list): list of cutoff used to retrieve the recommendations
            metrics (list): list of metrics to evaluate
            test_data (pd.DataFrame): ground truth data
        """
        self.test_data = self._check_test_data(test_data)

        self._check_metrics(metrics)
        self.metrics = metrics

        self.cutoff_list = cutoff_list

        # set by _check_cutoff
        self.max_cutoff = max(cutoff_list)
        self.test_data = test_data

        self.result_dict = {}
        self.recommender_name = None

    def _check_test_data(self, test_data):
        if DEFAULT_RATING_COL not in test_data:
            logger.info("Adding rating = 1, considering implicit recommendation task")
            test_data[DEFAULT_RATING_COL] = 1
        return test_data

    def _check_metrics(self, metrics):
        for m in metrics:
            if m not in metrics_dict:
                raise ValueError(
                    f"metric: {m} not available \n Available metrics: {metrics_dict.keys()}"
                )

    def evaluate_recommender(self, recommender, user_data):
        """Evaluate a recommendation system algorithm

        Args:
            recommender: algorithm to evaulate
            user_data (dict):dictionary containing inside `interactions` the interactions of the users for which
                retrieve predictions stored inside a pd.DataFrame
        """
        assert issubclass(
            recommender.__class__, Recommender
        ), "recommender passed is not extending class: {}".format(Recommender)

        # todo: check kwargs for recommend arguments
        # retrieve the users idxs for which retrieve predictions
        recommendation = recommender.recommend(
            cutoff=self.max_cutoff, user_data=user_data
        )

        for m in self.metrics:
            for c in self.cutoff_list:
                # keep recommendations up to c
                recs = recommendation[recommendation["item_rank"] <= c]
                metric_value = metrics_dict[m](
                    rating_pred=recs, rating_true=self.test_data, relevancy_method=None,
                )

                # update result dict
                self.result_dict[f"{m}@{c}"] = metric_value

    def print_evaluation_results(self):
        """ Print evaluation results """
        print("=== RESULTS ===\n")
        for k, v in self.result_dict.items():
            print("{}: {}".format(k, v))
        print("")
