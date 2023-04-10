#!/usr/bin/env python
__author__ = "XXX"
__email__ = "XXX"

import argparse
import logging
import os
from constants import *
from datasets.dataset_preprocessing import kcore
from datasets.dataset_statistics import get_dataset_stats
from datasets.implemented_datasets import *
from datasets.split_dataset import split_stratified
from utils.general_utils import print_dict

logger = logging.getLogger(__name__)
logging.basicConfig(level=os.environ.get("LOGLEVEL"))

if __name__ == "__main__":
    DATASET = "LastFM"

    parser = argparse.ArgumentParser("Create dataset split")
    parser.add_argument("--dataset", type=str, default=DATASET)
    args = vars(parser.parse_args())

    dataset = eval(args["dataset"])()
    logger.info(f"Creating split for dataset: {DATASET}")

    pd_data = dataset.load()
    pd_data = kcore(pd_data, k=10)
    stats = get_dataset_stats(pd_data)
    print_dict(stats)

    splits_list = split_stratified(pd_data, ratio=[0.8, 0.1, 0.1])
    split_names = ["train", "val", "test"]
    split_dict = dict(zip(split_names, splits_list))
    dataset.save_split(split_dict, split_name="kcore10_stratified")
