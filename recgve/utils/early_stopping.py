#!/usr/bin/env python
__author__ = "XXX"
__email__ = "XXX"

import os
from pathlib import Path

import numpy as np
import shutil
from tensorflow import keras


class EarlyStoppingHandlerTensorFlow:
    """
    TensorFlow Early Stopping Handler

    Early stopping handler stop training when the watched metric is not improving and save the best
    model weights

    Attributes:
        patience (int): number of epoch without improvement to wait before stopping the training
            procedure
        save_path (str): path where to save the weights of the best model found
    """

    def __init__(self, patience, save_path):
        """
        TensorFlow Early Stopping Handler

        Early stopping handler stop training when the watched metric is not improving and save the best
        model weights

        Args:
            patience (int): number of epoch without improvement to wait before stopping the training
                procedure
            save_path (str): path where to save the weights of the best model found
        """
        self.patience = patience

        # create the save folder path if it doesn't exist
        Path(save_path).mkdir(exist_ok=True)
        self.save_path = save_path
        # initialize best result dict
        best_result_dict = {"epoch_best_result": 0, "best_result": -np.inf}
        self.best_result_dict = best_result_dict
        self.es_counter = 0

    def update(self, epoch, metric, metric_name, model):
        """
        Update EarlyStopping Handler with training results

        Args:
            epoch (int): current training epoch
            metric (float): current value of tracked metric
            metric_name (str): name of current tracked metric
            model (keras.Model): trained model
        """
        if metric > self.best_result_dict["best_result"]:
            print("New best model found!\n")
            self.best_result_dict["epoch_best_result"] = epoch
            self.best_result_dict["best_result"] = metric
            self.es_counter = 0

            models = os.listdir(self.save_path)
            # if there are files delete them
            if len(models) > 0:
                for filename in os.listdir(self.save_path):
                    file_path = os.path.join(self.save_path, filename)
                    try:
                        if os.path.isfile(file_path) or os.path.islink(file_path):
                            os.unlink(file_path)
                        elif os.path.isdir(file_path):
                            shutil.rmtree(file_path)
                    except Exception as e:
                        print("Failed to delete %s. Reason: %s" % (file_path, e))

            model_name = "epoch_{}_{}_{:.4f}".format(epoch, metric_name, metric)
            model.save_weights(os.path.join(self.save_path, model_name))
        else:
            self.es_counter += 1

    def stop_training(self):
        """Function to call to know whether to stop or not the training procedure"""
        return True if self.es_counter == self.patience else False
