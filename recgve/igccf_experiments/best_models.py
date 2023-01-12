#!/usr/bin/env python
__author__ = "XXX"
__email__ = "XXX"

####################
# LastFM
####################
import wandb
import tensorflow as tf
import shutil

from models.tensorflow.fism import FISM
from models.tensorflow.lightgcn import LightGCN
from models.tensorflow.matrix_factorization_bpr import MatrixFactorizationBPR
from models.tensorflow.ngcf import NGCF
from models.tensorflow.igccf import IGCCF


def get_wandb_project_dict(project_name):
    """Return wandb project dictionary

    Args:
        project_name (str): wandb project name

    Returns:
        dict: dictionary containing best models run_id split_name and dataset_name
    """
    res = None
    if project_name == "lastfm_dat":
        res = {
            "dataset": "LastFM",
            "split_name": "kcore10_stratified",
            "bprmf": "o6a7sngq",
            "ngcf": "zwxafb6r",
            "lrgccf": "qys5xomj",
            "lightgcn": "kn8qq88a",
            "fism": "dxs2avyz",
            "igccf": "guvmz8e2",

        }
    elif project_name == "ml1m_dat":
        res = {
            "dataset": "Movielens1M",
            "split_name": "kcore10_stratified",
            "bprmf": "yzdzx1jm",
            "ngcf": "wxjd3sk9",
            "lightgcn": "2p95cak5",
            "fism": "hitn1f45",
            "igccf": "jfq47wsf"
        }
    elif project_name == "gowalla_dat":
        res = {
            "dataset": "Gowalla",
            "split_name": "kcore10_stratified",
            "bprmf": "oq6rgsmj",
            "ngcf": "svsr8gx5",
            "lightgcn": "0w013uy1",
            "igccf": "j0t03v1w",
            "fism":"j0djegaq"
        }
    elif project_name == "Amaz_dat":
        res = {
            "dataset": "AmazonElectronics",
            "split_name": "kcore10_stratified",
            "bprmf": "hxrtbycw",
            "ngcf": "kj51vh0g",
            "lightgcn": "zt6ccusa",
            "fism": "c0gbd3hz",
            "igccf": "4janl10s"
        }
    return res


def get_wandb_project_dict_trainval(project_name):
    """Return wandb project dictionary

    Args:
        project_name (str): wandb project name

    Returns:
        dict: dictionary containing best models run_id split_name and dataset_name
    """
    res = None
    if project_name == "lastfm_dat":
        res = {
            "dataset": "LastFM",
            "split_name": "kcore10_stratified",
            "bprmf": "5y7kbwzv",
            "ngcf": "35osjirr",
            "lightgcn": "26f36lhh",
            "fism": "2rla3ydr",
            "igccf": "2kaj7rzf"
        }
    elif project_name == "ml1m_dat":
        res = {
            "dataset": "Movielens1M",
            "split_name": "kcore10_stratified",
            "bprmf": "2s4kujlh",
            "ngcf": "c11jz64c",
            "lightgcn": "2msa1gg0",
            "fism": "3lszlqdx",
            "igccf": "7ihs7iij",
        }
    elif project_name == "gowalla_dat":
        res = {
            "dataset": "Gowalla",
            "split_name": "kcore10_stratified",
            "bprmf": "1zadmepn",
            "ngcf": "xuokx3uw",
            "lightgcn": "1kzxf0wp",
            "igccf": "1mhblrhy",
            "fism": "ygilgt1o",
        }
    elif project_name == "Amaz_dat":
        res = {
            "dataset": "AmazonElectronics",
            "split_name": "kcore10_stratified",
            "bprmf": "1g8l4y1w",
            "ngcf": "js3n9r7l",
            "lightgcn": "8nayc55z",
            "fism":"100fl679",
            "igccf": "npr4reio"
        }
    return res


def restore_models(model_list, project_name, train_df):
    """Restore wandb model
    Args:
        model_list (list): list of models name
        project_name (str): wandb project name

    Returns:
        list: list of restored model
    """
    wandb_project_dict = get_wandb_project_dict_trainval(project_name)
    models = []
    for m in model_list:
        if not m in wandb_project_dict:
            raise ValueError(f"Model {m} not in project!")

        api = wandb.Api()

        run_identifier = "XXXXXX/{}/{}".format(
            project_name, wandb_project_dict[m]
        )
        run_object = api.run(run_identifier)

        for f in run_object.files():
            if "best_models" in str(f):
                f.download(replace=True)

        run_parameters_dict = run_object.config
        if m == "fism":
            model = FISM(
                train_df,
                embeddings_size=run_parameters_dict["embedding_size"],
                alpha=run_parameters_dict["alpha"],
            )
        elif m == "igccf":
            model = IGCCF(
                train_df,
                embeddings_size=run_parameters_dict["embedding_size"],
                convolution_depth=run_parameters_dict["convolution_depth"],
                user_profile_dropout=run_parameters_dict["user_profile_dropout"],
                top_k=run_parameters_dict["top_k"]
            )
        elif m == "lightgcn":
            model = LightGCN(
                train_df,
                embeddings_size=run_parameters_dict["embedding_size"],
                convolution_depth=run_parameters_dict["convolution_depth"],
            )
        elif m == "ngcf":
            model = NGCF(
                train_df,
                embeddings_size=run_parameters_dict["embedding_size"],
                convolution_depth=run_parameters_dict["convolution_depth"],
                mess_dropout=run_parameters_dict["mess_dropout"],
                node_dropout=run_parameters_dict["node_dropout"],
            )
        elif m == "bprmf":
            model = MatrixFactorizationBPR(
                train_df, embeddings_size=run_parameters_dict["embedding_size"]
            )
        else:
            raise ValueError("No model found")

        weights_path = "best_models"
        latest = tf.train.latest_checkpoint(weights_path)
        model.load_weights(latest)

        # delete the downloaded weights files
        print("Deleting restored files from wandb")
        shutil.rmtree("best_models")

        models.append(model)
    return models
