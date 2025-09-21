from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parents[2]))

import argparse
import copy
import datetime
import logging
import os
import random

import numpy as np
from ray.tune import ResultGrid
from ray.tune.search.hyperopt import HyperOptSearch
from ray.tune.schedulers import FIFOScheduler
from ray.train import RunConfig
import torch
from torch.utils.tensorboard import SummaryWriter

from cids.util import misc_funcs as misc, config as cfgutil
from cids.training.unsupervised import train_mlp_autoencoder, train_mlp_autoencoder_ssl
from cids.util import metrics
import util

def train_unsupervised(hp_config: dict, ds_config: dict, experiment_config: dict):

    trainable = train_mlp_autoencoder

    model_path = os.path.join(misc.root(), f"models/03_semisupervision/unsupervised/{ds_config["ds_name"]}/{experiment_config["model_type"]}/{experiment_config["id"]}/{experiment_config["trial"]}/ckpt.pt")
    tensorboard_path = os.path.join(misc.root(), f"results/03_semisupervision/unsupervised/{ds_config["ds_name"]}/{experiment_config["model_type"]}/{experiment_config["id"]}/{experiment_config["trial"]}/tensorboard/")
    writer = SummaryWriter(tensorboard_path)

    os.makedirs(os.path.split(model_path)[0], exist_ok=True)
    os.makedirs(tensorboard_path, exist_ok=True)

    ds_name = ds_config["ds_name"]
    eval_ds_config = copy.deepcopy(ds_config["eval"])
    ds_config = copy.deepcopy(ds_config["train"])
    md_train_config =copy.deepcopy(hp_config)

    trainable_args = {
        "model_config": md_train_config.pop("model"),
        "ds_name": ds_name, 
        "ds_args": ds_config,
        "eval_ds_args": eval_ds_config,
        "eval_metric": metrics.auc_score, 
        "checkpoint_path": model_path,
        "tensorboard_writer": writer,
        "device": experiment_config["device"]
    }

    trainable_args = trainable_args | md_train_config
    trainable(**trainable_args)
    return model_path

def finetune_supervised(hp_config, ds_config, experiment_config):

    trainable = train_mlp_autoencoder_ssl

    model_path = os.path.join(misc.root(), experiment_config["model_path"], "ckpt.pt")
    tensorboard_path = os.path.join(misc.root(), f"results/03_semisupervision/finetuned/{ds_config["ds_name"]}/{experiment_config["model_type"]}/{experiment_config["id"]}/{experiment_config["trial"]}/tensorboard/")
    ckpt_path =  os.path.join(misc.root(), f"models/03_semisupervision/finetuned/{ds_config["ds_name"]}/{experiment_config["model_type"]}/{experiment_config["id"]}/{experiment_config["trial"]}/ckpt.pt")
    writer = SummaryWriter(tensorboard_path)
    os.makedirs(tensorboard_path, exist_ok=True)

    ds_name = ds_config["ds_name"]
    eval_ds_config = copy.deepcopy(ds_config["eval"])
    ds_config = copy.deepcopy(ds_config["train"])
    md_train_config =copy.deepcopy(hp_config)
    md_train_config["lr"] *= 0.5
    if "patience" in md_train_config:
        md_train_config.pop("patience")

    trainable_args = {
        "model_config": md_train_config.pop("model"),
        "model_path":model_path,
        "ds_name": ds_name, 
        "ds_args": ds_config,
        "eval_ds_args": eval_ds_config,
        "eval_metric": metrics.auc_score, 
        "checkpoint_path":ckpt_path,
        "tensorboard_writer": writer,
        "device": experiment_config["device"],
        "loss_type": experiment_config["loss_type"],
    }

    trainable_args = trainable_args | md_train_config
    trainable(**trainable_args)
    return model_path

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train full pipeline for semi-supervised CIDS training")

    parser.add_argument("--config", type=str, required=True, help="Path to the config file")
    parser.add_argument("--nopretrain", action="store_true", help="Do not perform unsupervised pretraining.")

    args = parser.parse_args()
    config_path = args.config 

    log_dir = os.path.join(misc.root(), "logs/03_semisupervision/")
    os.makedirs(log_dir, exist_ok=True)
    logger = util.setup_logger(os.path.join(log_dir , f"ssl.log"), level=logging.INFO)

    logger.info("Load config file")

    config = cfgutil.read(args.config, convert_tuple=False)

    torch.manual_seed(config["experiment"]["seed"])
    np.random.seed(config["experiment"]["seed"])
    random.seed(config["experiment"]["seed"])

    assert config["experiment"]["loss_type"] in ["inverse", "reciprocal", "deepsad"], "Loss type must be either inverse, reciprocal or deepsad"

    epochs_pretraining = config["hp_config"].pop("epochs_pretraining")
    epochs_finetuning = config["hp_config"].pop("epochs_finetuning")
    gamma = config["hp_config"].pop("gamma")
    # first performe unsupervised pretraining for the model
    if not args.nopretrain:
        logger.info("Perform unsupervised pretraining")
        config["hp_config"]["epochs"] = epochs_pretraining

        ds_config = {k: v for k, v in config["dataset"].items() if "supervised" not in k}
        ds_config.update(config["dataset"]["unsupervised"])

        path = train_unsupervised(
            hp_config=config["hp_config"],
            ds_config=ds_config,
            experiment_config=config["experiment"]
        )
        logger.info("Done with unsupervised pretraining")
        config["experiment"]["model_path"] = path
    else:
        logger.info("Skip unsupervised pretraining")
    # now perform supervised training
    config["hp_config"]["gamma"] = gamma
    assert config["experiment"]["model_path"] is not None, "Model path must be set for supervised training"
    logger.info("Perform supervised finetuning")
    config["hp_config"]["epochs"] = epochs_finetuning
    ds_config = {k: v for k, v in config["dataset"].items() if "supervised" not in k}
    ds_config.update(config["dataset"]["supervised"])

    finetune_supervised(hp_config=config["hp_config"],
                        ds_config=ds_config,
                        experiment_config=config["experiment"])
    logger.info("Done with supervised finetuning")

