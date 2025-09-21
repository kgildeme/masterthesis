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
from cids.training.unsupervised import train_mlp_autoencoder, train_mlp_autoencoder_ssl, train_deep_sad
from cids.util import metrics
import util

def finetune_supervised(hp_config, ds_config, experiment_config):

    trainable = train_mlp_autoencoder_ssl if experiment_config["loss_type"] != "deepsad" else train_deep_sad


    model_path = os.path.join(misc.root(), experiment_config["model_path"], "ckpt.pt")
    tensorboard_path = os.path.join(misc.root(), f"results/03_semisupervision/finetuned/{experiment_config["model_type"]}/{experiment_config["id"]}/{experiment_config["loss_type"]}--gamma{hp_config["gamma"]}--lr{hp_config["lr"]}/seed{experiment_config["seed"]}/tensorboard/")
    ckpt_path =  os.path.join(misc.root(), f"models/03_semisupervision/finetuned/{experiment_config["model_type"]}/{experiment_config["id"]}/{experiment_config["loss_type"]}--gamma{hp_config["gamma"]}--lr{hp_config["lr"]}/seed{experiment_config["seed"]}/ckpt.pt")
    writer = SummaryWriter(tensorboard_path)
    os.makedirs(tensorboard_path, exist_ok=True)

    md_train_config =copy.deepcopy(hp_config)

    ds_name = ds_config["ds_name"]
    if experiment_config["loss_type"] != "deepsad":
        eval_ds_config = copy.deepcopy(ds_config["eval"])
        ds_config = copy.deepcopy(ds_config["train"])
    else:
        eval_ds_config = copy.deepcopy(ds_config["supervised"]["eval"])
        pretrain_ds_config = copy.deepcopy(ds_config["unsupervised"]["train"])
        ds_config = copy.deepcopy(ds_config["supervised"]["train"])

    trainable_args = {
        "model_config": md_train_config.pop("model"),
        "model_path":model_path,
        "ds_name": ds_name, 
        "ds_args": ds_config,
        "eval_ds_args": eval_ds_config,
        "checkpoint_path":ckpt_path,
        "tensorboard_writer": writer,
        "device": experiment_config["device"],
        "loss_type": experiment_config["loss_type"],
    }
    if experiment_config["loss_type"] == "deepsad":
        trainable_args["pretrain_ds_args"] = pretrain_ds_config
        trainable_args.pop("loss_type")

    trainable_args = trainable_args | md_train_config
    trainable(**trainable_args)
    return model_path

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train full pipeline for semi-supervised CIDS training")

    parser.add_argument("--config", type=str, required=True, help="Path to the config file")
    parser.add_argument("--resume", action="store_true")

    args = parser.parse_args()
    config_path = args.config 

    log_dir = os.path.join(misc.root(), "logs/03_semisupervision/")
    os.makedirs(log_dir, exist_ok=True)
    logger = util.setup_logger(os.path.join(log_dir , f"ssliter.log"), level=logging.INFO)

    logger.info("Load config file")

    config = cfgutil.read(args.config, convert_tuple=False)
    
    torch.manual_seed(config["experiment"]["seed"])
    np.random.seed(config["experiment"]["seed"])
    random.seed(config["experiment"]["seed"])

    assert config["experiment"]["loss_type"] in ["inverse", "reciprocal", "deepsad"], "Loss type must be either inverse or reciprocal or deepsad"

    # now perform supervised training
    assert config["experiment"]["model_path"] is not None, "Model path must be set for supervised training"
    logger.info("Perform supervised finetuning")

    config["hp_config"]["lr"] = config["hp_config"]["lr"] * config["hp_config"].pop("lr_factor")

    ds_config = config["dataset"]
    if config["experiment"]["loss_type"] != "deepsad":
        ds_config = {k: v for k, v in config["dataset"].items() if "supervised" not in k}
        ds_config.update(config["dataset"]["supervised"])

    finetune_supervised(hp_config=config["hp_config"],
                        ds_config=ds_config,
                        experiment_config=config["experiment"])
    logger.info("Done with supervised finetuning")
