from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parents[2]))

import random
import os
import argparse
import logging
import datetime

import torch
from torch import nn

import numpy as np
from ray import tune
from ray.tune.search.hyperopt import HyperOptSearch
from ray.tune.schedulers import FIFOScheduler
from ray.train import RunConfig, CheckpointConfig, ScalingConfig

from cids.training.supervised import train_mlp_ray, train_transformer_mlp_ray
from cids.util import misc_funcs as misc
from cids.util.config import transform_ray_hp_config, read
from cids.util.metrics import top1_accuracy
from cids.data import get_SCVIC_dataloader

import util

logger = logging.getLogger(__name__)

def run_hyperparameteroptimization(hp_config, ds_config: dict, id=None, n_trials=100, gpus_per_trial=1, cpus_per_trial=2,
                                   seed=42, model_type="MLP"):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if id is None:
        id = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')

    metric = "val/acc"
    mode = "max"
    config = transform_ray_hp_config(hp_config)
    if "mlp" == model_type.lower():
        trainable = train_mlp_ray
    elif "mlptransformer" == model_type.lower():
        trainable = train_transformer_mlp_ray
    else:
        raise ValueError("Unknown Model Type")


    logger.info("Start loading train and validation set")
    ds_name = ds_config.pop("name")
    if "scvic" in ds_name.lower():
        eval_ds_config = None
    elif "optc" in ds_name.lower():
        eval_ds_config = ds_config["eval"]
        ds_config = ds_config["train"]

    logger.info("Done loading data")
    searcher = HyperOptSearch(metric=metric, mode=mode)
    scheduler = FIFOScheduler()

    run_config = RunConfig(
        storage_path=os.path.join(misc.root(), f"results/00_hyperparameter-tuning/{ds_name}/supervised/{model_type}"),
        name=id,
        log_to_file=True
    )

    tune_config = tune.TuneConfig(
        mode=mode,
        metric=metric,
        search_alg=searcher,
        scheduler=scheduler,
        num_samples=n_trials,
        trial_dirname_creator=misc.trial_name,
        trial_name_creator=misc.trial_name,
    )

    args_trainable = {
        "ds_type": ds_name,
        "ds_args": ds_config,
        "eval_ds_args": eval_ds_config,
        "batch_size": hp_config["batch_size"],
        "epochs": hp_config["epochs"],
        "loss_fnc": nn.CrossEntropyLoss(),
        "accuracy": top1_accuracy,
        "early_stopping": hp_config["patience"]
    }
    logger.info("start tuner")
    tuner = tune.Tuner(
        tune.with_resources(tune.with_parameters(trainable=trainable, **args_trainable), {'cpu': cpus_per_trial, 'gpu': gpus_per_trial}),
        tune_config=tune_config,
        run_config=run_config,
        param_space=config
    )
    try:
        results = tuner.fit()
    except Exception as e:
        logger.exception("An unexpected error occured")

    return results
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Hyperparameter Tuning for simple MLP on ')
    parser.add_argument('--config', type=str, help='Path to config file')

    args = parser.parse_args()
    log_dir = os.path.join(misc.root(), "logs/00_hyperparameter_tuning/")
    os.makedirs(log_dir, exist_ok=True)
    logger = util.setup_logger(os.path.join(log_dir , f"mlp_classification.log"), level=logging.INFO)
    logger.info("Load config file")
    config = read(args.config, convert_tuple=True)
    hp_config = config["hp_config"]
    ds_config = config["dataset"]
    exp_config = config["experiment"]

    logger.info(f"Start hyperparameter optimization for id {exp_config["id"]}")
    results = run_hyperparameteroptimization(
        hp_config=hp_config,
        ds_config=ds_config,
        id=exp_config["id"],
        n_trials=exp_config["n_trials"],
        gpus_per_trial=exp_config["gpus_per_trial"],
        cpus_per_trial=exp_config["cpus_per_trial"],
        seed=exp_config["seed"],
        model_type="MLPTransformer"
    )

