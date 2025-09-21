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
from torch.nn import functional as F

import numpy as np
from ray import tune
from ray.tune.search.hyperopt import HyperOptSearch
from ray.tune.schedulers import FIFOScheduler
from ray.train import RunConfig

from cids.training.unsupervised import train_mlp_autoencoder_ray
from cids.util import misc_funcs as misc
from cids.util.config import transform_ray_hp_config, read
from cids.util.metrics import auc_score

import util

logger = logging.getLogger(__name__)

def run_hyperparameteroptimization(hp_config, ds_config: dict, id=None, n_trials=100, gpus_per_trial=1, cpus_per_trial=2,
                                   seed=42, resume=False):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if id is None:
        id = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')

    metric = "val/score"
    mode = "max"
    config = transform_ray_hp_config(hp_config)

    logger.info("Start loading train and validation set")
    ds_name = ds_config["name"]
    n_worker = ds_config["n_worker"]
    if "scvic" in ds_name.lower():
        eval_ds_config = None
    elif "optc" in ds_name.lower():
        eval_ds_config = ds_config["eval"]
        ds_config = ds_config["train"]

    logger.info("Done loading data")
    searcher = HyperOptSearch(metric=metric, mode=mode)
    scheduler = FIFOScheduler()

    run_config = RunConfig(
        storage_path=os.path.join(misc.root(), f"results/00_hyperparameter-tuning/{ds_name}/unsupervised/MLPAE"),
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
        "ds_name": ds_name,
        "n_worker": n_worker,
        "ds_args": ds_config,
        "eval_ds_args": eval_ds_config,
        "loss_fnc": F.mse_loss,
        "eval_metric": auc_score,
        "early_stopping": hp_config["patience"],
    }

    logger.info("start tuner")
    tuner = tune.Tuner(
        tune.with_resources(tune.with_parameters(trainable=train_mlp_autoencoder_ray, **args_trainable), {'cpu': cpus_per_trial, 'gpu': gpus_per_trial}),
        tune_config=tune_config,
        run_config=run_config,
        param_space=config
    )
    if resume:
        tuner = tune.Tuner.restore(
            path=os.path.join(misc.root(), f"results/00_hyperparameter-tuning/{ds_name}/unsupervised/MLPAE", id),
            trainable=tune.with_resources(tune.with_parameters(trainable=train_mlp_autoencoder_ray, **args_trainable), {'cpu': cpus_per_trial, 'gpu': gpus_per_trial}),
            param_space=config,
            resume_unfinished=True,
            resume_errored=True,
        )
    try:
        results = tuner.fit()
    except Exception as e:
        logger.exception("An unexpected error occured")

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Hyperparameter Tuning for simple MLP-AE on ')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--train_best', action='store_true')
    parser.add_argument('--no-hp', action='store_true')

    args = parser.parse_args()
    log_dir = os.path.join(misc.root(), "logs/00_hyperparameter_tuning/")
    os.makedirs(log_dir, exist_ok=True)
    logger = util.setup_logger(os.path.join(log_dir , f"mlp_autoencoder.log"), level=logging.INFO)
    logger.info("Load config file")
    config = read(args.config, convert_tuple=True)
    hp_config = config["hp_config"]
    ds_config = config["dataset"]
    exp_config = config["experiment"]

    if not args.no_hp:
        logger.info(f"Start hyperparameter optimization for id {exp_config["id"]}")
        results = run_hyperparameteroptimization(
            hp_config=hp_config,
            ds_config=ds_config,
            id=exp_config["id"],
            n_trials=exp_config["n_trials"],
            gpus_per_trial=exp_config["gpus_per_trial"],
            cpus_per_trial=exp_config["cpus_per_trial"],
            seed=exp_config["seed"],
            resume=args.resume
        )
    if args.train_best:
        logger.info(f"Start training best 3 modelconfigurations for 10 seeds")


