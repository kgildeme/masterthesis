from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parents[2]))

import random
import os
import argparse
import logging
import datetime

import numpy as np
from ray import tune
from ray.tune.search.basic_variant import BasicVariantGenerator
from ray.tune.schedulers import FIFOScheduler
from ray.train import RunConfig

from cids.training.supervised import train_rf_ray
from cids.util import misc_funcs as misc
from cids.util.config import transform_ray_hp_config, read

import util

logger = logging.getLogger(__name__)

def run_hyperparameteroptimization(hp_config, ds_config: dict, id=None, gpus_per_trial=1, cpus_per_trial=2,
                                   seed=42):

    random.seed(seed)
    np.random.seed(seed)

    ds_name = ds_config.pop("name")

    if id is None:
        id = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')

    metric = "val/acc"
    mode = "max"
    config = transform_ray_hp_config(hp_config)["model"]

    searcher = BasicVariantGenerator()
    scheduler = FIFOScheduler()

    run_config = RunConfig(
        storage_path=os.path.join(misc.root(), f"results/00_hyperparameter-tuning/{ds_name}/supervised/RF"),
        name=id,
        log_to_file=True
    )

    tune_config = tune.TuneConfig(
        mode=mode,
        metric=metric,
        search_alg=searcher,
        scheduler=scheduler,
        trial_dirname_creator=misc.trial_name,
        trial_name_creator=misc.trial_name,
    )

    args_trainable = {
        "ds_type": ds_name,
        "ds_args": ds_config
    }
    logger.info("start tuner")
    tuner = tune.Tuner(
        tune.with_resources(tune.with_parameters(trainable=train_rf_ray, **args_trainable), {'cpu': cpus_per_trial, 'gpu': gpus_per_trial}),
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
    logger = util.setup_logger(os.path.join(log_dir , f"rf_classification.log"), level=logging.INFO)
    logger.info("Load config file")
    config = read(args.config, convert_tuple=True)
    hp_config = config["hp_config"]
    ds_config = config["dataset"]
    exp_config = config["experiment"]

    hp_config["model"]["n_jobs"] = exp_config["cpus_per_trial"]

    logger.info(f"Start hyperparameter optimization for id {exp_config["id"]}")
    results = run_hyperparameteroptimization(
        hp_config=hp_config,
        ds_config=ds_config,
        id=exp_config["id"],
        gpus_per_trial=exp_config["gpus_per_trial"],
        cpus_per_trial=exp_config["cpus_per_trial"],
        seed=exp_config["seed"]
    )

