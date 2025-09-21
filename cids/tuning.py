import datetime
import logging
import os
import random

import numpy as np
from torch.nn import functional as F
from ray import tune
from ray.tune import ResultGrid
from ray.tune.search.hyperopt import HyperOptSearch
from ray.tune.schedulers import FIFOScheduler
from ray.train import RunConfig
import torch

from .util import config as cfgutil
from .util import misc_funcs as misc
from .util.metrics import auc_score

logger = logging.getLogger(__name__)

def hp_optimization(
        hp_config,
        ds_config: dict,
        trainable,
        id=None, 
        n_trials=100, 
        gpus_per_trial=1, 
        cpus_per_trial=2,
        seed=42,
        storage_path="results/debug",
        metric="val/score",
        mode="max",
        loss_fnc= F.mse_loss,
        eval_metric= auc_score) -> ResultGrid:
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if id is None:
        id = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')

    config = cfgutil.transform_ray_hp_config(hp_config)

    logger.info("Start loading train and validation set")
    ds_name = ds_config["name"]

    logger.info("Done loading data")
    searcher = HyperOptSearch(metric=metric, mode=mode)
    scheduler = FIFOScheduler()

    run_config = RunConfig(
        storage_path=os.path.join(misc.root(), storage_path),
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
        "n_worker": ds_config.pop("n_worker"),
        "ds_args": ds_config,
        "eval_ds_args": eval_ds_args,
        "loss_fnc": loss_fnc,
        "eval_metric": eval_metric,
        "early_stopping": hp_config["patience"],
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