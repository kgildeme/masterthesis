from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parents[2]))

import random
import os
import argparse
import logging
import datetime

import torch
import numpy as np
from ray import tune
from ray.tune.search.hyperopt import HyperOptSearch
from ray.train import RunConfig, CheckpointConfig, ScalingConfig

from cids.training import hyperparameter_tuning_masked_encoder
from cids.util import misc_funcs as misc
import util

def setup_ray(exp_id, ds_name, report_every: int = 10000, num_samples=10, gpus_per_trial=1, cpus_per_trial=2, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    metric = "val_loss"
    mode = "min"

    

    config = {
        "window": tune.randint(10, 200),
        "n_layer": tune.randint(2, 7),
        "n_head": tune.choice([2, 4, 8, 16]),
        "model_dim": tune.choice([32, 64, 128, 256]),
        "factor_dim_feedforward": tune.choice([0.5, 1, 2]),
        "batch_size": tune.choice([32, 64, 128]),
        "lr": tune.loguniform(1e-6, 1e-3),
        "patience": 2,
        "epochs": 20
    }

    sampled_values = random.sample(range(16), 3)
    remaining_values = [x for x in range(16) if x not in sampled_values]
    logger.info(f"Keep out from training: {sampled_values}")

    train_data_config = {
        "ds_name": ds_name,
        "parts": remaining_values,
        "shuffle": True,
        "last_part": 16
    }
    eval_data_config = {
        "ds_name": ds_name,
        "parts": sampled_values,
        "shuffle": False,
        "last_part": 16
    }
    args_trainable = {
        "train_dataset_config": train_data_config,
        "eval_dataset_config": eval_data_config,
        "n_worker": 1,
        "report_every": report_every,
    }

    searcher = HyperOptSearch(metric=metric, mode=mode)
    fifo_scheduler = tune.schedulers.MedianStoppingRule(time_attr='training_iteration', grace_period=10, min_samples_required=5)

    run_config = RunConfig(
        checkpoint_config=CheckpointConfig(
            num_to_keep=3,
            checkpoint_score_attribute=metric,
            checkpoint_score_order=mode
        ),
        storage_path=os.path.join(misc.root(), f"results/hyperparameter-tuning/masked-encoder/{ds_name}"), name=exp_id,
        log_to_file=True
    )

    tune_config = tune.TuneConfig(
        mode=mode,
        metric=metric,
        search_alg=searcher,
        scheduler=fifo_scheduler,
        num_samples=num_samples,
        trial_dirname_creator=misc.trial_name,
        trial_name_creator=misc.trial_name,
    )

    tuner = tune.Tuner(
        tune.with_resources(tune.with_parameters(trainable=hyperparameter_tuning_masked_encoder, **args_trainable), {'cpu': cpus_per_trial, 'gpu': gpus_per_trial}),
        tune_config=tune_config,
        run_config=run_config,
        param_space=config
    )

    results = tuner.fit()

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Hyperparameter Tuning Masked Encoder')
    parser.add_argument('--ds-name', type=str, help='Dataset name')

    args = parser.parse_args()
    ds_name = args.ds_name
    log_dir = os.path.join(misc.root(), "logs/hyperoptimization")
    os.makedirs(log_dir, exist_ok=True)
    logger = util.setup_logger(os.path.join(misc.root(), "logs/hyperoptimization/hyperparameter-tuning-masked-encoder.log"), level=logging.INFO)

    date = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    logger.info(f"Starting hyperparameter tuning for masked encoder on {ds_name} at {date}")
    setup_ray(exp_id=f"{date}", ds_name=ds_name, report_every=10000, num_samples=100, gpus_per_trial=0.5, cpus_per_trial=2, seed=1)
    date = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    logger.info(f"Finished hyperparameter tuning for masked encoder on {ds_name} at {date}")