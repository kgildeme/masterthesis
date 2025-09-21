from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parents[2]))
import os
import logging
from datetime import datetime
import argparse
import random

import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import torch 
import torch.nn as nn

from cids.training.supervised import train_mlp
from cids.util import misc_funcs as misc
import util

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Masked Encoder')
    parser.add_argument('--window', type=int, default=10, help='Window size for the encoder')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use for training')
    parser.add_argument('--ds_name', type=str, default='hids-v5_201_train', help='Dataset name')
    parser.add_argument('--input_dim', type=int, default=103, help='Input dimension')
    parser.add_argument('--output_dim', type=int, default=15, help='Output dimension')
    args = parser.parse_args()

    logger_dict = os.path.join(misc.root(), "logs/model_training")
    os.makedirs(logger_dict, exist_ok=True)
    logger = util.setup_logger(os.path.join(misc.root(), "logs/model_training/supervised_mlp.log"), level=logging.INFO)
    device = args.device
    window = args.window
    ds_name = args.ds_name
    time = datetime.now().strftime('%Y%m%d-%H%M%S')

    model_config = {
        "input_dim": args.input_dim * window,
        "output_dim":args.output_dim,
        "hidden_dims": [128, 64] if "hids" in ds_name else [255, 64],
    }

    sampled_values = random.sample(range(16), 3)
    remaining_values = [x for x in range(16) if x not in sampled_values]
    logger.info(f"Keep out from training: {sampled_values}")

    train_data_config = {
        "ds_name": ds_name,
        "parts": remaining_values,
        "shuffle": True,
        "last_part": 16,
        "eval_mode": True
    }
    eval_data_config = {
        "ds_name": ds_name,
        "parts": sampled_values,
        "shuffle": False,
        "last_part": 16,
        "eval_mode": True
    }

    epochs = 30
    batch_size = 32
    lr = 1e-6
    if "hids" in ds_name:
        writer = SummaryWriter(os.path.join(misc.root(), "logs/tensorboard/model_training/supervised/mlp/hids/", time))
    elif "cids" in ds_name:
        writer = SummaryWriter(os.path.join(misc.root(), "logs/tensorboard/model_training/supervised/mlp/cids/", time))

    ckpt_dir = os.path.join(misc.root(), f"models/supervised/mlp/{"hids" if "hids" in ds_name else "cids"}", time)
    os.makedirs(ckpt_dir, exist_ok=True)
    try:
        results = train_mlp(
                model_config=model_config,
                train_dataset_config=train_data_config,
                eval_dataset_config=eval_data_config,
                undersampling_dist={0: 0.67, 1: 0.33},
                epochs=epochs,
                batch_size=batch_size,
                lr=lr,
                early_stopping=5,
                checkpoint_path=os.path.join(ckpt_dir, "ckpt.pt"),
                report_every=100,
                device=device,
                tensorboard_writer=writer,
                restart_ckpt_path=None
            )
        writer.flush()
        writer.close()
    except Exception as e:
        logger.exception(e)
        raise e

    