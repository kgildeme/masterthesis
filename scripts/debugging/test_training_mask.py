from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parents[2]))
import os
import logging
from datetime import datetime
import argparse

import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import torch 
import torch.nn as nn

from cids.training import train_masked_encoder
from cids.util import misc_funcs as misc
import util

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Masked Encoder')
    parser.add_argument('--window', type=int, default=10, help='Window size for the encoder')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use for training')
    args = parser.parse_args()

    test_model_mem = False

    logger = util.setup_logger(os.path.join(misc.root(), "logs/debugging/train_mask_encoder_host.log"), level=logging.INFO)
    device = args.device
    window = args.window
    time = datetime.now().strftime('%Y%m%d-%H%M%S')

    model_config = {
        "n_layer": 3,
        "head":nn.Sequential(nn.Linear(64, 102), nn.Sigmoid()),
        "embedding":nn.Linear(102, 64),
        "d_model":64,
        "n_head":8,
        "dim_feedforward":128,
        "dropout":0.1,
        "activation":F.relu,
        "window_size":window,
        "layer_norm_eps":1e-05,
        "norm_first":True, "bias":True,
        "device":None,
        "dtype":None
    }

    data_config = {
        "ds_name": "hids-v5_201_train",
        "parts": 16,
        "window_size": window,
        "shuffle": True
    }

    epochs = 60
    batch_size = 128
    lr = 1e-5
    writer = SummaryWriter(os.path.join(misc.root(), "logs/tensorboard/debugging/host/masked", time))

    results = train_masked_encoder(
        model_config=model_config,
        dataset_config=data_config,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        use_scheduler=True,
        early_stopping=None,
        checkpoint_path=os.path.join(misc.root(), "models/debugging/host/masked", time, "ckpt.pt"),
        report_every=40,
        n_worker=1,
        device=device,
        tensorboard_writer=writer,
        restart_ckpt_path=None
    )
    writer.flush()
    writer.close()

    