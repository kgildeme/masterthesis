from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parents[2]))
import os
import logging
from datetime import datetime

import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import torch 
import torch.nn as nn

from cids.training import train_autoregressive_encoder
from cids.model import AutoregressiveTransformerEncoder
from cids.util import misc_funcs as misc
import util

if __name__ == '__main__':
    test_model_mem = False

    logger = util.setup_logger(os.path.join(misc.root(), "logs/debugging/train_autoregressive_encoder_host.log"), level=logging.INFO)
    device = "cuda:2"
    window = 30
    time = datetime.now().strftime('%Y%m%d-%H%M%S')

    model_config = {
        "n_layer": 6,
        "head":nn.Linear(256, 102),
        "embedding":nn.Linear(102, 256),
        "d_model":256,
        "n_head":8,
        "dim_feedforward":512,
        "dropout":0.1,
        "activation":F.relu,
        "window_size":window,
        "layer_norm_eps":1e-05,
        "norm_first":True, "bias":True,
        "device":None,
        "dtype":None
    }

    if test_model_mem:
        for i in range(torch.cuda.device_count()):
            print(torch.cuda.get_device_properties(i).name)
        model = AutoregressiveTransformerEncoder(**model_config)
        model.to(device=device)
        mask=torch.nn.Transformer.generate_square_subsequent_mask(20)
        mask.to(device)
        inp = torch.rand((16, 20, 169))
        inp = inp.to(device)
        print(inp.device)
        out = model(inp, mask)
        input("Press Enter to continue")
        exit()

    data_config = {
        "ds_name": "hids-v5_201_train",
        "parts": 16,
        "window_size": window,
        "shuffle": True
    }

    epochs = 60
    batch_size = 128
    writer = SummaryWriter(os.path.join(misc.root(), "logs/tensorboard/debugging/host", time))

    results = train_autoregressive_encoder(
        model_config=model_config,
        dataset_config=data_config,
        batch_size=batch_size,
        epochs=epochs,
        early_stopping=None,
        checkpoint_path=os.path.join(misc.root(), "models/debugging/host", time, "ckpt.pt"),
        tensorboard_writer=writer,
        report_every=40,
        n_worker = 1,
        device=device,
        restart_ckpt_path=None
    )
    writer.flush()
    writer.close()

    