from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parents[2]))

import argparse
import copy
import logging
import os

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from cids.models.nn import MLP, Autoencoder
from cids.util.metrics import auc_score
import util
from cids.util import config as cfgutil, misc_funcs as misc
from cids.data import OpTCDataset, worker_init
import pandas as pd

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train full pipeline for semi-supervised CIDS training")

    parser.add_argument("--config", type=str, required=True, help="Path to the config file")
    parser.add_argument("--host")

    args = parser.parse_args()
    configs_path = args.config 

    log_dir = os.path.join(misc.root(), "logs/03_semisupervision/")
    os.makedirs(log_dir, exist_ok=True)
    logger = util.setup_logger(os.path.join(log_dir , f"ssl.log"), level=logging.INFO)

    logger.info("Load config file")
    configs = {}
    models = {}

    aucs = {}

    ds = OpTCDataset(
        ds_name=f"cids-v5_{args.host}_eval-ff_train",
        parts=8,
        window_size=10,
        shuffle=False,
        eval_mode=True,
        only_benign=False,
        stage=5
    )

    dl = torch.utils.data.DataLoader(ds, batch_size=32, num_workers=1, worker_init_fn=worker_init)
    for path in configs_path.split(','):
        config = cfgutil.read(path, convert_tuple=False)
        configs[config["experiment"]["id"]] = config
        models[config["experiment"]["id"]] = []
        aucs[config["experiment"]["id"]] = []
        for i, s in enumerate(config["experiment"]["seed"]):
            
            model_config = copy.deepcopy(config["hp_config"]["model"])
            encoder = MLP(**model_config)
            model_config["input_dim"], model_config["output_dim"] = model_config["output_dim"], model_config["input_dim"]
            decoder = MLP(**model_config)
            decoder = nn.Sequential(decoder, nn.Sigmoid())

            model = Autoencoder(encoder=encoder, decoder=decoder)
            model_path = os.path.join(misc.root(), f"models/03_semisupervision/unsupervised/{config["dataset"]["ds_name"]}/{config["experiment"]["model_type"]}/{config["experiment"]["id"]}/{config["experiment"]["id"]}-{i}/ckpt.pt")
            logger.info("start training")
            ckpt = torch.load(model_path, weights_only=True, map_location='cpu')
            model.load_state_dict(ckpt["model_state_dict"])

            model.to(device=config["experiment"]["device"][0])
            device = config["experiment"]["device"][0]
            loss_fnc = nn.functional.mse_loss

            with torch.no_grad():
                model.eval()
                eval_loss = 0
                eval_steps = 0
                scores = []
                labels = []
                for eval_data in dl:
                    eval_inp = torch.flatten(eval_data[0], start_dim=1)
                    eval_inp = eval_inp.to(device)

                    eval_out = model(eval_inp)
                    eval_labels = eval_data[-1].to(device)  
                    loss = loss_fnc(eval_out, eval_inp)
                    eval_loss += loss.item()

                    loss_expanded = F.mse_loss(eval_inp, eval_out, reduction="none")                    
                    scores.extend(torch.mean(loss_expanded, dim=1).detach().cpu().numpy())
                    labels.extend(eval_labels.detach().cpu().numpy())

                    eval_steps += 1
                
                eval_loss /= eval_steps
            aucs[config["experiment"]["id"]].append(auc_score(labels, scores))
            logger.info(f"auc for iter {i} = {aucs[config["experiment"]["id"]][-1]}")



    scores = {}
    for key in aucs.keys():
        scores[key] = np.mean(aucs[key])

    results_path = os.path.join(misc.root(), f"results/03_semisupervision/unsupervised/{config['dataset']["ds_name"]}/{config['experiment']['model_type']}/results.csv")
    existing_scores = None
    if os.path.exists(results_path):
        existing_scores = pd.read_csv(results_path, index_col=0)

    scores_df = pd.DataFrame.from_dict(scores, orient='index', columns=['mean_auc'])
    if existing_scores is not None:
        scores_df = pd.concat([existing_scores, scores_df])
    scores_df.to_csv(os.path.join(misc.root(), results_path))

        

        

