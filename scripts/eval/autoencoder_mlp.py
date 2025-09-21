from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parents[2]))

import argparse
import os
import json

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from cids.util import misc_funcs as misc
from cids.eval.unsupervised import eval_autoencoder
from cids.models.nn import MLP, Autoencoder
from cids.anomaly_scores import ReconstructionAnomaly
from cids.data import worker_init
from cids.util.metrics import precision, recall, f1_score, fpr, auc_score
import util

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Masked Encoder')
    parser.add_argument('--window', type=int, default=10, help='Window size for the encoder')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use for training')
    parser.add_argument('--ds_name', type=str, default='hids-v5_201_train', help='Dataset name')
    parser.add_argument('--input_dim', type=int, default=103, help='Input dimension')
    parser.add_argument('--latent_dim', type=int, default=15, help='Output dimension')
    args = parser.parse_args()

    os.makedirs(os.path.join(misc.root(), "logs/eval"), exist_ok=True)
    logger = util.setup_logger(os.path.join(misc.root(), f"logs/eval/baseline_mlp.log"))
    device = args.device
    window = args.window
    ds_name = args.ds_name
    if "hids" in ds_name:
        model_path = os.path.join(misc.root(), "models/autoencoder/mlp/hids/20250203-085007/ckpt.pt")
    else:
        model_path = os.path.join(misc.root(), "models/autoencoder/mlp/cids/20250203-094431/ckpt.pt")

    logger.info(f"Loading model from {model_path}")
    model_config = {
        "input_dim": args.input_dim * window,
        "output_dim":args.latent_dim,
        "hidden_dims": [128] if "hids" in ds_name else [255],
    }
    encoder = MLP(**model_config)
    model_config["input_dim"], model_config["output_dim"] = model_config["output_dim"], model_config["input_dim"]
    decoder = MLP(**model_config)
    decoder = nn.Sequential(decoder, nn.Sigmoid())

    model = Autoencoder(encoder=encoder, decoder=decoder)

    ckpt = torch.load(model_path, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device=device)
    model.eval()
    del ckpt

    logger.info("initializing anomaly score object")
    anomaly = ReconstructionAnomaly(model, device=device)

    data_config_train = {
        "ds_name": ds_name,
        "parts": 16,
        "window_size": window,
        "shuffle": True,
        "eval_mode": False
    }
    data_config_eval = {
        "ds_name": ds_name.replace("train", "eval"),
        "parts": 16,
        "window_size": window,
        "shuffle": True,
        "eval_mode": True
    }
    exp_dict = "results/autoencoder/mlp/baseline/hids" if "hids" in ds_name else "results/autoencoder/mlp/baseline/cids"
    batch_size = 128
    logger.info("Start evaluation")
   

    results = eval_autoencoder(
        anomaly=anomaly,
        score_file_dict=os.path.join(misc.root(), exp_dict, "scores_eval"),
        data_config_train=data_config_train,
        data_config_eval=data_config_eval,
        batch_size=batch_size,
        device=device,
        n_worker=1,
        worker_init=worker_init,
        flatten=True
    )

    results["precision"] = precision(results["TP"], results["FP"])
    results["recall"] = recall(results["TP"], results["FN"])
    results["f1_score"] = f1_score(results["TP"], results["FP"], results["TN"], results["FN"])
    results["fpr"] = fpr(results["FP"], results["TN"])
    
    scores_malicious = np.load(os.path.join(misc.root(), exp_dict, "scores_eval/malicious_eval.npy"))
    scores_benign = np.load(os.path.join(misc.root(), exp_dict, "scores_eval/benign_eval.npy"))

    y_scores = np.concatenate([scores_benign, scores_malicious])
    y_true = np.concatenate([np.zeros_like(scores_benign), np.ones_like(scores_malicious)])

    results["roc_auc"] = auc_score(y_true, y_scores)

    results_file = os.path.join(misc.root(), exp_dict, "counts_eval.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to {results_file}")
    

    