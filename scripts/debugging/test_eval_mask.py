from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parents[2]))
import os
import json

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from cids.util import misc_funcs as misc
from cids.eval import eval_masked_encoder
from cids.model import MaskedTransformerEncoder
from cids.anomaly_scores import ReconstructionAnomaly
from cids.data import worker_init
from cids.util.metrics import precision, recall, f1_score, fpr, auc_score
import util

if __name__ == "__main__":

    logger = util.setup_logger(os.path.join(misc.root(), f"logs/debugging/eval_autoregressive_encoder_mask_host.log"))
    device = "cuda:2"
    predict_window = False

    window = 32
    model_path = os.path.join(misc.root(), "models/debugging/host/masked/20250128-192408/ckpt.pt")

    logger.info(f"Loading model from {model_path}")
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
    model = MaskedTransformerEncoder(**model_config)

    ckpt = torch.load(model_path, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device=device)
    model.eval()
    del ckpt

    logger.info("initializing anomaly score object")
    anomaly = ReconstructionAnomaly(model, device=device)

    data_config_train = {
        "ds_name": "hids-v5_201_train",
        "parts": 16,
        "window_size": window,
        "shuffle": True,
        "eval_mode": False
    }
    data_config_eval = {
        "ds_name": "hids-v5_201_eval",
        "parts": 16,
        "window_size": window,
        "shuffle": True,
        "eval_mode": True
    }
    exp_dict = "results/debugging/host/masked/20250128-192408"
    batch_size = 128
    logger.info("Start evaluation")
   
    results = eval_masked_encoder(
        anomaly=anomaly,
        score_file_dict=os.path.join(misc.root(), exp_dict, "scores_eval"),
        data_config_train=data_config_train,
        data_config_eval=data_config_eval,
        batch_size=batch_size,
        device=device,
        n_worker=1,
        worker_init=worker_init,
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
    

    