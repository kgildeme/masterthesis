
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parents[2]))

import argparse
from datetime import datetime
import json
import logging
import os
import random

import torch
import torch.nn.functional as F

from cids.util import misc_funcs as misc
from cids.models.nn import MLP
from cids.eval.supervised import eval_mlp
from cids.util.metrics import precision, recall, f1_score, fpr
import util

if __name__ == "__main__":
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

    logger.info(f"Loading model")
    model_config = {
        "input_dim": args.input_dim * window,
        "output_dim":args.output_dim,
        "hidden_dims": [128, 64] if "hids" in ds_name else [255, 64],
    }

    model_path = os.path.join(misc.root(), "models/supervised/mlp/hids/20250205-145633/ckpt.pt")
    model = MLP(**model_config)
    ckpt = torch.load(model_path, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device=device)
    model.eval()
    del ckpt

    with open(os.path.join(misc.root(), "data/tmp/dataset_features/hids-v5_201_train_STAGE_4.json"), 'r') as f:
        ds_features = list(json.load(f).keys())
    ds_features = [f for f in ds_features if f not in ['row_idx', 'new_process', 'label']]

    eval_data_config = {
        "ds_name": ds_name,
        "parts": 16,
        "shuffle": False,
        "eval_mode": True,
        "features": ds_features
    }

    logger.info("Start evaluation")
    results = eval_mlp(model, eval_data_config, device=device, loss_fnc=F.binary_cross_entropy_with_logits)

    results["precision"] = precision(results["TP"], results["FP"])
    results["recall"] = recall(results["TP"], results["FN"])
    results["f1_score"] = f1_score(results["TP"], results["FP"], results["TN"], results["FN"])
    results["fpr"] = fpr(results["FP"], results["TN"])

    results_file = os.path.join(misc.root(), f"results/supervised/mlp/{ds_name}/counts_eval.json")
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to {results_file}")

