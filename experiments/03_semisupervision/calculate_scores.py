from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parents[2]))

import argparse
import logging
import os
import copy
import glob
import torch
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from cids.data import OpTCDataset
from cids.util import misc_funcs as misc
from cids.util import config as cfgutil
import util

def load_model(model_path, model_config, device, is_deepsad=False):
    # Load the 025.pt checkpoint and instantiate the model.
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    from cids.models.nn import MLP, Autoencoder 
    if is_deepsad:
        model = MLP(**model_config, bias=False)

        model.load_state_dict(ckpt["model_state_dict"])
        c = ckpt["c"]
    else:
        encoder = MLP(**model_config)
        model_config["input_dim"], model_config["output_dim"] = model_config["output_dim"], model_config["input_dim"]
        decoder = MLP(**model_config)
        decoder = nn.Sequential(decoder, nn.Sigmoid())

        model = Autoencoder(encoder=encoder, decoder=decoder)
        model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    if is_deepsad:
        return model, c
    return model

def get_scores(model, data_config, device, batch_size=128, isdeepsad=False):
    dataset = OpTCDataset(**data_config)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    scores_all = []
    labels_all = []
    if isdeepsad:
        model, c = model
    with torch.no_grad():
        for inp, label in loader:
            inp = torch.flatten(inp, start_dim=1)
            inp = inp.to(device)
            out = model(inp)
            if isdeepsad:
                scores = torch.mean(F.mse_loss(out, c.repeat(inp.shape[0], 1), reduction="none"), dim=1)
            else:
                scores = torch.mean(F.mse_loss(out, inp, reduction="none"), dim=1)
            
            scores_all.append(scores.cpu().numpy())
            labels_all.append(label.cpu().numpy())
    scores_all = np.concatenate(scores_all)
    labels_all = np.concatenate(labels_all)
    return scores_all, labels_all

def save_scores(result_dir, prefix, scores, labels):
    os.makedirs(result_dir, exist_ok=True)
    # Separate scores based on label: 0 = benign, 1 = malicious.
    benign_scores = scores[labels == 0]
    malicious_scores = scores[labels == 1]
    np.save(os.path.join(result_dir, f"benign_{prefix}.npy"), benign_scores)
    np.save(os.path.join(result_dir, f"malicious_{prefix}.npy"), malicious_scores)

def main(hp_config, ds_config, experiment_config):
    ckpt_path =  os.path.join(misc.root(), f"models/03_semisupervision/finetuned/{experiment_config["model_type"]}/{experiment_config["id"]}/{experiment_config["loss_type"]}--gamma{hp_config["gamma"]}--lr{hp_config["lr"]}/seed{experiment_config["seed"]}/last.pt")
    
    md_train_config =copy.deepcopy(hp_config["model"])
    if "patience" in md_train_config:
        md_train_config.pop("patience")

    ds_name = ds_config["ds_name"]
    eval_ds_config = copy.deepcopy(ds_config["supervised"]["eval"])
    test_ds_config = copy.deepcopy(ds_config["supervised"]["test"])
    ds_config = copy.deepcopy(ds_config["supervised"]["train"])

    device = experiment_config["device"]
    model_path = ckpt_path
        # Derive results folder be y replacing directory parts.
    logger.info(f"Calculate score for model {model_path}")
    result_path = model_path.replace("models", "results")
    result_dir = os.path.dirname(result_path)
    try:
        model = load_model(model_path, md_train_config, device, is_deepsad="deepsad" in model_path)
    except RuntimeError as e:
        logger.error(e)
        return
    # Define dataset configurations for OPTC training and evaluation.
    train_data_config = {
        "ds_name": ds_config["ds_name"],
        "parts": ds_config["parts"],
        "window_size": ds_config["window_size"],
        "shuffle": False,
        "stage": ds_config["stage"],
        "only_benign": False,
        "eval_mode": True
    }
    eval_data_config = {
        "ds_name": eval_ds_config["ds_name"],
        "parts": eval_ds_config["parts"],
        "window_size": eval_ds_config["window_size"],
        "shuffle": False,
        "stage": eval_ds_config["stage"],
        "only_benign": False,
        "eval_mode": True
    }

    test_data_config = {
        "ds_name": test_ds_config["ds_name"],
        "parts": test_ds_config["parts"],
        "window_size": test_ds_config["window_size"],
        "shuffle": False,
        "stage": test_ds_config["stage"],
        "only_benign": False,
        "eval_mode": True
    }
    # Calculate scores for the training set.
    train_scores, train_labels = get_scores(model, train_data_config, device, isdeepsad="deepsad" in model_path)
    # Calculate scores for the evaluation set.
    eval_scores, eval_labels = get_scores(model, eval_data_config, device, isdeepsad="deepsad" in model_path)

    test_scores, test_labels = get_scores(model, test_data_config, device, isdeepsad="deepsad" in model_path)
    
    # Save full scores and labels.
    np.save(os.path.join(result_dir, "train_scores.npy"), train_scores)
    np.save(os.path.join(result_dir, "train_labels.npy"), train_labels)
    np.save(os.path.join(result_dir, "eval_scores.npy"), eval_scores)
    np.save(os.path.join(result_dir, "eval_labels.npy"), eval_labels)
    np.save(os.path.join(result_dir, "test_scores.npy"), test_scores)
    np.save(os.path.join(result_dir, "test_labels.npy"), test_labels)
    
    # Also save separated benign and malicious scores.
    save_scores(result_dir, "train", train_scores, train_labels)
    save_scores(result_dir, "eval", eval_scores, eval_labels)
    save_scores(result_dir, "test", test_scores, test_labels)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train full pipeline for semi-supervised CIDS training")

    parser.add_argument("--config", type=str, required=True, help="Path to the config file")
    parser.add_argument("--resume", action="store_true")

    args = parser.parse_args()
    config_path = args.config 

    log_dir = os.path.join(misc.root(), "logs/03_semisupervision/")
    os.makedirs(log_dir, exist_ok=True)
    logger = util.setup_logger(os.path.join(log_dir , f"ssliter.log"), level=logging.INFO)

    logger.info("Load config file")

    config = cfgutil.read(args.config, convert_tuple=False)
    

    assert config["experiment"]["loss_type"] in ["inverse", "reciprocal", "deepsad"], "Loss type must be either inverse or reciprocal or deepsad"

    config["hp_config"]["lr"] = config["hp_config"]["lr"] * config["hp_config"].pop("lr_factor")

    ds_config = config["dataset"]

    main(hp_config=config["hp_config"],
                        ds_config=ds_config,
                        experiment_config=config["experiment"])
    logger.info("Done with supervised finetuning")
