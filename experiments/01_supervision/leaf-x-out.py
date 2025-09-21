from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parents[2]))


import argparse
import logging
import os

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from cids.models.nn import MLP, CollaborativeIDSNet
from cids.training.supervised import train_cids, train_mlp
from cids.eval.supervised import eval_mlp_confusion, eval_collaborative_confusion
from cids.data import SCVIC_CIDS_CLASSES, get_SCVIC_Dataset, SCVICCIDSDataset
from cids.util import misc_funcs as misc, metrics
from cids.util.config import read
import util

def train(hp_config: dict, ds_config: dict, experiment_config: dict):

    is_mlp = "mlp" in experiment_config["model_type"].lower()
    trainable = train_mlp if is_mlp else train_cids

    model_path = os.path.join(misc.root(), f"models/01_supervision/leaf-x-out/{experiment_config["model_type"]}/{experiment_config["id"]}/{experiment_config["trial"]}/ckpt.pt")
    tensorboard_path = os.path.join(misc.root(), f"results/01_supervision/leaf-x-out/{experiment_config["model_type"]}/{experiment_config["id"]}/{experiment_config["trial"]}/tensorboard/")
    writer = SummaryWriter(tensorboard_path)

    ds_train_config = ds_config.copy()
    md_train_config = hp_config.copy()

    trainable_args = {
        "model_config": md_train_config.pop("model"),
        "ds_type": ds_train_config.pop("ds_name"), 
        "ds_args": ds_train_config,
        "loss_fnc": nn.CrossEntropyLoss() if is_mlp else metrics.combined_loss,
        "accuracy": metrics.top1_accuracy, 
        "checkpoint_path": model_path,
        "tensorboard_writer": writer,
        "device": experiment_config["device"]
    }

    trainable_args = trainable_args | md_train_config
    trainable(**trainable_args)
    return model_path

def test(hp_config: dict, ds_config: dict, experiment_config: dict):

    is_mlp = "mlp" in experiment_config["model_type"].lower()
    eval_fnc = eval_mlp_confusion if is_mlp else eval_collaborative_confusion

    if "model_path" not in experiment_config.keys():
        experiment_config["model_path"] = os.path.join(misc.root(), f"models/01_supervision/leaf-x-out/{experiment_config["model_type"]}/{experiment_config["id"]}/{experiment_config["trial"]}/ckpt.pt")
    # create dataloader
    exclude = ds_config["exclude"]
    exclude_labels = [SCVIC_CIDS_CLASSES[c] for c in exclude]
    include = [c for c in SCVIC_CIDS_CLASSES.keys() if c not in exclude]

    dss = get_SCVIC_Dataset(exclude=None, train=True, validation=True, test=True, random_state=ds_config["random_state"], network_data=ds_config["network_data"],
                            host_data=ds_config["host_data"], host_embeddings=ds_config["host_embeddings"])
    
    val_ds = dss["val"]
    train_ds = dss["train"]
    test_ds = dss["test"]

    for point in train_ds.data:
        if point[-1] in exclude_labels:
            val_ds.data.append(point)
    
    train_ds = SCVICCIDSDataset(data=train_ds.data, exclude_classes=exclude_labels, network_data=ds_config["network_data"],
                            host_data=ds_config["host_data"], host_embeddings=ds_config["host_embeddings"])

    train_dl = torch.utils.data.DataLoader(train_ds, ds_config["batch_size"], num_workers=ds_config["num_workers"])
    val_dl = torch.utils.data.DataLoader(val_ds, ds_config["batch_size"], num_workers=ds_config["num_workers"])
    test_dl = torch.utils.data.DataLoader(test_ds, ds_config["batch_size"], num_workers=ds_config["num_workers"])
    save_path = os.path.join(misc.root(), f"results/01_supervision/leaf-x-out/{experiment_config["model_type"]}/{experiment_config["id"]}/{experiment_config["trial"]}/")

    train_confussion = eval_fnc(
        model_path=experiment_config["model_path"],
        model_config=hp_config["model"],
        dataloader=train_dl,
        num_classes=len(SCVIC_CIDS_CLASSES),
        device=experiment_config["device"],
        left_out_mask=torch.tensor([1 if c not in exclude else 0 for c in SCVIC_CIDS_CLASSES.keys()], device=experiment_config["device"])
    )
    misc.save_confusion_matrix(train_confussion, list(SCVIC_CIDS_CLASSES.keys()), save_path + "train_confussion.csv")

    val_confussion = eval_fnc(
        model_path=experiment_config["model_path"],
        model_config=hp_config["model"],
        dataloader=val_dl,
        num_classes=len(SCVIC_CIDS_CLASSES),
        device=experiment_config["device"],
        left_out_mask=torch.tensor([1 if c not in exclude else 0 for c in SCVIC_CIDS_CLASSES.keys()], device=experiment_config["device"])
    )
    misc.save_confusion_matrix(val_confussion, list(SCVIC_CIDS_CLASSES.keys()), save_path + "val_confussion.csv")

    test_confussion = eval_fnc(
        model_path=experiment_config["model_path"],
        model_config=hp_config["model"],
        dataloader=test_dl,
        num_classes=len(SCVIC_CIDS_CLASSES),
        device=experiment_config["device"],
        left_out_mask=torch.tensor([1 if c not in exclude else 0 for c in SCVIC_CIDS_CLASSES.keys()], device=experiment_config["device"])
    )
    misc.save_confusion_matrix(test_confussion, list(SCVIC_CIDS_CLASSES.keys()), save_path + "test_confussion.csv")

if __name__ == "__main__":
    """
    In this script we will perform the leaf-x-out experiment. The goal is to train a model on all classes, but 
    a predefined subset of classes. We will then evaluate the performance of the model on the left out class. If a model_path
    is given in the config we assume that the model is already trained and only run the test-phase
    """

    # Load the configuration file
    parser = argparse.ArgumentParser(description='Leaf-one-out experiment')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the config file')
    parser.add_argument('--test_only', action="store_true", help="Only run the test phase")
    args = parser.parse_args()

    args = parser.parse_args()
    log_dir = os.path.join(misc.root(), "logs/01_supervision/")
    os.makedirs(log_dir, exist_ok=True)
    logger = util.setup_logger(os.path.join(log_dir , f"leafe_x_out.log"), level=logging.INFO)

    logger.info("Load config file")
    config = read(args.config, convert_tuple=False)

    torch.manual_seed(config["experiment"]["seed"])

    if not args.test_only:
        logger.info(f"Training model while leaving out classes: {config['dataset']['exclude']}")
        # train model, in train_fnc model is trained and best version saved at path constructed from experiment_config
        config["experiment"]["model_path"] = train(hp_config=config["hp_config"], ds_config=config["dataset"], experiment_config=config["experiment"])

    logger.info(f"Testing model while leaving out classes: {config['dataset']['exclude']}")
    test(hp_config=config["hp_config"], ds_config=config["dataset"], experiment_config=config["experiment"])
    logger.info("Experiment done")

    


