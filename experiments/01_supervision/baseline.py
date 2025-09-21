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
from cids.training.supervised import train_cids, train_mlp, train_transformer_mlp
from cids.eval.supervised import eval_mlp_confusion_binary, eval_collaborative_confusion_binary
from cids.data import SCVIC_CIDS_CLASSES, get_SCVIC_Dataset, SCVICCIDSDataset
from cids.util import misc_funcs as misc, metrics
from cids.util.config import read
import util

def train(hp_config: dict, ds_config: dict, experiment_config: dict):

    model_type = experiment_config["model_type"]
    is_mlp = "mlp" in model_type.lower()

    if "mlp" == model_type.lower() or "cmlp" == model_type.lower():
        trainable = train_mlp
    elif "mlptransformer" == model_type.lower():
        trainable = train_transformer_mlp
    elif "cids" == model_type.lower():
        trainable = train_cids
    else:
        raise ValueError("Unknown Model Type")

    ds_name = ds_config.pop("ds_name")
    model_path = os.path.join(misc.root(), f"models/01_supervision/baseline-binary/{ds_name}/{experiment_config["model_type"]}/{experiment_config["id"]}/{experiment_config["trial"]}/ckpt.pt")
    tensorboard_path = os.path.join(misc.root(), f"results/01_supervision/baseline-binary/{ds_name}/{experiment_config["model_type"]}/{experiment_config["id"]}/{experiment_config["trial"]}/tensorboard/")
    writer = SummaryWriter(tensorboard_path)
    
    os.makedirs(os.path.split(model_path)[0], exist_ok=True)
    os.makedirs(tensorboard_path, exist_ok=True)
    
    if "scvic" in ds_name.lower():
        eval_ds_config = None
    elif "optc" in ds_name.lower():
        eval_ds_config = ds_config["eval"]
        ds_config = ds_config["train"]
    md_train_config = hp_config.copy()

    trainable_args = {
        "model_config": md_train_config.pop("model"),
        "ds_type": f"{ds_name}_binary", 
        "ds_args": ds_config,
        "eval_ds_args": eval_ds_config, 
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
    eval_fnc = eval_mlp_confusion_binary if is_mlp else eval_collaborative_confusion_binary

    if "model_path" not in experiment_config.keys():
        experiment_config["model_path"] = model_path = os.path.join(misc.root(), f"models/01_supervision/baseline-binary/{experiment_config["model_type"]}/{experiment_config["id"]}/{experiment_config["trial"]}/ckpt.pt")
    # create dataloader

    dss = get_SCVIC_Dataset(exclude=None, train=True, validation=True, test=True, random_state=ds_config["random_state"], network_data=ds_config["network_data"],
                            host_data=ds_config["host_data"], host_embeddings=ds_config["host_embeddings"])
    
    val_ds = dss["val"]
    train_ds = dss["train"]
    test_ds = dss["test"]

    train_dl = torch.utils.data.DataLoader(train_ds, ds_config["batch_size"], num_workers=ds_config["num_workers"])
    val_dl = torch.utils.data.DataLoader(val_ds, ds_config["batch_size"], num_workers=ds_config["num_workers"])
    test_dl = torch.utils.data.DataLoader(test_ds, ds_config["batch_size"], num_workers=ds_config["num_workers"])
    save_path = os.path.join(misc.root(), f"results/01_supervision/baseline-binary/{experiment_config["model_type"]}/{experiment_config["id"]}/{experiment_config["trial"]}/")

    train_confussion = eval_fnc(
        model_path=experiment_config["model_path"],
        model_config=hp_config["model"],
        dataloader=train_dl,
        num_classes=len(SCVIC_CIDS_CLASSES),
        device=experiment_config["device"]
    )
    misc.save_confusion_matrix(train_confussion, ["benign", "malicious"], save_path + "train_confussion.csv", index=[f"true_{k}" for k in SCVIC_CIDS_CLASSES.keys()])

    val_confussion = eval_fnc(
        model_path=experiment_config["model_path"],
        model_config=hp_config["model"],
        dataloader=val_dl,
        num_classes=len(SCVIC_CIDS_CLASSES),
        device=experiment_config["device"]
    )
    misc.save_confusion_matrix(val_confussion, ["benign", "malicious"], save_path + "val_confussion.csv", index=[f"true_{k}" for k in SCVIC_CIDS_CLASSES.keys()])

    test_confussion = eval_fnc(
        model_path=experiment_config["model_path"],
        model_config=hp_config["model"],
        dataloader=test_dl,
        num_classes=len(SCVIC_CIDS_CLASSES),
        device=experiment_config["device"]
    )
    misc.save_confusion_matrix(test_confussion, ["benign", "malicious"], save_path + "test_confussion.csv", index=[f"true_{k}" for k in SCVIC_CIDS_CLASSES.keys()])

if __name__ == "__main__":
    """
    In this script we will perform the leaf-x-out experiment. The goal is to train a model on all classes, but 
    a predefined subset of classes. We will then evaluate the performance of the model on the left out class. If a model_path
    is given in the config we assume that the model is already trained and only run the test-phase
    """

    # Load the configuration file
    parser = argparse.ArgumentParser(description='baseline-binary Supervision')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the config file')
    parser.add_argument('--test_only', action="store_true", help="Only run the test phase")
    parser.add_argument('--train_only', action="store_true",)
    args = parser.parse_args()

    args = parser.parse_args()
    assert not (args.test_only & args.train_only), "cannot train and test only"
    log_dir = os.path.join(misc.root(), "logs/01_supervision/")
    os.makedirs(log_dir, exist_ok=True)
    logger = util.setup_logger(os.path.join(log_dir , f"baseline-binary.log"), level=logging.INFO)

    logger.info("Load config file")
    config = read(args.config, convert_tuple=False)

    torch.manual_seed(config["experiment"]["seed"])

    if not args.test_only:
        logger.info(f"Training model on all classes")
        # train model, in train_fnc model is trained and best version saved at path constructed from experiment_config
        config["experiment"]["model_path"] = train(hp_config=config["hp_config"], ds_config=config["dataset"], experiment_config=config["experiment"])

    if not args.train_only:
        logger.info(f"Testing model on all classes")
        test(hp_config=config["hp_config"], ds_config=config["dataset"], experiment_config=config["experiment"])
    logger.info("Experiment done")

    


