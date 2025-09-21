from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parents[2]))


import argparse
import logging
import os
import random

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from cids.models.nn import MLP, CollaborativeIDSNet
from cids.eval.supervised import eval_rf_confusion, eval_rf_confusion_binary
from cids.data import SCVIC_CIDS_CLASSES, get_SCVIC_Dataset, SCVICCIDSDataset, get_SCVIC_DatasetBinary, SCVICCIDSDatasetBinary
from cids.util import misc_funcs as misc, metrics
from cids.util.config import read
import util

def convert_ds(ds):
    return (
        np.array([np.concatenate([x[i].numpy().flatten() for i in range(len(x) - 1)]) for x in ds]),
        np.array([y[-1] for y in ds])            
    )

def test(hp_config: dict, ds_config: dict, experiment_config: dict, binary=False):
    
    save_path = os.path.join(misc.root(), f"results/01_supervision/leaf-x-out-binary/SCVIC/{experiment_config["model_type"]}/{experiment_config["id"]}/{experiment_config["trial"]}/")
    os.makedirs(save_path, exist_ok=True)

    # create dataloader
    exclude = ds_config["exclude"]
    exclude_labels = [SCVIC_CIDS_CLASSES[c] for c in exclude]

    logger.info("Load data fro training")
    if binary:
        train_ds = get_SCVIC_DatasetBinary(exclude=exclude, train=True, validation=True, test=False, random_state=ds_config["random_state"], network_data=ds_config["network_data"],
                            host_data=ds_config["host_data"], host_embeddings=ds_config["host_embeddings"])["train"]
    else:
        train_ds = get_SCVIC_Dataset(exclude=exclude, train=True, validation=True, test=False, random_state=ds_config["random_state"], network_data=ds_config["network_data"],
                                host_data=ds_config["host_data"], host_embeddings=ds_config["host_embeddings"])["train"]
    train_ds = convert_ds(train_ds)

    logger.info("Fitting Model")
    logger.debug(f"Train highest class: {np.max(train_ds[1])}")
    rf = RandomForestClassifier(**hp_config["model"])
    rf.fit(train_ds[0], train_ds[1])

    logger.info("Load data for evaluation")
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

    train_ds = convert_ds(train_ds)
    val_ds = convert_ds(val_ds)
    test_ds = convert_ds(test_ds)

    eval_fnc = eval_rf_confusion if not binary else eval_rf_confusion_binary

    logger.info("Calculate confusion matricies")
    train_confussion = eval_fnc(
        model=rf,
        X=train_ds[0],
        y=train_ds[1],
        num_classes=len(SCVIC_CIDS_CLASSES)
    )
    misc.save_confusion_matrix(train_confussion, ["benign", "malicious"], path=save_path + "train_confussion.csv", index=list(SCVIC_CIDS_CLASSES.keys()))

    val_confussion = eval_fnc(
        model=rf,
        X=val_ds[0],
        y=val_ds[1],
        num_classes=len(SCVIC_CIDS_CLASSES)
    )
    misc.save_confusion_matrix(val_confussion, ["benign", "malicious"], save_path + "val_confussion.csv", index= list(SCVIC_CIDS_CLASSES.keys()), )

    test_confussion = eval_fnc(
        model=rf,
        X=test_ds[0],
        y=test_ds[1],
        num_classes=len(SCVIC_CIDS_CLASSES)
    )
    misc.save_confusion_matrix(test_confussion, ["benign", "malicious"], save_path + "test_confussion.csv", index= list(SCVIC_CIDS_CLASSES.keys()), )

if __name__ == "__main__":
    """
    In this script we will perform the leaf-x-out experiment. The goal is to train a model on all classes, but 
    a predefined subset of classes. We will then evaluate the performance of the model on the left out class. If a model_path
    is given in the config we assume that the model is already trained and only run the test-phase
    """

    # Load the configuration file
    parser = argparse.ArgumentParser(description='Leaf-one-out experiment')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the config file')
    parser.add_argument('--binary', action="store_true")
    args = parser.parse_args()

    args = parser.parse_args()
    log_dir = os.path.join(misc.root(), "logs/01_supervision/")
    os.makedirs(log_dir, exist_ok=True)
    logger = util.setup_logger(os.path.join(log_dir , f"leafe_x_out.log"), level=logging.INFO)

    logger.info("Load config file")
    config = read(args.config, convert_tuple=False)

    random.seed(config["experiment"]["seed"])
    np.random.seed(config["experiment"]["seed"])
    
    logger.info(f"Testing model while leaving out classes: {config['dataset']['exclude']}")
    test(hp_config=config["hp_config"], ds_config=config["dataset"], experiment_config=config["experiment"], binary=args.binary)
    logger.info("Experiment done")
