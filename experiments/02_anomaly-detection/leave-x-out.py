from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parents[2]))


import argparse
import logging
import os

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter

from cids.anomaly_scores import ReconstructionAnomaly
from cids.models.nn import MLP, CollaborativeIDSNet, Autoencoder
from cids.training.unsupervised import train_mlp_autoencoder
from cids.eval.unsupervised import eval_autoencoder_multiclass, set_threshold_anomaly
from cids.data import SCVIC_CIDS_CLASSES, get_SCVIC_Dataset, SCVICCIDSDataset, SCVIC_CIDS_CLASSES_INV
from cids.util import misc_funcs as misc, metrics
from cids.util.config import read
import util

def test(hp_config: dict, ds_config: dict, experiment_config: dict):
    eval_fnc = eval_autoencoder_multiclass

    if "model_path" not in experiment_config.keys():
        experiment_config["model_path"] = os.path.join(misc.root(), f"models/02_anomaly-detection/baseline/MLPAE/MLPAE-2/MLPAE-2-{int(experiment_config["trial"].split("-")[-1])}/ckpt.pt")
    # create dataloader
    exclude = ds_config["exclude"]
    exclude_labels = [SCVIC_CIDS_CLASSES[c] for c in SCVIC_CIDS_CLASSES.keys() if c != "Benign"]
    exclude = [SCVIC_CIDS_CLASSES[c] for c in exclude]

    dss = get_SCVIC_Dataset(exclude=None, train=True, validation=True, test=True, random_state=ds_config["seed"], network_data=ds_config["network_data"],
                            host_data=ds_config["host_data"], host_embeddings=ds_config["host_embeddings"])
    
    val_ds = dss["val"]
    train_ds = dss["train"]
    test_ds = dss["test"]
    
    non_test_labels = [l for l in exclude_labels if l not in exclude]
    for point in train_ds.data:
        if point[-1] in non_test_labels:
            val_ds.data.append(point)
        elif point[-1] in exclude:
            test_ds.data.append(point)
    
    for point in val_ds.data:
        if point[-1] in exclude:
            test_ds.data.append(point)
    
    train_ds = SCVICCIDSDataset(data=train_ds.data, exclude_classes=exclude_labels, network_data=ds_config["network_data"],
                            host_data=ds_config["host_data"], host_embeddings=ds_config["host_embeddings"])
    
    val_ds = SCVICCIDSDataset(data=val_ds.data, exclude_classes=exclude, network_data=ds_config["network_data"],
                            host_data=ds_config["host_data"], host_embeddings=ds_config["host_embeddings"])

    train_dl = torch.utils.data.DataLoader(train_ds, ds_config["batch_size"], num_workers=ds_config["n_worker"])
    val_dl = torch.utils.data.DataLoader(val_ds, ds_config["batch_size"], num_workers=ds_config["n_worker"])
    test_dl = torch.utils.data.DataLoader(test_ds, ds_config["batch_size"], num_workers=ds_config["n_worker"])
    save_path = os.path.join(misc.root(), f"results/02_anomaly-detection/leave-x-out/{experiment_config["model_type"]}/{experiment_config["id"]}/{experiment_config["trial"]}/")
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(os.path.join(save_path, "scores_train"), exist_ok=True)
    os.makedirs(os.path.join(save_path, "scores_val"), exist_ok=True)
    os.makedirs(os.path.join(save_path, "scores_test"), exist_ok=True)

    logger.info("Load model")
    model_config = hp_config["model"]
    encoder = MLP(**model_config)
    model_config["input_dim"], model_config["output_dim"] = model_config["output_dim"], model_config["input_dim"]
    decoder = MLP(**model_config)
    decoder = nn.Sequential(decoder, nn.Sigmoid())

    model = Autoencoder(encoder=encoder, decoder=decoder)
    ckpt = torch.load(experiment_config["model_path"], weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    del ckpt
    model.to(device=experiment_config["device"])
    anomaly = ReconstructionAnomaly(model=model, device=experiment_config["device"])

    logger.info("Start calculating threshold")
    set_threshold_anomaly(anomaly=anomaly, train_dl=train_dl, val_dl=val_dl, iqr=False, device=experiment_config["device"])

    logger.info("Evaluate model performance")
    train_confussion = eval_fnc(
        anomaly=anomaly,
        class_map=SCVIC_CIDS_CLASSES_INV,
        score_file_dict=os.path.join(save_path, "scores_train"),
        dataloader=train_dl,
        device=experiment_config["device"]
    )
    misc.save_confusion_matrix(train_confussion, ["benign", "malicious"], save_path + "train_confusion.csv", index=[f"true_{k}" for k in SCVIC_CIDS_CLASSES.keys()])

    val_confusion = eval_fnc(
        anomaly=anomaly,
        class_map=SCVIC_CIDS_CLASSES_INV,
        score_file_dict=os.path.join(save_path, "scores_val"),
        dataloader=val_dl,
        device=experiment_config["device"]
    )
    misc.save_confusion_matrix(val_confusion, ["benign", "malicious"], save_path + "val_confusion.csv", index=[f"true_{k}" for k in SCVIC_CIDS_CLASSES.keys()])

    test_confusion = eval_fnc(
        anomaly=anomaly,
        class_map=SCVIC_CIDS_CLASSES_INV,
        score_file_dict=os.path.join(save_path, "scores_test"),
        dataloader=test_dl,
        device=experiment_config["device"]
    )
    misc.save_confusion_matrix(test_confusion, ["benign", "malicious"], save_path + "test_confusion.csv", index=[f"true_{k}" for k in SCVIC_CIDS_CLASSES.keys()])

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
    log_dir = os.path.join(misc.root(), "logs/02_anomaly-detection/")
    os.makedirs(log_dir, exist_ok=True)
    logger = util.setup_logger(os.path.join(log_dir , f"leafe_x_out.log"), level=logging.INFO)

    logger.info("Load config file")
    config = read(args.config, convert_tuple=False)

    torch.manual_seed(config["experiment"]["seed"])

    logger.info(f"Testing model while leaving out classes: {config['dataset']['exclude']}")
    test(hp_config=config["hp_config"], ds_config=config["dataset"], experiment_config=config["experiment"])
    logger.info("Experiment done")
