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
from torch.utils.tensorboard import SummaryWriter

from cids.anomaly_scores import ReconstructionAnomaly
from cids.models.nn import MLP, CollaborativeIDSNet, Autoencoder
from cids.training.unsupervised import train_mlp_autoencoder
from cids.eval.unsupervised import eval_autoencoder_multiclass, set_threshold_anomaly, calculate_scores_ae_multiclass
from cids.data import SCVIC_CIDS_CLASSES, get_SCVIC_Dataset, SCVICCIDSDataset, SCVIC_CIDS_CLASSES_INV,  OpTCDataset, worker_init
from cids.util import misc_funcs as misc, metrics
from cids.util.config import read
import util

def train(hp_config: dict, ds_config: dict, experiment_config: dict):

    trainable = train_mlp_autoencoder

    model_path = os.path.join(misc.root(), f"models/02_anomaly-detection/baseline/{ds_config["ds_name"]}/{experiment_config["model_type"]}/{experiment_config["id"]}/{experiment_config["trial"]}/ckpt.pt")
    tensorboard_path = os.path.join(misc.root(), f"results/02_anomaly-detection/baseline/{ds_config["ds_name"]}/{experiment_config["model_type"]}/{experiment_config["id"]}/{experiment_config["trial"]}/tensorboard/")
    writer = SummaryWriter(tensorboard_path)

    os.makedirs(os.path.split(model_path)[0], exist_ok=True)
    os.makedirs(tensorboard_path, exist_ok=True)
    ds_name = ds_config["ds_name"]
    if "scvic" in ds_name.lower():
        eval_ds_config = None
    elif "optc" in ds_name.lower():
        eval_ds_config = copy.deepcopy(ds_config["eval"])
        ds_config = copy.deepcopy(ds_config["train"])
    md_train_config =copy.deepcopy(hp_config)

    trainable_args = {
        "model_config": md_train_config.pop("model"),
        "ds_name": ds_name, 
        "ds_args": ds_config,
        "eval_ds_args": eval_ds_config,
        "eval_metric": metrics.auc_score, 
        "checkpoint_path": model_path,
        "tensorboard_writer": writer,
        "device": experiment_config["device"]
    }

    trainable_args = trainable_args | md_train_config
    trainable(**trainable_args)
    return model_path

def test(hp_config: dict, ds_config: dict, experiment_config: dict, create_confusion_only=False):
    eval_fnc = eval_autoencoder_multiclass

    if "model_path" not in experiment_config.keys():
        experiment_config["model_path"] = os.path.join(misc.root(), f"models/02_anomaly-detection/baseline/{ds_config["ds_name"]}/{experiment_config["model_type"]}/{experiment_config["id"]}/{experiment_config["trial"]}/ckpt.pt")
    # create dataloader
    ds_name = ds_config["ds_name"]
    if "scvic" in ds_name.lower():
        dss = get_SCVIC_Dataset(exclude=None, train=True, validation=True, test=True, random_state=ds_config["seed"], network_data=ds_config["network_data"],
                                host_data=ds_config["host_data"], host_embeddings=ds_config["host_embeddings"])
        
        val_ds = dss["val"]
        train_ds = dss["train"]
        test_ds = dss["test"]

        for point in train_ds.data:
            if point[-1]  != 0:
                val_ds.data.append(point)
        
        train_ds = SCVICCIDSDataset(data=train_ds.data, exclude_classes=[SCVIC_CIDS_CLASSES[c] for c in SCVIC_CIDS_CLASSES.keys() if c != "Benign"],
                                    network_data=ds_config["network_data"],
                                    host_data=ds_config["host_data"], host_embeddings=ds_config["host_embeddings"])
        

        train_dl = torch.utils.data.DataLoader(train_ds, hp_config["batch_size"], num_workers=ds_config["n_worker"])
        val_dl = torch.utils.data.DataLoader(val_ds, hp_config["batch_size"], num_workers=ds_config["n_worker"])
        test_dl = torch.utils.data.DataLoader(test_ds, hp_config["batch_size"], num_workers=ds_config["n_worker"])
    elif "optc" in ds_name.lower():
        train_ds = copy.deepcopy(ds_config["train"])
        train_ds["parts"] = 16
        train_ds.pop("last_part")
        train_ds["eval_mode"] = True
        eval_ds = copy.deepcopy(train_ds)
        eval_ds["ds_name"] = eval_ds["ds_name"].replace("train", "eval")
        logger.info(eval_ds["ds_name"])

        train_dataset = OpTCDataset(**train_ds)
        eval_dataset = OpTCDataset(**eval_ds)
        train_dl = torch.utils.data.DataLoader(train_dataset,  hp_config["batch_size"], num_workers=0)
        val_dl = torch.utils.data.DataLoader(eval_dataset,  hp_config["batch_size"], num_workers=0)
        test_dl = None

    save_path = os.path.join(misc.root(), f"results/02_anomaly-detection/baseline/{ds_config["ds_name"]}/{experiment_config["model_type"]}/{experiment_config["id"]}/{experiment_config["trial"]}/")
    if not create_confusion_only:
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
        set_threshold_anomaly(anomaly=anomaly, dl=val_dl, iqr=True, device=experiment_config["device"])
    
    logger.info("Evaluate model performance")
    class_map = SCVIC_CIDS_CLASSES_INV if "scvic" in ds_name else {0: "benign",1: "malicious"}
    train_confussion = eval_fnc(
        anomaly=anomaly,
        class_map=class_map,
        score_file_dict=os.path.join(save_path, "scores_train"),
        dataloader=train_dl,
        device=experiment_config["device"]
    )
    misc.save_confusion_matrix(train_confussion, ["benign", "malicious"], save_path + "train_confusion.csv", index=[f"true_{k}" for k in SCVIC_CIDS_CLASSES.keys()] if "scvic" in ds_name else ["true_benign", "true_malicious"])

    val_confusion = eval_fnc(
        anomaly=anomaly,
        class_map=class_map,
        score_file_dict=os.path.join(save_path, "scores_val"),
        dataloader=val_dl,
        device=experiment_config["device"]
    )
    misc.save_confusion_matrix(val_confusion, ["benign", "malicious"], save_path + "val_confusion.csv", index=[f"true_{k}" for k in SCVIC_CIDS_CLASSES.keys()] if "scvic" in ds_name else ["true_benign", "true_malicious"])

    if test_dl is not None:
        test_confusion = eval_fnc(
            anomaly=anomaly,
            class_map=class_map,
            score_file_dict=os.path.join(save_path, "scores_test"),
            dataloader=test_dl,
            device=experiment_config["device"]
        )
        misc.save_confusion_matrix(test_confusion, ["benign", "malicious"], save_path + "test_confusion.csv", index=[f"true_{k}" for k in SCVIC_CIDS_CLASSES.keys()] if "scvic" in ds_name else ["true_benign", "true_malicious"])

def test_from_scores(hp_config: dict, ds_config: dict, experiment_config: dict, recalculate=False):
    save_path = os.path.join(misc.root(), f"results/02_anomaly-detection/baseline/{ds_config["ds_name"]}/{experiment_config["model_type"]}/{experiment_config["id"]}/{experiment_config["trial"]}/")
    class_map = SCVIC_CIDS_CLASSES_INV
    inv_class_map = {v: k for k, v in class_map.items()}
    if recalculate:
        if "model_path" not in experiment_config.keys():
            experiment_config["model_path"] = os.path.join(misc.root(), f"models/02_anomaly-detection/baseline/{ds_config["ds_name"]}/{experiment_config["model_type"]}/{experiment_config["id"]}/{experiment_config["trial"]}/ckpt.pt")
        # create dataloader
        ds_name = ds_config["ds_name"]
        if "scvic" in ds_name.lower():
            dss = get_SCVIC_Dataset(exclude=None, train=True, validation=True, test=True, random_state=ds_config["seed"], network_data=ds_config["network_data"],
                                    host_data=ds_config["host_data"], host_embeddings=ds_config["host_embeddings"])
            
            val_ds = dss["val"]
            train_ds = dss["train"]
            test_ds = dss["test"]

            for point in train_ds.data:
                if point[-1]  != 0:
                    val_ds.data.append(point)
            
            train_ds = SCVICCIDSDataset(data=train_ds.data, exclude_classes=[SCVIC_CIDS_CLASSES[c] for c in SCVIC_CIDS_CLASSES.keys() if c != "Benign"],
                                        network_data=ds_config["network_data"],
                                        host_data=ds_config["host_data"], host_embeddings=ds_config["host_embeddings"])
            

            train_dl = torch.utils.data.DataLoader(train_ds, hp_config["batch_size"], num_workers=ds_config["n_worker"])
            val_dl = torch.utils.data.DataLoader(val_ds, hp_config["batch_size"], num_workers=ds_config["n_worker"])
            test_dl = torch.utils.data.DataLoader(test_ds, hp_config["batch_size"], num_workers=ds_config["n_worker"])

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

        calculate_scores_ae_multiclass(
            anomaly=anomaly,
            class_map=class_map,
            score_file_dict=os.path.join(save_path, "scores_train"),
            dataloader=train_dl,
            device=experiment_config["device"]
        )
        calculate_scores_ae_multiclass(
            anomaly=anomaly,
            class_map=class_map,
            score_file_dict=os.path.join(save_path, "scores_val"),
            dataloader=val_dl,
            device=experiment_config["device"]
        )
        calculate_scores_ae_multiclass(
            anomaly=anomaly,
            class_map=class_map,
            score_file_dict=os.path.join(save_path, "scores_test"),
            dataloader=test_dl,
            device=experiment_config["device"]
        )

    train_scores = np.load(os.path.join(save_path, "scores_train", "benign_scores.npy"))
    q1 = np.quantile(train_scores, 0.25)
    q3 = np.quantile(train_scores, 0.75)
    iqr = q3 - q1
    threshold = q3 + 1.5 * iqr
    logger.info(f"Threshold: {threshold}")

    for mode in ["val", "test"]:
        logger.info(f"Evaluate model performance on {mode} set")
        score_file_dict = os.path.join(save_path, f"scores_{mode}")
        scores = {k: [] for k in class_map.values()}
        confusion_matrix = torch.zeros(len(class_map), 2)

        for k in scores.keys():
            lbl = inv_class_map[k]
            score = np.load(os.path.join(score_file_dict, f"{k.lower()}_scores.npy"))
            score = torch.tensor(score)
            lbl = lbl * torch.ones_like(score)
            prediction = (score > threshold).int() 

            index = lbl * 2 + prediction
            index = index.long()
            ones = torch.ones_like(index, dtype=torch.float)
            confusion_matrix.view(-1).scatter_add_(0, index, ones)

        misc.save_confusion_matrix(confusion_matrix, ["benign", "malicious"], os.path.join(save_path, f"{mode}_confusion.csv"), index=[f"true_{k}" for k in SCVIC_CIDS_CLASSES.keys()])

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
    logger = util.setup_logger(os.path.join(log_dir , f"baseline.log"), level=logging.INFO)

    logger.info("Load config file")

    config = read(args.config, convert_tuple=False)
    logger.info(f"Run on device {config["experiment"]["device"]}")
    torch.manual_seed(config["experiment"]["seed"])

    if not args.test_only:
        logger.info(f"Training model with all classes")
        # train model, in train_fnc model is trained and best version saved at path constructed from experiment_config
        config["experiment"]["model_path"] = train(hp_config=config["hp_config"], ds_config=config["dataset"], experiment_config=config["experiment"])

    logger.info(f"Testing model with all classes")
    test_from_scores(hp_config=config["hp_config"], ds_config=config["dataset"], experiment_config=config["experiment"], recalculate=True)
    logger.info("Experiment done")