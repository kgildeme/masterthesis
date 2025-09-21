import logging
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader 

from ..anomaly_scores import AnomalyBase
from ..data import OpTCDataset

logger = logging.getLogger(__name__)

def set_threshold_anomaly(anomaly: AnomalyBase, dl: DataLoader, iqr: bool =True, device: str = "cpu"):
    benign_scores = []
    malicious_scores = []
    
    if not iqr:
        logger.info("get_scores_for threshold")
        for data in dl:
            lbl = data[-1]
            if len(data) > 2:
                inp = torch.concat([torch.flatten(data[i], start_dim=1) for i in range(len(data) - 1)], dim=1)
            else:
                inp = torch.flatten(data[0], start_dim=1)

            inp = inp.to(device)

            scores = anomaly.get_scores(inp)
            benign_scores += scores[~lbl.bool()].tolist()
            malicious_scores += scores[lbl.bool()].tolist()
        logger.debug(f"{torch.tensor(benign_scores).shape}, {torch.tensor(malicious_scores).shape}")
        anomaly.set_threshold(torch.tensor(benign_scores), IQR=True)
    else:
        for data in dl:
            lbl = data[-1]
            if len(data) > 2:
                inp = torch.concat([torch.flatten(data[i], start_dim=1) for i in range(len(data) - 1)], dim=1)
            else:
                inp = torch.flatten(data[0], start_dim=1)
            
            inp = inp.to(device)

            scores = anomaly.get_scores(inp)
            logger.debug(f"Scores: {scores}")
            benign_scores += scores[~lbl.bool()].tolist()
        anomaly.set_threshold(torch.tensor(benign_scores), torch.tensor(malicious_scores), IQR=False)

def eval_autoencoder_multiclass(anomaly: AnomalyBase, class_map: dict[int: str], score_file_dict: str | Path, dataloader: DataLoader = None, device: str = "cpu", force=False):

    scores = {k: [] for k in class_map.values()}
    confusion_matrix = torch.zeros(len(class_map), 2, device=device)
    inv_class_map = {v: k for k, v in class_map.items()}

    if all(os.path.exists(os.path.join(score_file_dict, f"{k.lower()}_scores.npy")) for k in scores.keys()) and not force:
        for k in scores.keys():
            lbl = inv_class_map[k]
            score = np.load(os.path.join(score_file_dict, f"{k.lower()}_scores.npy"))
            score = torch.tensor(score, device=device)
            lbl = lbl * torch.ones_like(score)
            prediction, score = anomaly.classify(score, score_mode=True)

            index = lbl * 2 + prediction
            index = index.long()
            ones = torch.ones_like(index, dtype=torch.float).to(device)
            confusion_matrix.view(-1).scatter_add_(0, index, ones)

    else:
        for data in dataloader:
            lbl = data[-1]
            if len(data) > 2:
                inp = torch.concat([torch.flatten(data[i], start_dim=1) for i in range(len(data) - 1)], dim=1)
            else:
                inp = torch.flatten(data[0], start_dim=1)
            
            inp = inp.to(device)
            lbl = lbl.to(device)

            prediction, score = anomaly.classify(inp)

            index = lbl * 2 + prediction
            index = index.long()
            ones = torch.ones_like(index, dtype=torch.float).to(device)
            confusion_matrix.view(-1).scatter_add_(0, index, ones)

            for l, s,  in zip(lbl.cpu(), score.cpu()):
                scores[class_map[l.item()]].append(s.item())

        for k, v in scores.items():
            with open(os.path.join(score_file_dict, f"{k.lower()}_scores.npy"), 'wb') as f:
                np.save(f, np.asarray(v))

    return confusion_matrix

def calculate_scores_ae_multiclass(anomaly:AnomalyBase, dataloader:DataLoader, class_map:dict[int: str], score_file_dict: str| Path, device = "cpu"):
    scores = {k: [] for k in class_map.values()}
    inv_class_map = {v: k for k, v in class_map.items()}

    for data in dataloader:
        lbl = data[-1]
        if len(data) > 2:
            inp = torch.concat([torch.flatten(data[i], start_dim=1) for i in range(len(data) - 1)], dim=1)
        else:
            inp = torch.flatten(data[0], start_dim=1)
        
        inp = inp.to(device)
        lbl = lbl.to(device)
        score = anomaly.get_scores(inp)

        for l, s,  in zip(lbl.cpu(), score.cpu()):
            scores[class_map[l.item()]].append(s.item())

    for k, v in scores.items():
        with open(os.path.join(score_file_dict, f"{k.lower()}_scores.npy"), 'wb') as f:
            np.save(f, np.asarray(v))


def eval_autoencoder_transformer(anomaly: AnomalyBase, score_file_dict: str | Path = None, data_config_train = None, data_config_eval = None,
                        batch_size: int = 32, n_worker: int = 0, worker_init: callable = None, device: str = "cpu", flatten=False):
    """ Evaluate the performance of a masked encoder model that predicts n steps in the future. """
    if isinstance(score_file_dict, str):
        score_file_dict = Path(score_file_dict)

    if not score_file_dict.exists():
        score_file_dict.mkdir(parents=True, exist_ok=True)
    
    benign_train_score_file = score_file_dict / "benign_train.npy"
    benign_eval_score_file = score_file_dict / "benign_eval.npy"
    malicious_eval_score_file = score_file_dict / "malicious_eval.npy"

    if not benign_train_score_file.exists():
        logger.info("Start calculating anomaly scores for trainingsdata")

        train_dataset = OpTCDataset(**data_config_train)
        dl = DataLoader(train_dataset, batch_size=batch_size, num_workers=n_worker, worker_init_fn=worker_init)

        distances = []

        for i, inp in enumerate(dl):
            if flatten:
                inp = torch.flatten(inp, start_dim=1)
            
            inp = inp.to(device)
            scores = anomaly.get_scores(inp)

            scores = scores
            scores = scores.tolist()
            distances += scores

            if i % 1000 == 0:
                logger.info(f"Batch {i} processed")
        
        distances = np.asarray(distances)
        with open(benign_train_score_file, 'wb') as f:
            np.save(f, distances)
        
        del distances, dl, train_dataset

    logger.info("Calculate IQR-based threshold")
    scores = np.load(benign_train_score_file)
    scores = torch.tensor(scores)
    threshold = anomaly.set_threshold(scores)

    eval_dataset = OpTCDataset(**data_config_eval)
    dl = DataLoader(eval_dataset, batch_size=batch_size, num_workers=n_worker, worker_init_fn=worker_init)

    benign_scores = []
    malicious_scores = []
    tp, fp, fn, tn = 0, 0, 0, 0

    for i,(x, label) in enumerate(dl):

        if flatten:
            x = torch.flatten(x, start_dim=1)
        x = x.to(device)

        prediction, scores = anomaly.classify(x)
        benign_scores += scores[~label.bool()].tolist()
        malicious_scores += scores[label.bool()].tolist()

        tp += (prediction * label).sum().item()
        fp += (prediction * (1-label)).sum().item()
        fn += ((1-prediction) * label).sum().item()
        tn += ((1-prediction) * (1-label)).sum().item()

        if i % 1000 == 0:
            logger.info(f"Batch {i} processed")

    benign_scores = np.asarray(benign_scores)
    malicious_scores = np.asarray(malicious_scores)

    with open(benign_eval_score_file, 'wb') as f:
        np.save(f, benign_scores)
    with open(malicious_eval_score_file, 'wb') as f:
        np.save(f, malicious_scores)
    
    return {"threshold": threshold, "TP": tp, "FP": fp, "FN": fn, "TN": tn}