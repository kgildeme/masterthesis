from pathlib import Path
import logging

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..anomaly_scores import AnomalyBase
from ..data import OpTCDataset

logger = logging.getLogger(__name__)

def eval_autoregressive_encoder(anomaly: AnomalyBase, pred_n: int,
                                score_file_dict: str | Path = None, data_config_train = None, data_config_eval = None,
                                batch_size: int = 32, n_worker: int = 0, worker_init: callable = None, device: str = "cpu"):
    
    """
    Evaluate the performance of an autoregressive model. That should predict n steps in the future 
    Returns: dict containing amount of labelling
    """

    if isinstance(score_file_dict, str):
        score_file_dict = Path(score_file_dict)

    if not score_file_dict.exists():
        score_file_dict.mkdir(parents=True, exist_ok=True)
    
    benign_train_score_file = score_file_dict / "benign_train.npy"
    benign_eval_score_file = score_file_dict / "benign_eval.npy"
    malicious_eval_score_file = score_file_dict / "malicious_eval.npy"

    if not benign_train_score_file.exists():
        logger.info("Start calculating anomaly scores for trainingsdata")

        data_config_train["window_size"] = data_config_train["window_size"] + pred_n
        train_dataset = OpTCDataset(**data_config_train)
        dl = DataLoader(train_dataset, batch_size=batch_size, num_workers=n_worker, worker_init_fn=worker_init)

        distances = []

        for i, inp in enumerate(dl):
            x, y = inp[:, :-pred_n, :], inp[:, -pred_n:, :]
            x = x.to(device)
            scores = anomaly.get_scores(x, y)

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

    data_config_eval["window_size"] = data_config_eval["window_size"] + pred_n
    eval_dataset = OpTCDataset(**data_config_eval)
    dl = DataLoader(eval_dataset, batch_size=batch_size, num_workers=n_worker, worker_init_fn=worker_init)

    benign_scores = []
    malicious_scores = []
    tp, fp, fn, tn = 0, 0, 0, 0

    for i,(inp, label) in enumerate(dl):
        x, y = inp[:, :-pred_n, :], inp[:, -pred_n:, :]
        x = x.to(device)

        prediction, scores = anomaly.classify(x, y)
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

def eval_next_window_encoder(anomaly: AnomalyBase, tau: int = 1,score_file_dict: str | Path = None, data_config_train = None, data_config_eval = None,
                                batch_size: int = 32, n_worker: int = 0, worker_init: callable = None, device: str = "cpu"):
    """ Predict the next window shiftet by tau steps and evaluate the performance of the model."""
    if isinstance(score_file_dict, str):
        score_file_dict = Path(score_file_dict)

    if not score_file_dict.exists():
        score_file_dict.mkdir(parents=True, exist_ok=True)
    
    benign_train_score_file = score_file_dict / "benign_train.npy"
    benign_eval_score_file = score_file_dict / "benign_eval.npy"
    malicious_eval_score_file = score_file_dict / "malicious_eval.npy"

    if not benign_train_score_file.exists():
        logger.info("Start calculating anomaly scores for trainingsdata")

        data_config_train["window_size"] = data_config_train["window_size"] + tau
        train_dataset = OpTCDataset(**data_config_train)
        dl = DataLoader(train_dataset, batch_size=batch_size, num_workers=n_worker, worker_init_fn=worker_init)

        distances = []

        for i, inp in enumerate(dl):
            x, y = inp[:, :-tau, :], inp[:, tau:, :]
            x = x.to(device)
            scores = anomaly.get_scores(x, y)

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

    data_config_eval["window_size"] = data_config_eval["window_size"] + tau
    eval_dataset = OpTCDataset(**data_config_eval)
    dl = DataLoader(eval_dataset, batch_size=batch_size, num_workers=n_worker, worker_init_fn=worker_init)

    benign_scores = []
    malicious_scores = []
    tp, fp, fn, tn = 0, 0, 0, 0

    for i,(inp, label) in enumerate(dl):
        x, y = inp[:, :-tau, :], inp[:, tau:, :]
        x = x.to(device)

        prediction, scores = anomaly.classify(x, y)
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