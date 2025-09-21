import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from ..data import OpTCDataset, worker_init
from ..util import metrics
from ..models.nn import MLP, CollaborativeIDSNet
from ..models.transformer import TransformerEncoder

import logging
logger = logging.getLogger(__name__)

def eval_mlp(model, data_config, device, loss_fnc):
    train_dataset = OpTCDataset(**data_config)
    dl = DataLoader(train_dataset, batch_size=32, num_workers=1, worker_init_fn=worker_init)
    model.eval()
    total_loss = 0
    total_acc = 0
    total_steps = 0
    tp, fp, tn, fn = 0, 0, 0, 0
    with torch.no_grad():
        for inp, label in dl:
            inp = torch.flatten(inp, start_dim=1)
            inp = inp.to(device)
            label = label.unsqueeze(1).to(dtype=torch.float32, device=device)
            out = model(inp)
            loss = loss_fnc(out, label)
            prediction = (F.sigmoid(out) > 0.5).int()
            eq = prediction == label
            total_loss += loss.item()
            total_acc += eq.float().mean().item()
            tp += (prediction * label).sum().item()
            fp += (prediction * (1-label)).sum().item()
            fn += ((1-prediction) * label).sum().item()
            tn += ((1-prediction) * (1-label)).sum().item()
            total_steps += 1

    return {"loss": total_loss / total_steps,
            "acc": total_acc / total_steps,
            "TP": tp, "FP": fp, "FN": fn, "TN": tn} 

def eval_mlp_confusion(model_path, model_config, dataloader, num_classes, device, left_out_mask: torch.Tensor = None):

    ckpt = torch.load(model_path, weights_only=True)
    model = MLP(**model_config)
    model.load_state_dict(ckpt["model_state_dict"])
    del ckpt
    model.to(device)
    model.eval()

    if left_out_mask is None:
        left_out_mask = torch.ones(num_classes, device=device)

    confusion_matrix = torch.zeros(num_classes, num_classes, device=device)
    with torch.no_grad():
        for data in dataloader:
            label = data[-1]
            if len(data) == 2:
                inp = torch.flatten(data[0], start_dim=1)
            else:
                inps = [torch.flatten(data[i], start_dim=1) for i in range(len(data) - 1)]
                inp = torch.cat(inps, dim=1)

            label = label.to(device)
            inp = inp.to(device)
            
            out = model(inp)
            expanded_logits = torch.full((out.shape[0], len(left_out_mask)), float('-inf')).to(device)  # Fill missing classes with -inf
            present_classes = torch.nonzero(left_out_mask, as_tuple=True)[0]
            expanded_logits[:, present_classes] = out
            pred = torch.argmax(expanded_logits, dim=-1)

            # Use scatter_add to efficiently accumulate confusion matrix counts
            index = label * num_classes + pred
            index = index.long()
            ones = torch.ones_like(index, dtype=torch.float).to(device)
            confusion_matrix.view(-1).scatter_add_(0, index, ones)

    return confusion_matrix

def eval_mlp_confusion_binary(model_path, model_config, dataloader, num_classes, device):

    ckpt = torch.load(model_path, weights_only=True)
    model = MLP(**model_config)
    model.load_state_dict(ckpt["model_state_dict"])
    del ckpt
    model.to(device)
    model.eval()

    confusion_matrix = torch.zeros(num_classes, 2, device=device)
    with torch.no_grad():
        for data in dataloader:
            label = data[-1]
            if len(data) == 2:
                inp = torch.flatten(data[0], start_dim=1)
            else:
                inps = [torch.flatten(data[i], start_dim=1) for i in range(len(data) - 1)]
                inp = torch.cat(inps, dim=1)

            label = label.to(device)
            inp = inp.to(device)
            
            out = model(inp)
            pred = torch.argmax(out, dim=-1)

            # Use scatter_add to efficiently accumulate confusion matrix counts
            index = label * 2 + pred
            index = index.long()
            ones = torch.ones_like(index, dtype=torch.float).to(device)
            confusion_matrix.view(-1).scatter_add_(0, index, ones)

    return confusion_matrix

def eval_collaborative_confusion(model_path, model_config, dataloader, num_classes, device, left_out_mask: torch.Tensor = None):

    ckpt = torch.load(model_path, weights_only=True)
    network_config = model_config["network"]
    host_config = model_config["host"]
    embedding_config = model_config["embedding"]
    aggregation_config = model_config["aggregation"]
    model = CollaborativeIDSNet(
        network_encoder=MLP(**network_config),
        host_encoder=TransformerEncoder(**host_config),
        embedding_encoder=TransformerEncoder(**embedding_config),
        aggregation_module=MLP(**aggregation_config)
    )
    model.load_state_dict(ckpt["model_state_dict"])
    del ckpt
    model.to(device)
    model.eval()

    if left_out_mask is None:
        left_out_mask = torch.ones(num_classes, device=device)

    confusion_matrix = torch.zeros(num_classes, num_classes, device=device)
    with torch.no_grad():
        for data in dataloader:
            label = data[-1]

            inp_network = data[0].to(device)
            inp_host = data[1].to(device)
            inp_embedding = data[2].to(device)
            label = label.to(device=device)

            out, _ = model(inp_network, inp_host, inp_embedding)
            expanded_logits = torch.full((out.shape[0], len(left_out_mask)), float('-inf')).to(device)  # Fill missing classes with -inf
            present_classes = torch.nonzero(left_out_mask, as_tuple=True)[0]
            expanded_logits[:, present_classes] = out
            pred = torch.argmax(expanded_logits, dim=-1)

            # Use scatter_add to efficiently accumulate confusion matrix counts
            index = label * num_classes + pred
            index = index.long()
            ones = torch.ones_like(index, dtype=torch.float).to(device)
            confusion_matrix.view(-1).scatter_add_(0, index, ones)

    return confusion_matrix

def eval_collaborative_confusion_binary(model_path, model_config, dataloader, num_classes, device):

    ckpt = torch.load(model_path, weights_only=True)
    network_config = model_config["network"]
    host_config = model_config["host"]
    embedding_config = model_config["embedding"]
    aggregation_config = model_config["aggregation"]
    model = CollaborativeIDSNet(
        network_encoder=MLP(**network_config),
        host_encoder=TransformerEncoder(**host_config),
        embedding_encoder=TransformerEncoder(**embedding_config),
        aggregation_module=MLP(**aggregation_config)
    )
    model.load_state_dict(ckpt["model_state_dict"])
    del ckpt
    model.to(device)
    model.eval()

    confusion_matrix = torch.zeros(num_classes, 2, device=device)
    with torch.no_grad():
        for data in dataloader:
            label = data[-1]

            inp_network = data[0].to(device)
            inp_host = data[1].to(device)
            inp_embedding = data[2].to(device)
            label = label.to(device=device)

            out, _ = model(inp_network, inp_host, inp_embedding)
            pred = torch.argmax(out, dim=-1)

            # Use scatter_add to efficiently accumulate confusion matrix counts
            index = label * 2 + pred
            index = index.long()
            ones = torch.ones_like(index, dtype=torch.float).to(device)
            confusion_matrix.view(-1).scatter_add_(0, index, ones)

    return confusion_matrix

def eval_rf_confusion(model: RandomForestClassifier, X: np.ndarray, y: np.ndarray, num_classes: int, left_out_mask: np.ndarray = None):

    confusion_matrix = np.zeros((num_classes, num_classes))

    y_hat = model.predict(X).reshape(-1)
    logger.debug(f"y_hat max label: {np.max(y_hat)}")
    y_hat = np.eye(left_out_mask.sum())[y_hat]

    expanded_y_hat = np.zeros((y_hat.shape[0], num_classes))
    expanded_y_hat[:, left_out_mask == 1] = y_hat

    for true_label, pred_label in zip(y, np.argmax(expanded_y_hat, axis=1)):
        confusion_matrix[true_label, pred_label] += 1

    return confusion_matrix

def eval_rf_confusion_binary(model: RandomForestClassifier, X: np.ndarray, y: np.ndarray, num_classes: int):

    confusion_matrix = np.zeros((num_classes, 2))

    y_hat = model.predict(X).reshape(-1)
    logger.debug(f"y_hat max label: {np.max(y_hat)}")

    for true_label, pred_label in zip(y, y_hat):
        confusion_matrix[true_label, pred_label] += 1

    return confusion_matrix

def eval_mlp_transformer_confusion_binary(model_path, model_config, dataloader, num_classes, device):
    ckpt = torch.load(model_path, weights_only=True)
    encoder_config = model_config["encoder"]
    encoder = TransformerEncoder(**encoder_config)

    head_config = model_config["head"]
    head = MLP(**head_config)
    model = nn.Sequential(encoder, nn.Flatten(), head)

    model.load_state_dict(ckpt["model_state_dict"])
    del ckpt
    model.to(device)
    model.eval()

    confusion_matrix = torch.zeros(num_classes, 2, device=device)
    with torch.no_grad():
        for inp, label in dataloader:

            label = label.to(device)
            inp = inp.to(device)
            
            out = model(inp)
            pred = torch.argmax(out, dim=-1)

            # Use scatter_add to efficiently accumulate confusion matrix counts
            index = label * 2 + pred
            index = index.long()
            ones = torch.ones_like(index, dtype=torch.float).to(device)
            confusion_matrix.view(-1).scatter_add_(0, index, ones)

    return confusion_matrix