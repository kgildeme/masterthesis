from pathlib import Path
import json
import logging
import os
import tempfile

from matplotlib import pyplot as plt
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from ray.train.torch import get_devices
from ray import train
from ray.train import Checkpoint

from ..models.transformer import AutoregressiveTransformerEncoder, MaskedTransformerEncoder, TransformerEncoder
from ..models.nn import Autoencoder, MLP, CNN_1D, CNNTranspose_1D
from ..data import OpTCDataset, OpTCDatasetWithIndexing, worker_init, SCVICCIDSDatasetBinary, SCVICCIDSDataset, worker_init_wrapped, Undersampling, OpTCDatasetUndersampling
from ..util import metrics, misc_funcs as misc
from .util import train_step, train_step_ssl

logger = logging.getLogger(__name__)

def train_mlp_autoencoder(model_config: dict, ds_name: str, ds_args:dict, eval_ds_args:dict = None, batch_size=32, epochs=20, lr=1e-3, patience: int = None, eval_metric: callable = None, checkpoint_path: str | Path = None,
                                  tensorboard_writer: SummaryWriter = None, report_every: int = None, n_worker:int = 0, device: str = "cpu", restart_ckpt_path: str | Path = None):

    if checkpoint_path:
        os.makedirs(os.path.split(checkpoint_path)[0],exist_ok=True)

    logger.info("Start loading dataset and model")

    if "optc" in ds_name.lower():
        if "last_part" in ds_args.keys():
            train_dataset = OpTCDatasetWithIndexing(**ds_args)
            eval_dataset = OpTCDatasetWithIndexing(**eval_ds_args)
        else:
            train_dataset = OpTCDataset(**ds_args)
            eval_dataset = OpTCDataset(**eval_ds_args)
        train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=n_worker, worker_init_fn=worker_init)
        eval_dl = torch.utils.data.DataLoader(eval_dataset, batch_size=batch_size, num_workers=n_worker, worker_init_fn=worker_init)
    else:
        idxs = np.loadtxt(os.path.join(misc.data(), "scvic/train_indices.txt"), dtype=int)
        dataset = SCVICCIDSDataset(misc.data_raw(scvic=True), subset=idxs, network_data=ds_args["network_data"], host_data=ds_args["host_data"],
                                         host_embeddings=ds_args["host_embeddings"])
        train_idxs, val_idxs = next(StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=ds_args["seed"]).split(np.zeros(len(dataset)), [d[-1] for d in dataset]))
        malicious_idxs = []
        benign_idxs = []
        for train_idx in train_idxs:
            if dataset[train_idx][-1] != 0:
                malicious_idxs.append(train_idx)
            else:
                benign_idxs.append(train_idx)

        train_idxs = np.asarray(benign_idxs)
        val_idxs = np.append(val_idxs, malicious_idxs)

        logger.info(f"Lenght of trainset is {len(train_idxs)}")

        train_dataset = SCVICCIDSDatasetBinary(data=dataset.data, subset=train_idxs, network_data=ds_args["network_data"], host_data=ds_args["host_data"],
                                         host_embeddings=ds_args["host_embeddings"])
        val_dataset = SCVICCIDSDatasetBinary(data=dataset.data, subset=val_idxs, network_data=ds_args["network_data"], host_data=ds_args["host_data"],
                                         host_embeddings=ds_args["host_embeddings"])
        train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=n_worker, shuffle=True)
        eval_dl = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, num_workers=n_worker, shuffle=False)

    encoder = MLP(**model_config)
    model_config["input_dim"], model_config["output_dim"] = model_config["output_dim"], model_config["input_dim"]
    decoder = MLP(**model_config)
    decoder = nn.Sequential(decoder, nn.Sigmoid())

    model = Autoencoder(encoder=encoder, decoder=decoder)
    model.to(device=device)

    cooloff = 0
    best_loss = float('inf')
    
    loss_fnc = nn.MSELoss()

    if restart_ckpt_path:
        ckpt = torch.load(restart_ckpt_path)
        model.load_state_dict(ckpt["model_state_dict"])
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params=params, lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        best_loss = ckpt["loss"]
        restart_epoch = ckpt["epoch"] + 1
        del ckpt
    else:
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params=params, lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        restart_epoch = 0
    
    restart_string = f"Restart training from epoch {restart_epoch}" if restart_ckpt_path else "Start training from scratch"
    ckp_string = "with" if checkpoint_path else "without"
    es_string = f"with early_stopping after {patience} epochs without improvement" if patience else "without early stopping"
    logger.info(f"{restart_string} for {epochs} epochs, {ckp_string} checkpointing and {es_string}")

    progress = []
    total_steps = 0
    try:
        for epoch in range(restart_epoch, epochs):
            logger.info(f"Start epoch {epoch}")
            total_loss = 0
            steps = 0

            for data in train_dl:
                
                if "optc" in ds_name.lower():
                    inp = torch.flatten(data, start_dim=1)
                else:
                    inp = torch.concat([torch.flatten(data[i], start_dim=1) for i in range(len(data) - 1)], dim=1)
                inp = inp.to(device)

                loss = train_step((inp,), inp, model=model, loss_criterion=loss_fnc, optimizer=optimizer)

                total_loss += loss.item()
                steps += 1
                total_steps += 1

                if report_every is not None and steps % report_every == 0:
                    with torch.no_grad():
                        model.eval()
                        eval_loss = 0
                        eval_steps = 0
                        for eval_inp in eval_dl:
                            eval_inp = torch.flatten(eval_inp, start_dim=1)
                            eval_inp = eval_inp.to(device)
                            eval_out = model(eval_inp)

                            eval_loss += F.mse_loss(eval_out, eval_inp).item()

                            eval_steps += 1
                        
                        eval_loss /= eval_steps
                    model.train()
                    if tensorboard_writer:
                        tensorboard_writer.add_scalar('train/loss', loss.item(), total_steps)
                        tensorboard_writer.add_scalar('val/loss', eval_loss, total_steps)

            epoch_loss = total_loss / steps
            cooloff += 1
            with torch.no_grad():
                model.eval()
                eval_loss = 0
                eval_steps = 0
                scores = []
                labels = []
                for eval_data in eval_dl:
                    if "optc" in ds_name.lower():
                        eval_inp = torch.flatten(eval_data[0], start_dim=1)
                        eval_inp = eval_inp.to(device)

                    else:
                        eval_inp = torch.concat([torch.flatten(eval_data[i], start_dim=1) for i in range(len(eval_data) - 1)], dim=1)
                        eval_inp = eval_inp.to(device)

                    eval_out = model(eval_inp)
                    eval_labels = eval_data[-1].to(device)  
                    loss_expanded = F.mse_loss(eval_inp, eval_out, reduction="none")                  
                    eval_loss += (loss_expanded.mean(dim=1) * (1 - eval_labels)).mean().item()
                    
                    scores.extend(torch.mean(loss_expanded, dim=1).detach().cpu().numpy())
                    labels.extend(eval_labels.detach().cpu().numpy())

                    eval_steps += 1
                
                eval_loss /= eval_steps
                if eval_metric:
                    eval_score = eval_metric(labels, scores)
            model.train()

            progress.append(eval_loss)

            if eval_loss < best_loss:
                best_loss = eval_loss
                cooloff = 0
                if checkpoint_path:
                    torch.save(
                        {"epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "loss": eval_loss}, f=checkpoint_path
                    )
                    logger.info("Saved Checkpoint")
            logger.info(f"Finished Epoch {epoch} with loss={epoch_loss} and best_loss={best_loss}")
            if tensorboard_writer:
                tensorboard_writer.add_scalar('train/loss', epoch_loss, total_steps)
                tensorboard_writer.add_scalar('val/best_loss', best_loss, total_steps)
                tensorboard_writer.add_scalar('val/loss', eval_loss, total_steps)
                tensorboard_writer.add_scalar('epoch', epoch, total_steps)
                tensorboard_writer.add_scalar('lr', scheduler.get_last_lr()[0], total_steps)

                if eval_metric:
                    tensorboard_writer.add_scalar('val/score', eval_score, total_steps)

            scheduler.step()
            if patience:
                if cooloff > patience:
                    break
    except KeyboardInterrupt as e:
        logger.info("Received KeyboardInterrrupt. Save last model if checkpointing activated")
        if checkpoint_path:
            torch.save(
                        {"epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "loss": best_loss}, f=os.path.join(os.path.split(checkpoint_path)[0], "last.pt")
                    )
        raise e
    
    logger.info("Finished Training")
    if checkpoint_path:
            torch.save(
                        {"epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "loss": best_loss}, f=os.path.join(os.path.split(checkpoint_path)[0], "last.pt")
                    )
    return progress

def train_mlp_autoencoder_ssl(
        model_config:dict, model_path:str, ds_name: str, ds_args:dict,
        eval_ds_args:dict = None, batch_size=32, epochs=20, lr=1e-3,
        checkpoint_path: str | Path = None,
        tensorboard_writer: SummaryWriter = None, device: str = "cpu",
        loss_type="inverse", gamma: float = 1., store_every: int = 5):

    if checkpoint_path:
        os.makedirs(os.path.split(checkpoint_path)[0],exist_ok=True)

    logger.info("Start loading dataset and model")

    train_dataset = OpTCDatasetUndersampling(**ds_args)
    
    eval_dataset = OpTCDataset(**eval_ds_args)

    train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=1)
    eval_dl = torch.utils.data.DataLoader(eval_dataset, batch_size=batch_size, num_workers=1)

    if loss_type == "inverse":
        loss_fnc = metrics.inverse_mse
    elif loss_type == "reciprocal":
        loss_fnc = metrics.reciprocal_mse
    else:
        raise ValueError("Unknown loss type. Must be either inverse or reciprocal")

    encoder = MLP(**model_config)
    model_config["input_dim"], model_config["output_dim"] = model_config["output_dim"], model_config["input_dim"]
    decoder = MLP(**model_config)
    decoder = nn.Sequential(decoder, nn.Sigmoid())

    model = Autoencoder(encoder=encoder, decoder=decoder)

    ckpt = torch.load(model_path, weights_only=True, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params=params, lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    logger.info(f"supervised finetuning for {epochs} epochs, using checkpointing")
    del ckpt
    model.to(device=device)

    progress = []
    total_steps = 0
    best_mcc = float("-inf")

    with torch.no_grad():
        model.eval()
        eval_loss = 0
        eval_steps = 0
        scores_base = []
        labels_base = []
        for eval_data in eval_dl:
            eval_inp = torch.flatten(eval_data[0], start_dim=1)
            eval_inp = eval_inp.to(device)

            eval_out = model(eval_inp)
            eval_labels = eval_data[-1].to(device)  
            loss = loss_fnc(eval_out, eval_inp, eval_labels, gamma=gamma)
            eval_loss += loss.item()

            loss_expanded = F.mse_loss(eval_inp, eval_out, reduction="none")                    
            scores_base.extend(torch.mean(loss_expanded, dim=1).detach().cpu().numpy())
            labels_base.extend(eval_labels.detach().cpu().numpy())

            eval_steps += 1
        
        eval_loss /= eval_steps
        eval_auc = metrics.auc_score(labels_base, scores_base)

        scores_base = np.asarray(scores_base)
        labels_base = np.asarray(labels_base)
    model.train()

    progress.append(eval_loss)
    logger.info(f"Baseline has eval_loss={eval_loss} and auc={eval_auc}")
    if tensorboard_writer:
        tensorboard_writer.add_scalar('val/loss', eval_loss, 0)
        tensorboard_writer.add_scalar('val/auc', eval_auc, 0)
        fig, ax = plt.subplots(1, 1, figsize=(10, 4))

        ax.hist(scores_base[labels_base == 0], bins=200, histtype='step', log=True)
        ax.hist(scores_base[labels_base == 1], bins=200, histtype='step', log=True)

        # Add the threshold as a vertical red dashed line
        ax.legend(['eval_benign', 'eval_malicious'])
        ax.set_xlim(0, max(scores_base))
        tensorboard_writer.add_figure("score_distribution", fig, 0)
        plt.close(fig)


    try:
        for epoch in range(epochs):
            model.train()
            logger.info(f"Start epoch {epoch}")
            total_loss = 0
            steps = 0
            scores_train = []
            labels_train = []

            for data in train_dl:
                lbl = data[-1].to(device)
                inp = torch.flatten(data[0], start_dim=1)
                inp = inp.to(device)

                loss = train_step_ssl((inp,), inp, lbl, model=model, loss_criterion=loss_fnc, optimizer=optimizer, gamma=gamma)

                total_loss += loss.item()
                steps += 1
                total_steps += 1
                with torch.no_grad():
                    out = model(inp)
                    scores_train.extend(torch.mean(F.mse_loss(inp, out, reduction="none"), dim=1).detach().cpu().numpy())
                    labels_train.extend(lbl.detach().cpu().numpy())
            distance = torch.tensor(scores_train)[torch.tensor(labels_train) == 0]
            q1 = torch.quantile(distance, 0.25)
            q3 = torch.quantile(distance, 0.75)
            iqr = q3 - q1
            threshold = (q3 + 1.5 * iqr).item()

            epoch_loss = total_loss / steps
            with torch.no_grad():
                model.eval()
                eval_loss = 0
                eval_steps = 0
                scores_eval = []
                labels_eval = []
                for eval_data in eval_dl:
                    eval_inp = torch.flatten(eval_data[0], start_dim=1)
                    eval_inp = eval_inp.to(device)

                    eval_out = model(eval_inp)
                    eval_labels = eval_data[-1].to(device)  
                    loss = loss_fnc(eval_out, eval_inp, eval_labels, gamma=gamma)
                    eval_loss += loss.item()

                    loss_expanded = F.mse_loss(eval_inp, eval_out, reduction="none")                    
                    scores_eval.extend(torch.mean(loss_expanded, dim=1).detach().cpu().numpy())
                    labels_eval.extend(eval_labels.detach().cpu().numpy())

                    eval_steps += 1
                
                eval_loss /= eval_steps
                distance = torch.tensor(scores_eval)[torch.tensor(labels_eval) == 0]
                distance_mal = torch.tensor(scores_eval)[torch.tensor(labels_eval) == 1]

                scores_eval = np.asarray(scores_eval)
                labels_eval = np.asarray(labels_eval)
                scores_train = np.asarray(scores_train)
                labels_train = np.asarray(labels_train)

                eval_f1 = metrics.f1_score((distance_mal > threshold).int().sum(), (distance > threshold).int().sum(), (distance_mal < threshold).int().sum())
                eval_auc = metrics.auc_score(labels_eval, scores_eval)
                eval_mcc = metrics.mcc(labels_eval, (scores_eval > threshold).astype(int))

            model.train()

            progress.append(eval_loss)
            if checkpoint_path and eval_mcc > best_mcc:
                best_mcc = eval_mcc
                torch.save(
                    {"epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "loss": eval_loss}, f=checkpoint_path
                )
                logger.info("Saved Checkpoint")

            if checkpoint_path and (epoch+1) % store_every == 0:
                torch.save(
                            {"epoch": epoch,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "scheduler_state_dict": scheduler.state_dict(),
                            "loss": eval_mcc}, f=os.path.join(os.path.split(checkpoint_path)[0], f"{epoch:3d}.pt")
                        )
 
            logger.info(f"Finished Epoch {epoch} with loss={epoch_loss} and mcc={eval_mcc}")
            if tensorboard_writer:
                tensorboard_writer.add_scalar('train/loss', epoch_loss, total_steps)
                tensorboard_writer.add_scalar('val/loss', eval_loss, total_steps)
                tensorboard_writer.add_scalar('epoch', epoch, total_steps)
                tensorboard_writer.add_scalar('lr', scheduler.get_last_lr()[0], total_steps)
                tensorboard_writer.add_scalar('val/f1', eval_f1, total_steps)
                tensorboard_writer.add_scalar('val/auc', eval_auc, total_steps)
                tensorboard_writer.add_scalar('val/mcc', eval_mcc, total_steps)
                tensorboard_writer.add_scalar('val/threshold', threshold, total_steps)

                fig, ax = plt.subplots(1, 1, figsize=(10, 4))
                ax.hist(scores_train[labels_train == 0], bins=200, histtype='step', log=True)
                ax.hist(scores_train[labels_train == 1], bins=200, histtype='step', log=True)

                ax.hist(scores_eval[labels_eval == 0], bins=200, histtype='step', log=True)
                ax.hist(scores_eval[labels_eval == 1], bins=200, histtype='step', log=True)


                # Add the threshold as a vertical red dashed line
                ax.axvline(x=threshold, color='red', linestyle='--')
                ax.legend(['train_benign', 'train_malicious', 'eval_benign', 'eval_malicious'])
                ax.set_xlim(0, np.max(np.concat([scores_eval, scores_train])))

                tensorboard_writer.add_figure("score_distribution", fig, total_steps)
                plt.close(fig)
                
            scheduler.step()
    except KeyboardInterrupt as e:
        logger.info("Received KeyboardInterrrupt. Save last model if checkpointing activated")

        raise e

    logger.info("Finished Training")
    if checkpoint_path:
            torch.save(
                        {"epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "loss": best_mcc}, f=os.path.join(os.path.split(checkpoint_path)[0], "last.pt")
                    )

def train_deep_sad(model_config:dict, model_path:str, ds_name: str, ds_args:dict, pretrain_ds_args: dict,
        eval_ds_args:dict = None, batch_size=32, epochs=20, lr=1e-3, checkpoint_path: str | Path = None,
        tensorboard_writer: SummaryWriter = None, device: str = "cpu", gamma: float = 1., store_every: int = 5):
    
    if checkpoint_path:
        os.makedirs(os.path.split(checkpoint_path)[0],exist_ok=True)

    logger.info("Start loading dataset and model")
    pretrain_dataset = OpTCDataset(**pretrain_ds_args)
    pretrain_dl = torch.utils.data.DataLoader(pretrain_dataset, batch_size=batch_size, num_workers=1, worker_init_fn=worker_init)

    model = MLP(**model_config, bias=False)
    ckpt = torch.load(model_path, weights_only=True, map_location='cpu')

    state_dict_without_bias = {}
    for k, v in ckpt["model_state_dict"].items():
        if "encoder" in k and "bias" not in k:
            
            state_dict_without_bias[".".join(k.split(".")[1:])] = v

    model.load_state_dict(state_dict_without_bias)
    print(model)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params=params, lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    logger.info(f"deepSAD for {epochs} epochs, using checkpointing")
    del ckpt
    model.to(device=device)

    # init c
    logger.info("Initialize c")
    with torch.no_grad():
        model.eval()
        c = torch.zeros(model_config["output_dim"], requires_grad=False).to(device)
        n = 0
        for data in pretrain_dl:
            inp = torch.flatten(data, start_dim=1)
            inp = inp.to(device)

            out = model(inp)
            c = out.mean(dim=0) / (n + 1) + n / (n + 1) * c
            n += 1

    c = c[None, :]
    logger.info(f"c has shape {c.shape}")
    del pretrain_dl
    del pretrain_dataset

    loss_fnc = metrics.reciprocal_mse

    train_dataset = OpTCDatasetUndersampling(**ds_args)
    
    eval_dataset = OpTCDataset(**eval_ds_args)

    train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=1)
    eval_dl = torch.utils.data.DataLoader(eval_dataset, batch_size=batch_size, num_workers=1)

    with torch.no_grad():
        model.eval()
        scores_train = []
        labels_train = []
        for data in train_dl:
            lbl = data[-1].to(device)
            inp = torch.flatten(data[0], start_dim=1)
            inp = inp.to(device)

            out = model(inp)
            scores_train.extend(torch.mean(F.mse_loss(out, c.repeat(out.shape[0], 1), reduction="none"), dim=1).detach().cpu().numpy())
            labels_train.extend(lbl.detach().cpu().numpy())
        distance = torch.tensor(scores_train)[torch.tensor(labels_train) == 0]
        q1 = torch.quantile(distance, 0.25)
        q3 = torch.quantile(distance, 0.75)
        iqr = q3 - q1
        threshold = (q3 + 1.5 * iqr).item()
        eval_loss = 0
        eval_steps = 0
        scores_eval = []
        labels_eval = []
        for eval_data in eval_dl:
            eval_inp = torch.flatten(eval_data[0], start_dim=1)
            eval_inp = eval_inp.to(device)

            eval_out = model(eval_inp)
            eval_labels = eval_data[-1].to(device)  
            loss = loss_fnc(eval_out, c.repeat(eval_inp.shape[0], 1), eval_labels, gamma=gamma)
            eval_loss += loss.item()

            loss_expanded = F.mse_loss(eval_out, c.repeat(eval_inp.shape[0], 1), reduction="none")                    
            scores_eval.extend(torch.mean(loss_expanded, dim=1).detach().cpu().numpy())
            labels_eval.extend(eval_labels.detach().cpu().numpy())

            eval_steps += 1
        
        distance = torch.tensor(scores_eval)[torch.tensor(labels_eval) == 0]
        distance_mal = torch.tensor(scores_eval)[torch.tensor(labels_eval) == 1]
        scores_eval = np.asarray(scores_eval)
        labels_eval = np.asarray(labels_eval)

        scores_train = np.asarray(scores_train)
        labels_train = np.asarray(labels_train)
        eval_f1 = metrics.f1_score((distance_mal > threshold).int().sum(), (distance > threshold).int().sum(), (distance_mal < threshold).int().sum())
        eval_auc = metrics.auc_score(labels_eval, scores_eval)
        eval_mcc = metrics.mcc(labels_eval, (scores_eval > threshold).astype(int))
        eval_loss /= eval_steps
    model.train()

    logger.info(f"Baseline has eval_loss={eval_loss} and mcc={eval_mcc}")
    if tensorboard_writer:
        tensorboard_writer.add_scalar('train/threshold', threshold, 0)
        tensorboard_writer.add_scalar('val/loss', eval_loss, 0)
        tensorboard_writer.add_scalar('val/f1', eval_f1, 0)
        tensorboard_writer.add_scalar('val/auc', eval_auc, 0)
        tensorboard_writer.add_scalar('val/mcc', eval_mcc, 0)

        fig, ax = plt.subplots(1, 1, figsize=(10, 4))
        ax.hist(scores_train[labels_train == 0], bins=200, histtype='step', log=True)
        ax.hist(scores_train[labels_train == 1], bins=200, histtype='step', log=True)

        ax.hist(scores_eval[labels_eval == 0], bins=200, histtype='step', log=True)
        ax.hist(scores_eval[labels_eval == 1], bins=200, histtype='step', log=True)


        # Add the threshold as a vertical red dashed line
        ax.axvline(x=threshold, color='red', linestyle='--')
        ax.legend(['train_benign', 'train_malicious', 'eval_benign', 'eval_malicious'])
        ax.set_xlim(0, np.max(np.concat([scores_eval, scores_train])))
        tensorboard_writer.add_figure("score_distribution", fig, 0)
        plt.close(fig)

    total_steps = 0
    best_mcc = float("-inf")
    try:
        for epoch in range(epochs):
            model.train()
            logger.info(f"Start epoch {epoch}")
            total_loss = 0
            steps = 0
            scores_train = []
            labels_train = []

            for data in train_dl:
                lbl = data[-1].to(device)
                inp = torch.flatten(data[0], start_dim=1)
                inp = inp.to(device)

                loss = train_step_ssl((inp,), c.repeat(inp.shape[0], 1), lbl, model=model, loss_criterion=loss_fnc, optimizer=optimizer, gamma=gamma)

                total_loss += loss.item()
                steps += 1
                total_steps += 1
                with torch.no_grad():
                    out = model(inp)
                    scores_train.extend(torch.mean(F.mse_loss(out, c.repeat(inp.shape[0], 1), reduction="none"), dim=1).detach().cpu().numpy())
                    labels_train.extend(lbl.detach().cpu().numpy())
            distance = torch.tensor(scores_train)[torch.tensor(labels_train) == 0]
            q1 = torch.quantile(distance, 0.25)
            q3 = torch.quantile(distance, 0.75)
            iqr = q3 - q1
            threshold = (q3 + 1.5 * iqr).item()

            epoch_loss = total_loss / steps
            with torch.no_grad():
                model.eval()
                eval_loss = 0
                eval_steps = 0
                scores_eval = []
                labels_eval = []
                for eval_data in eval_dl:
                    eval_inp = torch.flatten(eval_data[0], start_dim=1)
                    eval_inp = eval_inp.to(device)

                    eval_out = model(eval_inp)
                    eval_labels = eval_data[-1].to(device)  
                    loss = loss_fnc(eval_out, c.repeat(eval_inp.shape[0], 1), eval_labels, gamma=gamma)
                    eval_loss += loss.item()

                    loss_expanded = F.mse_loss(eval_out, c.repeat(eval_inp.shape[0], 1), reduction="none")                    
                    scores_eval.extend(torch.mean(loss_expanded, dim=1).detach().cpu().numpy())
                    labels_eval.extend(eval_labels.detach().cpu().numpy())

                    eval_steps += 1
                
                eval_loss /= eval_steps
                distance = torch.tensor(scores_eval)[torch.tensor(labels_eval) == 0]
                distance_mal = torch.tensor(scores_eval)[torch.tensor(labels_eval) == 1]

                scores_eval = np.asarray(scores_eval)
                labels_eval = np.asarray(labels_eval)
                scores_train = np.asarray(scores_train)
                labels_train = np.asarray(labels_train)
                eval_f1 = metrics.f1_score((distance_mal > threshold).int().sum(), (distance > threshold).int().sum(), (distance_mal < threshold).int().sum())
                eval_auc = metrics.auc_score(labels_eval, scores_eval)
                eval_mcc = metrics.mcc(labels_eval, (scores_eval > threshold).astype(int))

            if checkpoint_path and eval_mcc > best_mcc:
                best_mcc = eval_mcc
                torch.save(
                    {"epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "loss": eval_loss, "c": c}, f=checkpoint_path
                )
                logger.info("Saved Checkpoint")
            if checkpoint_path and (epoch+1) % store_every == 0:
                torch.save(
                            {"epoch": epoch,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "scheduler_state_dict": scheduler.state_dict(),
                            "loss": eval_mcc, "c": c}, f=os.path.join(os.path.split(checkpoint_path)[0], f"{epoch:3d}.pt")
                        )
            logger.info(f"Finished Epoch {epoch} with loss={epoch_loss} and mcc={eval_mcc}")
            if tensorboard_writer:
                tensorboard_writer.add_scalar('train/loss', epoch_loss, total_steps)
                tensorboard_writer.add_scalar('val/loss', eval_loss, total_steps)
                tensorboard_writer.add_scalar('epoch', epoch, total_steps)
                tensorboard_writer.add_scalar('lr', scheduler.get_last_lr()[0], total_steps)
                tensorboard_writer.add_scalar('val/f1', eval_f1, total_steps)
                tensorboard_writer.add_scalar('val/auc', eval_auc, total_steps)
                tensorboard_writer.add_scalar('val/mcc', eval_mcc, total_steps)
                tensorboard_writer.add_scalar('train/threshold', threshold, total_steps)

                fig, ax = plt.subplots(1, 1, figsize=(10, 4))
                ax.hist(scores_train[labels_train == 0], bins=200, histtype='step', log=True)
                ax.hist(scores_train[labels_train == 1], bins=200, histtype='step', log=True)

                ax.hist(scores_eval[labels_eval == 0], bins=200, histtype='step', log=True)
                ax.hist(scores_eval[labels_eval == 1], bins=200, histtype='step', log=True)


                # Add the threshold as a vertical red dashed line
                ax.axvline(x=threshold, color='red', linestyle='--')
                ax.legend(['train_benign', 'train_malicious', 'eval_benign', 'eval_malicious'])
                
                ax.set_xlim(0, np.max(np.concat([scores_eval, scores_train])))
                
                tensorboard_writer.add_figure("score_distribution", fig, total_steps)
                plt.close(fig)

            scheduler.step()
    except KeyboardInterrupt as e:
        logger.info("Received KeyboardInterrrupt. Save last model if checkpointing activated")

        raise e

    if checkpoint_path:
            torch.save(
                        {"epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "loss": best_mcc, "c": c}, f=os.path.join(os.path.split(checkpoint_path)[0], "last.pt")
                    )
    logger.info(f"Training Done")

def train_mlp_autoencoder_ray(config, ds_name:str, ds_args: dict, eval_ds_args: dict = None, loss_fnc=F.mse_loss, eval_metric: callable = None, early_stopping: int = None, n_worker: int = 1):

    logger.info("Start loading dataset and model")

    if "optcfull" in ds_name.lower():
        train_dataset = OpTCDataset(**ds_args)
        eval_dataset = OpTCDataset(**eval_ds_args)
        train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=config["batch_size"], num_workers=n_worker, worker_init_fn=worker_init)
        eval_dl = torch.utils.data.DataLoader(eval_dataset, batch_size=config["batch_size"], num_workers=n_worker, worker_init_fn=worker_init)
    elif "optc" in ds_name.lower():
        train_dataset = OpTCDatasetWithIndexing(**ds_args)
        eval_dataset = OpTCDatasetWithIndexing(**eval_ds_args)
        train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=config["batch_size"], num_workers=n_worker, worker_init_fn=worker_init)
        eval_dl = torch.utils.data.DataLoader(eval_dataset, batch_size=config["batch_size"], num_workers=n_worker, worker_init_fn=worker_init)
    else:
        idxs = np.loadtxt(os.path.join(misc.data(), "scvic/train_indices.txt"), dtype=int)
        dataset = SCVICCIDSDataset(misc.data_raw(scvic=True), subset=idxs, network_data=ds_args["network_data"], host_data=ds_args["host_data"],
                                         host_embeddings=ds_args["host_embeddings"])
        train_idxs, val_idxs = next(StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=ds_args["seed"]).split(np.zeros(len(dataset)), [d[-1] for d in dataset]))
        malicious_idxs = []
        benign_idxs = []
        for train_idx in train_idxs:
            if dataset[train_idx][-1] != 0:
                malicious_idxs.append(train_idx)
            else:
                benign_idxs.append(train_idx)

        train_idxs = np.asarray(benign_idxs)
        val_idxs = np.append(val_idxs, malicious_idxs)

        logger.info(f"Lenght of trainset is {len(train_idxs)}")

        train_dataset = SCVICCIDSDatasetBinary(data=dataset.data, subset=train_idxs, network_data=ds_args["network_data"], host_data=ds_args["host_data"],
                                         host_embeddings=ds_args["host_embeddings"])
        val_dataset = SCVICCIDSDatasetBinary(data=dataset.data, subset=val_idxs, network_data=ds_args["network_data"], host_data=ds_args["host_data"],
                                         host_embeddings=ds_args["host_embeddings"])
        train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=config["batch_size"], num_workers=n_worker, shuffle=True)
        eval_dl = torch.utils.data.DataLoader(val_dataset, batch_size=config["batch_size"], num_workers=n_worker, shuffle=False)
    
    model_config = config["model"].copy()
    encoder = MLP(**model_config)
    model_config["input_dim"], model_config["output_dim"] = model_config["output_dim"], model_config["input_dim"]
    decoder = MLP(**model_config)
    decoder = nn.Sequential(decoder, nn.Sigmoid())

    model = Autoencoder(encoder=encoder, decoder=decoder)
    
    device = get_devices()[0]
    model.to(device=device)

    cooloff = 0
    best_loss = float('inf')
    best_score = 0.
    loss_fnc = nn.MSELoss()

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params=params, lr=config["lr"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])

    logger.info(f" Start training for {config["epochs"]} epochs")

    progress = []
    total_steps = 0
    for epoch in range(config["epochs"]):
        logger.info(f"Start epoch {epoch}")
        total_loss = 0
        steps = 0

        for data in train_dl:
            if "optc" in ds_name.lower():
                inp = torch.flatten(data, start_dim=1)
            else:
                inp = torch.concat([torch.flatten(data[i], start_dim=1) for i in range(len(data) - 1)], dim=1)
            inp = inp.to(device)

            loss = train_step((inp,), inp, model=model, loss_criterion=loss_fnc, optimizer=optimizer)

            total_loss += loss.item()
            steps += 1
            total_steps += 1

        epoch_loss = total_loss / steps
        cooloff += 1
        with torch.no_grad():
            model.eval()
            eval_loss = 0
            eval_steps = 0
            scores = []
            labels = []
            for eval_data in eval_dl:
                if "optc" in ds_name.lower():
                    eval_inp = torch.flatten(eval_data[0], start_dim=1)
                    eval_inp = eval_inp.to(device)
                else:
                    eval_inp = torch.concat([torch.flatten(eval_data[i], start_dim=1) for i in range(len(eval_data) - 1)], dim=1)
                    eval_inp = eval_inp.to(device)

                eval_out = model(eval_inp)
                eval_labels = eval_data[-1].to(device)  
                loss_expanded = F.mse_loss(eval_inp, eval_out, reduction="none")                  
                eval_loss += (loss_expanded.mean(dim=1) * (1 - eval_labels)).mean().item()
                
                scores.extend(torch.mean(loss_expanded, dim=1).detach().cpu().numpy())
                labels.extend(eval_labels.detach().cpu().numpy())

                eval_steps += 1
            
            eval_loss /= eval_steps
            if eval_metric:
                eval_score = eval_metric(labels, scores)
        model.train()

        if eval_score > best_score:
            best_loss = eval_loss
            best_score = eval_score
            cooloff = 0

        train.report(
            {
                "epoch": epoch,
                "steps": total_steps,
                "train/loss": epoch_loss,
                "val/loss": eval_loss,
                "val/best_loss": best_loss,
                "val/score": eval_score,
                "lr": scheduler.get_last_lr()[0]
            })
        logger.info(f"Finished Epoch {epoch} with loss={epoch_loss} and best_loss={best_loss}")

        scheduler.step()
        if early_stopping:
            if cooloff > early_stopping:
                break
            

def train_cnn_autoencoder(model_config: dict, train_dataset_config:dict, eval_dataset_config:dict, batch_size=32, epochs=20, lr=1e-3, early_stopping: int = None, checkpoint_path: str | Path = None,
                                  tensorboard_writer: SummaryWriter = None, report_every: int = None, n_worker:int = 0, device: str = "cpu", restart_ckpt_path: str | Path = None):

    if checkpoint_path:
        os.makedirs(os.path.split(checkpoint_path)[0],exist_ok=True)

    logger.info("Start loading dataset and model")

    train_dataset = OpTCDatasetWithIndexing(**train_dataset_config)
    eval_dataset = OpTCDatasetWithIndexing(**eval_dataset_config)
    train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=n_worker, worker_init_fn=worker_init)
    eval_dl = torch.utils.data.DataLoader(eval_dataset, batch_size=batch_size, num_workers=n_worker, worker_init_fn=worker_init)

    encoder = CNN_1D(**model_config)
    model_config["feature_dims"] = model_config["feature_dims"][::-1]
    model_config["residual_dims"] = model_config["residual_dims"][::-1]
    decoder = CNNTranspose_1D(**model_config)
    decoder = nn.Sequential(decoder, nn.Sigmoid())

    model = Autoencoder(encoder=encoder, decoder=decoder)
    model.to(device=device)

    cooloff = 0
    best_loss = float('inf')
    loss_fnc = nn.MSELoss()

    if restart_ckpt_path:
        ckpt = torch.load(restart_ckpt_path)
        model.load_state_dict(ckpt["model_state_dict"])
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params=params, lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        best_loss = ckpt["loss"]
        restart_epoch = ckpt["epoch"] + 1
        del ckpt
    else:
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params=params, lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        restart_epoch = 0
    
    restart_string = f"Restart training from epoch {restart_epoch}" if restart_ckpt_path else "Start training from scratch"
    ckp_string = "with" if checkpoint_path else "without"
    es_string = f"with early_stopping after {early_stopping} epochs without improvement" if early_stopping else "without early stopping"
    logger.info(f"{restart_string} for {epochs} epochs, {ckp_string} checkpointing and {es_string}")

    progress = []
    total_steps = 0
    try:
        for epoch in range(restart_epoch, epochs):
            logger.info(f"Start epoch {epoch}")
            total_loss = 0
            steps = 0

            for inp in train_dl:
                
                inp = torch.permute(inp, (0, 2, 1))
                inp = inp.to(device)

                loss = train_step((inp,), inp, model=model, loss_criterion=loss_fnc, optimizer=optimizer)

                total_loss += loss.item()
                steps += 1
                total_steps += 1

                if steps % report_every == 0:
                    with torch.no_grad():
                        model.eval()
                        eval_loss = 0
                        eval_steps = 0
                        for eval_inp in eval_dl:
                            eval_inp = torch.permute(eval_inp, (0, 2, 1))
                            eval_inp = eval_inp.to(device)
                            eval_out = model(eval_inp)

                            eval_loss += F.mse_loss(eval_out, eval_inp).item()

                            eval_steps += 1
                        
                        eval_loss /= eval_steps
                    model.train()
                    if tensorboard_writer:
                        tensorboard_writer.add_scalar('train/loss', loss.item(), total_steps)
                        tensorboard_writer.add_scalar('val/loss', eval_loss, total_steps)

            epoch_loss = total_loss / steps
            cooloff += 1
            with torch.no_grad():
                model.eval()
                eval_loss = 0
                eval_steps = 0
                for eval_inp in eval_dl:
                    eval_inp = torch.permute(eval_inp, (0, 2, 1))
                    eval_inp = eval_inp.to(device)
                    eval_out = model(eval_inp)

                    eval_loss += F.mse_loss(eval_out, eval_inp).item()

                    
                    eval_steps += 1
                
                eval_loss /= eval_steps
            model.train()

            progress.append(eval_loss)

            if eval_loss < best_loss:
                best_loss = eval_loss
                cooloff = 0
                if checkpoint_path:
                    torch.save(
                        {"epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "loss": eval_loss}, f=checkpoint_path
                    )
                    logger.info("Saved Checkpoint")
            logger.info(f"Finished Epoch {epoch} with loss={epoch_loss} and best_loss={best_loss}")
            if tensorboard_writer:
                tensorboard_writer.add_scalar('train/epoch_loss', epoch_loss, total_steps)
                tensorboard_writer.add_scalar('val/best_loss', best_loss, total_steps)
                tensorboard_writer.add_scalar('val/loss', eval_loss, total_steps)
                tensorboard_writer.add_scalar('epoch', epoch, total_steps)
                tensorboard_writer.add_scalar('lr', scheduler.get_last_lr()[0], total_steps)

            scheduler.step()
            if early_stopping:
                if cooloff > early_stopping:
                    break
    except KeyboardInterrupt as e:
        logger.info("Received KeyboardInterrrupt. Save last model if checkpointing activated")
        if checkpoint_path:
            torch.save(
                        {"epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "loss": best_loss}, f=os.path.join(os.path.split(checkpoint_path)[0], "last.pt")
                    )
        raise e
    
    logger.info("Finished Training")
    if checkpoint_path:
            torch.save(
                        {"epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "loss": best_loss}, f=os.path.join(os.path.split(checkpoint_path)[0], "last.pt")
                    )
    return progress

def train_ruids(model_config: dict, dataset_config:dict, batch_size=32, epochs=20, lr=1e-3, early_stopping: int = None, checkpoint_path: str | Path = None,
                                  tensorboard_writer: SummaryWriter = None, report_every: int = None, n_worker:int = 0, device: str = "cpu", restart_ckpt_path: str | Path = None):

    if checkpoint_path:
        os.makedirs(os.path.split(checkpoint_path)[0],exist_ok=True)
        
    logger.info("Start loading dataset and model")

    dataset = OpTCDataset(**dataset_config)
    dl = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=n_worker, worker_init_fn=worker_init)

    transformations = nn.ModuleList([MLP(**model_config["transformation"]) for _ in range(model_config["n_transformations"])])
    encoder_transformer = MaskedTransformerEncoder(**model_config["encoder"])
    decoder_transformer = TransformerEncoder(**model_config["decoder"])
    transformations.to(device=device)
    encoder_transformer.to(device=device)
    decoder_transformer.to(device=device)

    cooloff = 0
    best_loss = float('inf')
    loss_fnc = nn.MSELoss()

    if restart_ckpt_path:
        ckpt = torch.load(restart_ckpt_path)
        transformations.load_state_dict(ckpt["transformations_state_dict"])
        encoder_transformer.load_state_dict(ckpt["encoder_state_dict"])
        decoder_transformer.load_state_dict(ckpt["decoder_state_dict"])
        
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params=params, lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        best_loss = ckpt["loss"]
        restart_epoch = ckpt["epoch"] + 1
        del ckpt
    else:
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params=params, lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        restart_epoch = 0
    
    restart_string = f"Restart training from epoch {restart_epoch}" if restart_ckpt_path else "Start training from scratch"
    ckp_string = "with" if checkpoint_path else "without"
    es_string = f"with early_stopping after {early_stopping} epochs without improvement" if early_stopping else "without early stopping"
    logger.info(f"{restart_string} for {epochs} epochs, {ckp_string} checkpointing and {es_string}")

    progress = []
    total_steps = 0

def train_autoregressive_encoder(model_config: dict, dataset_config:dict, batch_size=32, epochs=20, early_stopping: int = None, checkpoint_path: str | Path = None,
                                  tensorboard_writer: SummaryWriter = None, report_every: int = None, n_worker:int = 0, device: str = "cpu", restart_ckpt_path: str | Path = None):

    window_size = dataset_config['window_size']
    dataset_config['window_size'] = window_size + 1
    
    if checkpoint_path:
        os.makedirs(os.path.split(checkpoint_path)[0],exist_ok=True)

    logger.info("Start loading dataset and model")

    dataset = OpTCDataset(**dataset_config)
    dl = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=n_worker, worker_init_fn=worker_init)

    model = AutoregressiveTransformerEncoder(**model_config)
    model.to(device=device)

    mask = nn.Transformer.generate_square_subsequent_mask(window_size)
    mask = mask.to(device=device)

    cooloff = 0
    best_loss = float('inf')
    loss_fnc = nn.MSELoss()

    if restart_ckpt_path:
        ckpt = torch.load(restart_ckpt_path)
        model.load_state_dict(ckpt["model_state_dict"])
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(params=params)
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        best_loss = ckpt["loss"]
        restart_epoch = ckpt["epoch"] + 1
        del ckpt
    else:
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(params=params)
        restart_epoch = 0
    
    restart_string = f"Restart training from epoch {restart_epoch}" if restart_ckpt_path else "Start training from scratch"
    ckp_string = "with" if checkpoint_path else "without"
    es_string = f"with early_stopping after {early_stopping} epochs without improvement" if early_stopping else "without early stopping"
    logger.info(f"{restart_string} for {epochs} epochs, {ckp_string} checkpointing and {es_string}")

    progress = []
    total_steps = 0
    try:
        for epoch in range(restart_epoch, epochs):
            logger.info(f"Start epoch {epoch}")
            total_loss = 0
            steps = 0

            for inp in dl:
                
                inp = inp.to(device)

                loss = train_step((inp[:, :window_size, :], mask), inp[:, 1:, :], model=model, loss_criterion=loss_fnc, optimizer=optimizer)

                total_loss += loss.item()
                steps += 1
                total_steps += 1 

                if tensorboard_writer:
                    if steps % report_every == 0:
                        tensorboard_writer.add_scalar('train/loss', loss.item(), total_steps)

            epoch_loss = total_loss / steps
            cooloff += 1
            progress.append(epoch_loss)
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                cooloff = 0
                if checkpoint_path:
                    torch.save(
                        {"epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "loss": best_loss}, f=checkpoint_path
                    )
                    logger.info("Saved Checkpoint")
            logger.info(f"Finished Epoch {epoch} with loss={epoch_loss} and best_loss={best_loss}")
            if tensorboard_writer:
                tensorboard_writer.add_scalar('train/epoch_loss', epoch_loss, total_steps)
                tensorboard_writer.add_scalar('train/best_loss', best_loss, total_steps)
                tensorboard_writer.add_scalar('epoch', epoch, total_steps)

            if early_stopping:
                if cooloff > early_stopping:
                    break
    except KeyboardInterrupt as e:
        logger.info("Received KeyboardInterrrupt. Save last model if checkpointing activated")
        if checkpoint_path:
            torch.save(
                        {"epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "loss": best_loss}, f=os.path.join(os.path.split(checkpoint_path)[0], "last.pt")
                    )
        raise e
    
    logger.info("Finished Training")
    if checkpoint_path:
            torch.save(
                        {"epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "loss": best_loss}, f=os.path.join(os.path.split(checkpoint_path)[0], "last.pt")
                    )
    return progress

def train_masked_encoder(model_config: dict, dataset_config:dict, batch_size=32, epochs=20, lr=1e-3, use_scheduler: bool = False,
                         early_stopping: int = None, checkpoint_path: str | Path = None,
                         tensorboard_writer: SummaryWriter = None, report_every: int = None, n_worker:int = 0, device: str = "cpu", restart_ckpt_path: str | Path = None):
    
    if checkpoint_path:
        os.makedirs(os.path.split(checkpoint_path)[0],exist_ok=True)

    logger.info("Start loading dataset and model")

    dataset = OpTCDataset(**dataset_config)
    dl = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=n_worker, worker_init_fn=worker_init)

    model = MaskedTransformerEncoder(**model_config)
    model.to(device=device)

    cooloff = 0
    best_loss = float('inf')
    def loss_fnc(outputs, targets):
        mask = 1 - outputs[1]
        outputs = outputs[0] * mask
        targets = targets * mask
        return F.mse_loss(outputs, targets, reduction='sum') / mask.sum()

    if restart_ckpt_path:
        ckpt = torch.load(restart_ckpt_path)
        model.load_state_dict(ckpt["model_state_dict"])
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params=params)
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if use_scheduler:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        best_loss = ckpt["loss"]
        restart_epoch = ckpt["epoch"] + 1
        del ckpt
    else:
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params=params, lr=lr)
        if use_scheduler:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        restart_epoch = 0
    
    restart_string = f"Restart training from epoch {restart_epoch}" if restart_ckpt_path else "Start training from scratch"
    ckp_string = "with" if checkpoint_path else "without"
    es_string = f"with early_stopping after {early_stopping} epochs without improvement" if early_stopping else "without early stopping"
    logger.info(f"{restart_string} for {epochs} epochs, {ckp_string} checkpointing and {es_string}")

    progress = []
    total_steps = 0
    try:
        for epoch in range(restart_epoch, epochs):
            logger.info(f"Start epoch {epoch}")
            total_loss = 0
            steps = 0

            for inp in dl:
                
                inp = inp.to(device)
                logger.debug(f"Input shape: {inp.shape}")

                loss = train_step((inp,), inp, model=model, loss_criterion=loss_fnc, optimizer=optimizer)

                total_loss += loss.item()
                steps += 1
                total_steps += 1 

                if tensorboard_writer:
                    if steps % report_every == 0:
                        tensorboard_writer.add_scalar('train/loss', loss.item(), total_steps)
                
            epoch_loss = total_loss / steps
            cooloff += 1
            progress.append(epoch_loss)
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                cooloff = 0
                if checkpoint_path:
                    save_dict = {"epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "best_loss": best_loss, "epoch_loss": epoch}
                    if use_scheduler:
                        save_dict["scheduler_state_dict"] = scheduler.state_dict()
                    torch.save(
                        save_dict, f=checkpoint_path
                    )
                    logger.info("Saved Checkpoint")
            logger.info(f"Finished Epoch {epoch} with loss={epoch_loss} and best_loss={best_loss}")
            if tensorboard_writer:
                tensorboard_writer.add_scalar('train/epoch_loss', epoch_loss, total_steps)
                tensorboard_writer.add_scalar('train/best_loss', best_loss, total_steps)
                tensorboard_writer.add_scalar('epoch', epoch, total_steps)
                if use_scheduler:
                    tensorboard_writer.add_scalar('train/lr', scheduler.get_last_lr()[0], total_steps)

            if early_stopping:
                if cooloff > early_stopping:
                    break
            if use_scheduler:
                scheduler.step()
    except KeyboardInterrupt as e:
        logger.info("Received KeyboardInterrrupt. Save last model if checkpointing activated")
        if checkpoint_path:
            save_dict = {"epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_loss": best_loss, "epoch_loss": epoch}
            if use_scheduler:
                save_dict["scheduler_state_dict"] = scheduler.state_dict()
            torch.save(
                save_dict, f=os.path.join(os.path.split(checkpoint_path)[0], "last.pt")
            )
        raise e
    
    logger.info("Finished Training")
    if checkpoint_path:
        save_dict = {"epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_loss": best_loss, "epoch_loss": epoch}
        if use_scheduler:
            save_dict["scheduler_state_dict"] = scheduler.state_dict()
        torch.save(
            save_dict, f=os.path.join(os.path.split(checkpoint_path)[0], "last.pt")
            )
    return progress

def hyperparameter_tuning_masked_encoder(config, train_dataset_config, eval_dataset_config, n_worker: int = 0, report_every: int = None):

    model_config = {
        "n_layer": config['n_layer'],
        "head":torch.nn.Linear(config['model_dim'], 102),
        "embedding":torch.nn.Linear(102, config['model_dim']),
        "d_model":config['model_dim'],
        "n_head":config['n_head'],
        "dim_feedforward":int(config['factor_dim_feedforward']*config['model_dim']),
        "dropout":0.1,
        "activation":torch.nn.functional.relu,
        "window_size":config['window'],
        "layer_norm_eps":1e-05,
        "norm_first":True, "bias":True,
        "device":None,
        "dtype":None
    }

    batch_size = config['batch_size']
    epochs = config['epochs']
    lr = config['lr']
    patience = config['patience']

    train_dataset_config['window_size'] = config['window']
    eval_dataset_config['window_size'] = config['window']

    model = MaskedTransformerEncoder(**model_config)
    train_dataset = OpTCDatasetWithIndexing(**train_dataset_config)
    eval_dataset = OpTCDatasetWithIndexing(**eval_dataset_config)

    train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=n_worker, worker_init_fn=worker_init)
    eval_dl = torch.utils.data.DataLoader(eval_dataset, batch_size=batch_size, num_workers=n_worker, worker_init_fn=worker_init)

    device = get_devices()[0]
    model.to(device=device)
    model.train()

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params=params, lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    cooloff = 0
    best_loss = float('inf')
    def loss_fnc(outputs, targets):
        mask = 1 - outputs[1]
        outputs = outputs[0] * mask
        targets = targets * mask
        return F.mse_loss(outputs, targets, reduction='sum') / mask.sum()
    total_steps = 0
    for epoch in range(epochs):
        steps = 0
        total_loss = 0
        report_loss = 0
        report_steps = 0


        for inp in train_dl:
            inp = inp.to(device)
            loss = train_step((inp,), inp, model=model, loss_criterion=loss_fnc, optimizer=optimizer)
            total_loss += loss.item()
            report_loss += loss.item()
            report_steps += 1
            steps += 1
            total_steps += 1

            if steps % report_every == 0:
                with torch.no_grad():
                    model.eval()
                    eval_loss = 0
                    eval_steps = 0
                    for eval_inp in eval_dl:

                        eval_inp = eval_inp.to(device)
                        eval_out = model(eval_inp)[0]

                        eval_loss += F.mse_loss(eval_out, eval_inp).item()

                        eval_steps += 1
                    
                    eval_loss /= eval_steps

                    with tempfile.TemporaryDirectory() as tmp_ckpt_dir:
                        tmp_ckpt_path = os.path.join(tmp_ckpt_dir, "ckpt.pt")
                        torch.save(
                            {"epoch": epoch,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "scheduler_state_dict": scheduler.state_dict(),
                            "loss": eval_loss}, f=tmp_ckpt_path
                        )
                        ckpt = Checkpoint.from_directory(tmp_ckpt_dir)
                        train.report(
                            {"epoch": epoch, "steps": total_steps, "val_loss": eval_loss, "train_loss": report_loss / report_steps, "epoch_loss": total_loss / steps, "lr": scheduler.get_last_lr()[0]},
                            checkpoint=ckpt,
                        )

                    report_loss = 0
                    report_steps = 0
                    model.train()
            
        with torch.no_grad():
            model.eval()
            eval_loss = 0
            eval_steps = 0
            for eval_inp in eval_dl:

                eval_inp = eval_inp.to(device)
                eval_out = model(eval_inp)[0]

                eval_loss += F.mse_loss(eval_out, eval_inp).item()

                eval_steps += 1
            
            eval_loss /= eval_steps
            cooloff += 1
            if eval_loss < best_loss:
                best_loss = eval_loss
                cooloff = 0

            with tempfile.TemporaryDirectory() as tmp_ckpt_dir:
                tmp_ckpt_path = os.path.join(tmp_ckpt_dir, "ckpt.pt")
                torch.save(
                    {"epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "loss": eval_loss}, f=tmp_ckpt_path
                )
                ckpt = Checkpoint.from_directory(tmp_ckpt_dir)
                train.report(
                    {"epoch": epoch, "steps": total_steps, "val_loss": eval_loss, "train_loss": report_loss / report_steps, "epoch_loss": total_loss / steps, "lr": scheduler.get_last_lr()[0]},
                    checkpoint = ckpt,
                    )

        scheduler.step()
        model.train()

        if cooloff > patience:
            break

