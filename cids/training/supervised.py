from pathlib import Path
import logging
import os
import tempfile
import json

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from sklearn.ensemble import RandomForestClassifier

from ray.train.torch import get_devices
from ray import train
from ray.train import Checkpoint

from ..data import OpTCDatasetWithIndexing, Undersampling, OpTCDataset, worker_init_wrapped, get_SCVIC_dataloader, get_SCVIC_Dataset, get_SCVIC_dataloader_binary
from ..models.nn import MLP, CollaborativeIDSNet
from ..models.transformer import TransformerEncoder
from .util import train_step, train_step_collaborative
from ..util import metrics

logger = logging.getLogger(__name__)

def train_mlp(model_config: dict, ds_type:str, ds_args:dict, eval_ds_args:dict = None, batch_size=32, epochs=20, lr=1e-3, patience: int = None,
              loss_fnc: callable = nn.BCEWithLogitsLoss, accuracy: callable = None, checkpoint_path: str | Path = None,
              tensorboard_writer: SummaryWriter = None, report_every: int = None, device: str = "cpu", restart_ckpt_path: str | Path = None):
    
    if checkpoint_path:
        os.makedirs(os.path.split(checkpoint_path)[0],exist_ok=True)

    logger.info("Start loading dataset and model")
    if "optc" in ds_type.lower():
        undersampling_dist = ds_args.pop("undersampling")
        train_dataset = Undersampling(OpTCDataset(**ds_args), **undersampling_dist)
        logger.info(f"Trainingsdistribution without undersampling:\n {json.dumps(train_dataset.actual_dist, indent=2)}")
        
        undersampling_dist = eval_ds_args.pop("undersampling")
        eval_dataset = Undersampling(OpTCDataset(**eval_ds_args), **undersampling_dist)
        logger.info(f"Evalsdistribution without undersampling:\n {json.dumps(train_dataset.actual_dist, indent=2)}")

        train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=1, worker_init_fn=worker_init_wrapped)
        eval_dl = torch.utils.data.DataLoader(eval_dataset, batch_size=batch_size, num_workers=1, worker_init_fn=worker_init_wrapped)

    elif "scvic_binary" in ds_type.lower():
        if eval_ds_args is not None:
            logger.warning(f"Eval Dataset is ignored for type scivc. Eval dataloader is generated based of train config")

        train_dl, eval_dl = get_SCVIC_dataloader_binary(**ds_args)

    elif "scvic" in ds_type.lower():
        if eval_ds_args is not None:
            logger.warning(f"Eval Dataset is ignored for type scivc. Eval dataloader is generated based of train config")
        train_dl, eval_dl = get_SCVIC_dataloader(**ds_args)
    
    else:
        raise ValueError(f"ds_type {ds_type} is not known")

    model = MLP(**model_config)
    model.to(device=device)

    cooloff = 0
    best_loss = float('inf')

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
            total_acc = 0
            steps = 0

            for data in train_dl:
                label = data[-1]
                logger.debug(f"lenght data: {len(data)}")
                if len(data) == 2:
                    inp = torch.flatten(data[0], start_dim=1)
                else:
                    inps = [torch.flatten(data[i], start_dim=1) for i in range(len(data) - 1)]
                    inp = torch.cat(inps, dim=1)
                inp = inp.to(device)
                label = label.to(device=device)

                loss, acc = train_step((inp,), label, model=model, loss_criterion=loss_fnc, optimizer=optimizer, accuracy=accuracy)

                total_loss += loss.item()
                total_acc += acc.item()
                steps += 1
                total_steps += 1

                if report_every is not None and steps % report_every == 0:
                    with torch.no_grad():
                        model.eval()
                        eval_loss = 0
                        eval_acc = 0
                        eval_steps = 0
                        for eval_data in eval_dl:
                            eval_label = eval_data[-1]
                            if len(eval_data) == 2:
                                eval_inp = torch.flatten(eval_data[0], start_dim=1)
                            else:
                                eval_inps = [torch.flatten(eval_data[i], start_dim=1) for i in range(len(data) - 1)]
                                eval_inp = torch.cat(eval_inps, dim=1)
                            eval_inp, eval_label = eval_inp.to(device), eval_label.to(device=device)
                            eval_out = model(eval_inp)

                            eval_loss +=loss_fnc(eval_out, eval_label).item()
                            eval_acc += accuracy(eval_out, eval_label).item()

                            eval_steps += 1
                    
                    eval_loss /= eval_steps
                    eval_acc /= eval_steps
                    model.train()
                    if tensorboard_writer:
                        tensorboard_writer.add_scalar('val/loss', eval_loss, total_steps)
                        tensorboard_writer.add_scalar('val/acc', eval_acc, total_steps)

            epoch_loss = total_loss / steps
            epoch_acc = total_acc / steps
            cooloff += 1
            if eval_dl is not None:
                with torch.no_grad():
                    model.eval()
                    eval_loss = 0
                    eval_acc = 0
                    eval_steps = 0
                    for eval_data in eval_dl:
                        eval_label = eval_data[-1]
                        if len(eval_data) == 2:
                            eval_inp = torch.flatten(eval_data[0], start_dim=1)
                        else:
                            eval_inps = [torch.flatten(eval_data[i], start_dim=1) for i in range(len(data) - 1)]
                            eval_inp = torch.cat(eval_inps, dim=1)
                        eval_inp, eval_label = eval_inp.to(device), eval_label.to(device=device)
                        eval_out = model(eval_inp)

                        eval_loss +=loss_fnc(eval_out, eval_label).item()
                        eval_acc += accuracy(eval_out, eval_label).item()

                        eval_steps += 1
                    
                    eval_loss /= eval_steps
                    eval_acc /= eval_steps
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
                tensorboard_writer.add_scalar('train/acc', epoch_acc, total_steps)
                tensorboard_writer.add_scalar('val/best_loss', best_loss, total_steps)
                tensorboard_writer.add_scalar('val/loss', eval_loss, total_steps)
                tensorboard_writer.add_scalar('val/acc', eval_acc, total_steps)
                tensorboard_writer.add_scalar('epoch', epoch, total_steps)
                tensorboard_writer.add_scalar('lr', scheduler.get_last_lr()[0], total_steps)

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

def train_mlp_ray(config: dict, ds_type:str, ds_args:dict, eval_ds_args:dict = None, epochs=20, batch_size=8, loss_fnc=nn.BCEWithLogitsLoss(), accuracy: callable = None, early_stopping: int = None):

    logger.info("Start loading dataset and model")
    if "optc" in ds_type.lower():
        undersampling_dist = ds_args.pop("undersampling")
        train_dataset = Undersampling(OpTCDataset(**ds_args), **undersampling_dist)
        logger.info(f"Trainingsdistribution without undersampling:\n {json.dumps(train_dataset.actual_dist, indent=2)}")
        
        undersampling_dist = eval_ds_args.pop("undersampling")
        eval_dataset = Undersampling(OpTCDataset(**eval_ds_args), **undersampling_dist)
        logger.info(f"Evalsdistribution without undersampling:\n {json.dumps(eval_dataset.actual_dist, indent=2)}")

        train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=1, worker_init_fn=worker_init_wrapped)
        eval_dl = torch.utils.data.DataLoader(eval_dataset, batch_size=batch_size, num_workers=1, worker_init_fn=worker_init_wrapped)

    elif "scvic_binary" in ds_type.lower():
        if eval_ds_args is not None:
            logger.warning(f"Eval Dataset is ignored for type scivc. Eval dataloader is generated based of train config")

        train_dl, eval_dl = get_SCVIC_dataloader_binary(**ds_args)

    elif "scvic" in ds_type.lower():
        if eval_ds_args is not None:
            logger.warning(f"Eval Dataset is ignored for type scivc. Eval dataloader is generated based of train config")
        train_dl, eval_dl = get_SCVIC_dataloader(**ds_args)
    
    else:
        raise ValueError(f"ds_type {ds_type} is not known")

    device = get_devices()[0]
    logger.debug(f"Use device {device}")
    model_config = config["model"]
    model = MLP(**model_config)
    model.to(device=device)

    cooloff = 0
    best_loss = float('inf')

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params=params, lr=config["lr"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    logger.info(f"Start training for {epochs} epochs, under ray conditions")

    total_steps = 0
    try:
        for epoch in range(epochs):
            logger.info(f"Start epoch {epoch}")
            total_loss = 0
            total_acc = 0
            steps = 0

            for data in train_dl:
                label = data[-1]
                if len(data) == 2:
                    inp = torch.flatten(data[0], start_dim=1)
                else:
                    inps = [torch.flatten(data[i], start_dim=1) for i in range(len(data) - 1)]
                    inp = torch.cat(inps, dim=1)
                inp = inp.to(device)
                label = label.to(device=device)

                loss, acc = train_step((inp,), label, model=model, loss_criterion=loss_fnc, optimizer=optimizer, accuracy=accuracy)

                total_loss += loss.item()
                total_acc += acc.item()
                
                steps += 1
                total_steps += 1

            epoch_loss = total_loss / steps
            epoch_acc = total_acc / steps
            cooloff += 1
            with torch.no_grad():
                model.eval()
                eval_loss = 0
                eval_acc = 0
                eval_steps = 0
                for eval_data in eval_dl:
                    eval_label = eval_data[-1]
                    if len(eval_data) == 2:
                        eval_inp = torch.flatten(eval_data[0], start_dim=1)
                    else:
                        eval_inps = [torch.flatten(eval_data[i], start_dim=1) for i in range(len(data) - 1)]
                        eval_inp = torch.cat(eval_inps, dim=1)
                    eval_inp, eval_label = eval_inp.to(device), eval_label.to(device=device)
                    eval_out = model(eval_inp)

                    eval_loss +=loss_fnc(eval_out, eval_label).item()
                    eval_acc += accuracy(eval_out, eval_label).item()

                    eval_steps += 1
                
                eval_loss /= eval_steps
                eval_acc /= eval_steps
            model.train()

            if eval_loss < best_loss:
                best_loss = eval_loss
                cooloff = 0

            train.report(
                {
                    "epoch": epoch,
                    "steps": total_steps,
                    "train/loss": epoch_loss,
                    "train/acc": epoch_acc,
                    "val/loss": eval_loss,
                    "val/best_loss": best_loss,
                    "val/acc": eval_acc,
                    "lr": scheduler.get_last_lr()[0]
                }
            )
            logger.info(f"Finished Epoch {epoch} with loss={epoch_loss} and best_loss={best_loss}")

            scheduler.step()
            if early_stopping:
                if cooloff > early_stopping:
                    break
    except KeyboardInterrupt as e:
        logger.info("Received KeyboardInterrrupt. Save last model if checkpointing activated")
        train.report(
            {
                "epoch": epoch,
                "steps": total_steps,
                "train/loss": epoch_loss,
                "train/acc": epoch_acc,
                "val/loss": eval_loss,
                "val/best_loss": best_loss,
                "val/acc": eval_acc,
                "lr": scheduler.get_last_lr()[0]
            }
        )
        raise e
    
    logger.info("Finished Training")

def train_cids(model_config, ds_type:str, ds_args: dict, eval_ds_args: dict = None, loss_fnc: callable = metrics.combined_loss, accuracy: callable = None, patience: int = None,
               device="cpu", batch_size: int = 8, lr: float = 1e-3, epochs: int = 20, alpha: float = 0., checkpoint_path: str | Path = None, 
               tensorboard_writer: SummaryWriter = None):
    logger.info("Start loading dataset and model")

    logger.debug(f"Use device {device}")

    logger.info("Load model")

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

    logger.info("load data")
    if "optc" in ds_type.lower():
        undersampling_dist = ds_args.pop("undersampling_dist")
        train_dataset = Undersampling(OpTCDatasetWithIndexing(**ds_args), **undersampling_dist)
        logger.info(f"Trainingsdistribution without undersampling:\n {json.dumps(train_dataset.actual_dist, indent=2)}")
        
        undersampling_dist = eval_ds_args.pop("undersampling_dist")
        eval_dataset = Undersampling(OpTCDatasetWithIndexing(**eval_ds_args), **undersampling_dist)
        logger.info(f"Evalsdistribution without undersampling:\n {json.dumps(train_dataset.actual_dist, indent=2)}")

        train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=1, worker_init_fn=worker_init_wrapped)
        eval_dl = torch.utils.data.DataLoader(eval_dataset, batch_size=batch_size, num_workers=1, worker_init_fn=worker_init_wrapped)

    elif "scvic_binary" in ds_type.lower():
        if eval_ds_args is not None:
            logger.warning(f"Eval Dataset is ignored for type scivc. Eval dataloader is generated based of train config")
            
        train_dl, eval_dl = get_SCVIC_dataloader_binary(**ds_args)

    elif "scvic" in ds_type.lower():
        if eval_ds_args is not None:
            logger.warning(f"Eval Dataset is ignored for type scivc. Eval dataloader is generated based of train config")
        train_dl, eval_dl = get_SCVIC_dataloader(**ds_args)
    
    else:
        raise ValueError(f"ds_type {ds_type} is not known")

    model.to(device=device)

    if checkpoint_path:
        os.makedirs(os.path.split(checkpoint_path)[0],exist_ok=True)

    cooloff = 0
    best_loss = float('inf')

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params=params, lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    logger.info(f"Start training for {epochs} epochs, under ray conditions")

    total_steps = 0
    alpha = alpha
    try:
        for epoch in range(epochs):
            logger.info(f"Start epoch {epoch}")
            total_loss = 0
            total_acc = 0
            steps = 0

            for data in train_dl:
                label = data[-1]

                inp_network = data[0].to(device)
                inp_host = data[1].to(device)
                inp_embedding = data[2].to(device)
                label = label.to(device=device)

                loss, acc = train_step_collaborative(
                    (inp_network, inp_host, inp_embedding),
                    label,
                    model=model,
                    loss_criterion=loss_fnc,
                    optimizer=optimizer,
                    accuracy=accuracy,
                    alpha=alpha
                )

                total_loss += loss.item()
                total_acc += acc.item()
                
                steps += 1
                total_steps += 1

            epoch_loss = total_loss / steps
            epoch_acc = total_acc / steps
            cooloff += 1
            with torch.no_grad():
                model.eval()
                eval_loss = 0
                eval_loss_agg = 0
                eval_loss_net = 0
                eval_acc = 0
                eval_acc_net = 0
                eval_steps = 0
                for eval_data in eval_dl:
                    eval_label = eval_data[-1]

                    eval_inp_network = eval_data[0].to(device)
                    eval_inp_host = eval_data[1].to(device)
                    eval_inp_embedding = eval_data[2].to(device)
                    eval_label = eval_label.to(device=device)

                    eval_out, eval_out_network = model(eval_inp_network, eval_inp_host, eval_inp_embedding)

                    eval_loss +=loss_fnc(eval_out, eval_out_network, eval_label, alpha=alpha).item()
                    eval_loss_agg += F.cross_entropy(eval_out, eval_label).item()
                    eval_loss_net += F.cross_entropy(eval_out_network, eval_label).item()
                    eval_acc += accuracy(eval_out, eval_label).item()
                    eval_acc_net+= accuracy(eval_out_network, eval_label).item()                    

                    eval_steps += 1
                
                eval_loss /= eval_steps
                eval_loss_agg /= eval_steps
                eval_loss_net /= eval_steps
                eval_acc /= eval_steps
                eval_acc_net /= eval_steps
            model.train()

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

            if tensorboard_writer:
                tensorboard_writer.add_scalar('train/loss', epoch_loss, total_steps)
                tensorboard_writer.add_scalar('train/acc', epoch_acc, total_steps)
                tensorboard_writer.add_scalar('val/best_loss', best_loss, total_steps)
                tensorboard_writer.add_scalar('val/loss', eval_loss, total_steps)
                tensorboard_writer.add_scalar('val/loss_agg', eval_loss_agg, total_steps)
                tensorboard_writer.add_scalar('val/loss_net', eval_loss_net, total_steps)
                tensorboard_writer.add_scalar('val/acc', eval_acc, total_steps)
                tensorboard_writer.add_scalar('val/acc_net', eval_acc_net, total_steps)
                tensorboard_writer.add_scalar('epoch', epoch, total_steps)
                tensorboard_writer.add_scalar('lr', scheduler.get_last_lr()[0], total_steps)

            logger.info(f"Finished Epoch {epoch} with loss={epoch_loss} and best_loss={best_loss}")

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
    
def train_cids_ray(config: dict, ds_args: dict, loss_fnc=metrics.combined_loss, accuracy: callable = None, early_stopping: int = None):
    
    logger.info("Start loading dataset and model")

    device = get_devices()[0]
    # device = "cuda:0"
    logger.debug(f"Use device {device}")

    logger.info("Load model")

    network_config = config["model"]["network"]
    host_config = config["model"]["host"]
    host_config["dim_feedforward"] = int(host_config.pop("factor_dim_feedforward") * host_config["d_model"])
    embedding_config = config["model"]["embedding"]
    embedding_config["dim_feedforward"] = int(embedding_config.pop("factor_dim_feedforward") * embedding_config["d_model"])
    aggregation_config = config["model"]["aggregation"]
    aggregation_config["input_dim"] = int(network_config["output_dim"] \
        + host_config["d_model"] * host_config["max_len"] \
        + (embedding_config["head"] if embedding_config["head"] is not None else embedding_config["d_model"]) * embedding_config["max_len"])

    model = CollaborativeIDSNet(
        network_encoder=MLP(**network_config),
        host_encoder=TransformerEncoder(**host_config),
        embedding_encoder=TransformerEncoder(**embedding_config),
        aggregation_module=MLP(**aggregation_config)
    )

    logger.info("load data")
    
    train_dl, eval_dl = get_SCVIC_dataloader(**ds_args)

    model.to(device=device)

    cooloff = 0
    best_loss = float('inf')

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params=params, lr=config["lr"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])
    
    logger.info(f"Start training for {config['epochs']} epochs, under ray conditions")

    total_steps = 0
    alpha = config["alpha"]
    try:
        for epoch in range(config["epochs"]):
            logger.info(f"Start epoch {epoch}")
            total_loss = 0
            total_acc = 0
            steps = 0

            for data in train_dl:
                label = data[-1]

                inp_network = data[0].to(device)
                inp_host = data[1].to(device)
                inp_embedding = data[2].to(device)
                label = label.to(device=device)

                loss, acc = train_step_collaborative(
                    (inp_network, inp_host, inp_embedding),
                    label,
                    model=model,
                    loss_criterion=loss_fnc,
                    optimizer=optimizer,
                    accuracy=accuracy,
                    alpha=alpha
                )

                total_loss += loss.item()
                total_acc += acc.item()
                
                steps += 1
                total_steps += 1

            epoch_loss = total_loss / steps
            epoch_acc = total_acc / steps
            cooloff += 1
            with torch.no_grad():
                model.eval()
                eval_loss = 0
                eval_loss_agg = 0
                eval_loss_net = 0
                eval_acc = 0
                eval_acc_net = 0
                eval_steps = 0
                for eval_data in eval_dl:
                    eval_label = eval_data[-1]

                    eval_inp_network = eval_data[0].to(device)
                    eval_inp_host = eval_data[1].to(device)
                    eval_inp_embedding = eval_data[2].to(device)
                    eval_label = eval_label.to(device=device)

                    eval_out, eval_out_network = model(eval_inp_network, eval_inp_host, eval_inp_embedding)

                    eval_loss +=loss_fnc(eval_out, eval_out_network, eval_label, alpha=alpha).item()
                    eval_loss_agg += F.cross_entropy(eval_out, eval_label).item()
                    eval_loss_net += F.cross_entropy(eval_out_network, eval_label).item()
                    eval_acc += accuracy(eval_out, eval_label).item()
                    eval_acc_net+= accuracy(eval_out_network, eval_label).item()                    

                    eval_steps += 1
                
                eval_loss /= eval_steps
                eval_loss_agg /= eval_steps
                eval_loss_net /= eval_steps
                eval_acc /= eval_steps
                eval_acc_net /= eval_steps
            model.train()

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
                    {
                        "epoch": epoch,
                        "steps": total_steps,
                        "train/loss": epoch_loss,
                        "train/acc": epoch_acc,
                        "val/loss": eval_loss,
                        "val/loss_agg": eval_loss_agg,
                        "val/loss_net": eval_loss_net,
                        "val/best_loss": best_loss,
                        "val/acc": eval_acc,
                        "val/acc_net": eval_acc_net,
                        "lr": scheduler.get_last_lr()[0]
                    },
                    checkpoint=ckpt,
                )
            logger.info(f"Finished Epoch {epoch} with loss={epoch_loss} and best_loss={best_loss}")

            scheduler.step()
            if early_stopping:
                if cooloff > early_stopping:
                    break
    except KeyboardInterrupt as e:
        logger.info("Received KeyboardInterrrupt. Save last model if checkpointing activated")
        train.report(
            {
                "epoch": epoch,
                "steps": total_steps,
                "train/loss": epoch_loss,
                "train/acc": epoch_acc,
                "val/loss": eval_loss,
                "val/loss_agg": eval_loss_agg,
                "val/loss_net": eval_loss_net,
                "val/best_loss": best_loss,
                "val/acc": eval_acc,
                "val/acc_net": eval_acc_net,
                "lr": scheduler.get_last_lr()[0]
            },
            checkpoint=ckpt,
        )
        raise e
    
    logger.info("Finished Training")

def train_rf_ray(config: dict, ds_type:str, ds_args: dict):
    logger.info("Start loading dataset and model")

    if "scvic" in ds_type.lower():
        dss = get_SCVIC_Dataset(ds_args["exclude"], train=True, validation=True, test=False, random_state=ds_args["seed"],
                                network_data=ds_args["network_data"], host_data=ds_args["host_data"], host_embeddings=ds_args["host_embeddings"])
        
        train_data = (
            np.array([np.concatenate([x[i].numpy().flatten() for i in range(len(x) - 1)]) for x in dss["train"].data]),
            np.array([y[-1] for y in dss["train"].data])            
        )
        val_data = (
            np.array([np.concatenate([x[i].numpy().flatten() for i in range(len(x) - 1)]) for x in dss["val"].data]),
            np.array([y[-1] for y in dss["val"].data])
        )
    
    else:
        raise ValueError("Wrong Datasettype provided")
    
    logger.info("Start Training")
    
    rf = RandomForestClassifier(**config)
    rf.fit(train_data[0], train_data[1])
    train_acc = rf.score(train_data[0], train_data[1])
    val_acc = rf.score(val_data[0], val_data[1])

    train.report({"train/acc": train_acc, "val/acc": val_acc})

    logger.info("Finished Training")

def train_transformer_mlp_ray(config: dict, ds_args:dict, eval_ds_args:dict = None, ds_type=None, epochs=20, batch_size = 8, loss_fnc = nn.BCEWithLogitsLoss(), accuracy:callable = None, early_stopping: int = None):
    logger.info("Start loading dataset and model")
    undersampling_dist = ds_args.pop("undersampling")
    train_dataset = Undersampling(OpTCDataset(**ds_args), **undersampling_dist)
    logger.info(f"Trainingsdistribution without undersampling:\n {json.dumps(train_dataset.actual_dist, indent=2)}")
    
    undersampling_dist = eval_ds_args.pop("undersampling")
    eval_dataset = Undersampling(OpTCDataset(**eval_ds_args), **undersampling_dist)
    logger.info(f"Evalsdistribution without undersampling:\n {json.dumps(eval_dataset.actual_dist, indent=2)}")

    train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=1, worker_init_fn=worker_init_wrapped)
    eval_dl = torch.utils.data.DataLoader(eval_dataset, batch_size=batch_size, num_workers=1, worker_init_fn=worker_init_wrapped)


    device = get_devices()[0]
    logger.debug(f"Use device {device}")
    model_config = config["model"]
    encoder_config = model_config["encoder"]
    encoder_config["dim_feedforward"] = int(encoder_config.pop("factor_dim_feedforward") * encoder_config["d_model"])
    encoder_config["embedding"] = None if encoder_config["d_model"] == encoder_config["embedding"] else encoder_config["embedding"]
    encoder = TransformerEncoder(**encoder_config)

    head_config = model_config["head"]
    head_config["input_dim"] = ds_args["window_size"] * encoder_config["d_model"]
    head = MLP(**head_config)
    model = nn.Sequential(encoder, nn.Flatten(), head)
    model.to(device=device)

    cooloff = 0
    best_loss = float('inf')

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params=params, lr=config["lr"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    logger.info(f"Start training for {epochs} epochs, under ray conditions")

    total_steps = 0
    try:
        for epoch in range(epochs):
            logger.info(f"Start epoch {epoch}")
            total_loss = 0
            total_acc = 0
            steps = 0

            for inp, label in train_dl:

                inp = inp.to(device)
                label = label.to(device=device)

                loss, acc = train_step((inp,), label, model=model, loss_criterion=loss_fnc, optimizer=optimizer, accuracy=accuracy)

                total_loss += loss.item()
                total_acc += acc.item()
                
                steps += 1
                total_steps += 1

            epoch_loss = total_loss / steps
            epoch_acc = total_acc / steps
            cooloff += 1
            with torch.no_grad():
                model.eval()
                eval_loss = 0
                eval_acc = 0
                eval_steps = 0
                for eval_inp, eval_label in eval_dl:
                    eval_inp, eval_label = eval_inp.to(device), eval_label.to(device=device)
                    eval_out = model(eval_inp)

                    eval_loss +=loss_fnc(eval_out, eval_label).item()
                    eval_acc += accuracy(eval_out, eval_label).item()

                    eval_steps += 1
                
                eval_loss /= eval_steps
                eval_acc /= eval_steps
            model.train()

            if eval_loss < best_loss:
                best_loss = eval_loss
                cooloff = 0

            train.report(
                {
                    "epoch": epoch,
                    "steps": total_steps,
                    "train/loss": epoch_loss,
                    "train/acc": epoch_acc,
                    "val/loss": eval_loss,
                    "val/best_loss": best_loss,
                    "val/acc": eval_acc,
                    "lr": scheduler.get_last_lr()[0]
                }
            )
            logger.info(f"Finished Epoch {epoch} with loss={epoch_loss} and best_loss={best_loss}")

            scheduler.step()
            if early_stopping:
                if cooloff > early_stopping:
                    break
    except KeyboardInterrupt as e:
        logger.info("Received KeyboardInterrrupt. Save last model if checkpointing activated")
        train.report(
            {
                "epoch": epoch,
                "steps": total_steps,
                "train/loss": epoch_loss,
                "train/acc": epoch_acc,
                "val/loss": eval_loss,
                "val/best_loss": best_loss,
                "val/acc": eval_acc,
                "lr": scheduler.get_last_lr()[0]
            }
        )
        raise e
    
    logger.info("Finished Training")

def train_transformer_mlp(model_config: dict, ds_type:str, ds_args:dict, eval_ds_args:dict = None, epochs=20, batch_size = 8,  lr=1e-3,
                              loss_fnc = nn.BCEWithLogitsLoss(), accuracy:callable = None, patience: int = None, checkpoint_path = None,
                              tensorboard_writer: SummaryWriter = None, device: str = "cpu", restart_ckpt_path: str | Path = None):
    logger.info("Start loading dataset and model")
    undersampling_dist = ds_args.pop("undersampling")
    train_dataset = Undersampling(OpTCDataset(**ds_args), **undersampling_dist)
    logger.info(f"Trainingsdistribution without undersampling:\n {json.dumps(train_dataset.actual_dist, indent=2)}")
    
    undersampling_dist = eval_ds_args.pop("undersampling")
    eval_dataset = Undersampling(OpTCDataset(**eval_ds_args), **undersampling_dist)
    logger.info(f"Evalsdistribution without undersampling:\n {json.dumps(eval_dataset.actual_dist, indent=2)}")

    train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=1, worker_init_fn=worker_init_wrapped)
    eval_dl = torch.utils.data.DataLoader(eval_dataset, batch_size=batch_size, num_workers=1, worker_init_fn=worker_init_wrapped)

    logger.debug(f"Use device {device}")
    encoder_config = model_config["encoder"]
    encoder = TransformerEncoder(**encoder_config)

    head_config = model_config["head"]
    head = MLP(**head_config)
    model = nn.Sequential(encoder, nn.Flatten(), head)
    model.to(device=device)

    cooloff = 0
    best_loss = float('inf')

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

    total_steps = 0
    try:
        for epoch in range(restart_epoch, epochs):
            logger.info(f"Start epoch {epoch}")
            total_loss = 0
            total_acc = 0
            steps = 0

            for inp, label in train_dl:

                inp = inp.to(device)
                label = label.to(device=device)

                loss, acc = train_step((inp,), label, model=model, loss_criterion=loss_fnc, optimizer=optimizer, accuracy=accuracy)

                total_loss += loss.item()
                total_acc += acc.item()
                
                steps += 1
                total_steps += 1

            epoch_loss = total_loss / steps
            epoch_acc = total_acc / steps
            cooloff += 1
            with torch.no_grad():
                model.eval()
                eval_loss = 0
                eval_acc = 0
                eval_steps = 0
                for eval_inp, eval_label in eval_dl:
                    eval_inp, eval_label = eval_inp.to(device), eval_label.to(device=device)
                    eval_out = model(eval_inp)

                    eval_loss +=loss_fnc(eval_out, eval_label).item()
                    eval_acc += accuracy(eval_out, eval_label).item()

                    eval_steps += 1
                
                eval_loss /= eval_steps
                eval_acc /= eval_steps
            model.train()

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
                tensorboard_writer.add_scalar('train/acc', epoch_acc, total_steps)
                tensorboard_writer.add_scalar('val/best_loss', best_loss, total_steps)
                tensorboard_writer.add_scalar('val/loss', eval_loss, total_steps)
                tensorboard_writer.add_scalar('val/acc', eval_acc, total_steps)
                tensorboard_writer.add_scalar('epoch', epoch, total_steps)
                tensorboard_writer.add_scalar('lr', scheduler.get_last_lr()[0], total_steps)

            scheduler.step()
            if patience:
                if cooloff > patience:
                    break
    except KeyboardInterrupt as e:
        logger.info("Received KeyboardInterrrupt. Save last model if checkpointing activated")
        train.report(
            {
                "epoch": epoch,
                "steps": total_steps,
                "train/loss": epoch_loss,
                "train/acc": epoch_acc,
                "val/loss": eval_loss,
                "val/best_loss": best_loss,
                "val/acc": eval_acc,
                "lr": scheduler.get_last_lr()[0]
            }
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
            