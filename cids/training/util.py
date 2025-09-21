import torch
from torch import nn
import logging
logger = logging.getLogger(__name__)

def train_step(src: tuple[torch.Tensor, ...], tgt: torch.Tensor, model: nn.Module, loss_criterion: callable, optimizer: torch.optim.Optimizer, accuracy: callable=None):
    
    optimizer.zero_grad()

    outputs = model(*src)
    logger.debug(f"Output shape: {[o.shape for o in outputs]} ")

    loss = loss_criterion(outputs, tgt)

    loss.backward()
    optimizer.step()

    if accuracy is not None:
        acc = accuracy(outputs, tgt)
        return loss, acc
    
    return loss

def train_step_collaborative(src: tuple[torch.Tensor, ...], tgt: torch.Tensor, model: nn.Module, loss_criterion: callable, optimizer: torch.optim.Optimizer, accuracy: callable=None, alpha = 0.):
    
    optimizer.zero_grad()

    outputs = model(*src)
    logger.debug(f"Output shape: {[o.shape for o in outputs]} ")

    loss = loss_criterion(*outputs, tgt, alpha=alpha)

    loss.backward()
    optimizer.step()

    if accuracy is not None:
        acc = accuracy(outputs[0], tgt)
        return loss, acc
    
    return loss

def train_step_ssl(src: tuple[torch.Tensor, ...], tgtx: torch.Tensor, tgty: torch.Tensor, model: nn.Module, loss_criterion: callable, optimizer: torch.optim.Optimizer, accuracy:callable=None, gamma=1.):
    optimizer.zero_grad()

    outputs = model(*src)
    logger.debug(f"Output shape: {[o.shape for o in outputs]} ")

    loss = loss_criterion(outputs, tgtx, tgty, gamma=gamma)

    loss.backward()
    optimizer.step()

    if accuracy is not None:
        acc = accuracy(outputs, tgtx, tgty, gamma=gamma)
        return loss, acc
    
    return loss
