"""
Implement different strategies to calculate the anomaly score and its corresponding threhold
"""
import logging

import numpy as np

from torch.utils.data import DataLoader
from torch import nn
import torch
from torch.nn import functional as F

from .util.metrics import f1_score

logger = logging.getLogger(__name__)


class AnomalyBase(object):

    def __init__(self):
        self.threshold = 0
    
    def set_threshold(self, distance: torch.Tensor, eval_distance: torch.Tensor = None, IQR = True):
        raise NotImplementedError
    
    def get_scores(self, x, y):
        raise NotImplementedError
    
    def classify(self, x, y):
        raise NotImplementedError
    

class NextNAnomaly(AnomalyBase):
    """
    This class first predicts the next n events from the input window. Anomaly score is the distance between the predicted and the true next n events.
    """
    def __init__(self,n: int, model: nn.Module, device='cpu'):
        super().__init__()
        self.model = model
        self.n = n
        self.device = device

    def set_threshold(self, distance: torch.Tensor, eval_distance: torch.Tensor = None, IQR = True):
        if not IQR:
            raise ValueError("cannot calculate threshold based on the provided data")
        
        if IQR:
            q1 = torch.quantile(distance, 0.25)
            q3 = torch.quantile(distance, 0.75)
            iqr = q3 - q1
            self.threshold = (q3 + 1.5 * iqr).item()
            return self.threshold
        else:
            raise NotImplementedError("So far only IQR supported")
        
    def get_scores(self, x, y):
        mask = nn.Transformer.generate_square_subsequent_mask(x.shape[1])
        mask = mask.to(device=self.device)

        y_pred = torch.zeros_like(y)

        with torch.no_grad():
            for i in range(y.shape[1]):
                next_event = self.model(x, mask)[:, -1, :]
                y_pred[:, i, :] = next_event.cpu()

                x[:, :-1, :] = x[:, 1:, :]
                x[:, -1, :] = next_event

        scores = torch.sqrt(torch.sum((y - y_pred) ** 2, dim=(1, 2)))
        return scores
    
    def classify(self, x: torch.Tensor, y: torch.Tensor = None, score_mode=False):
        # If score_mode=True assume that x is a np.array of scores
        logger.debug(f"Calculating predictions and anomaly score")
        if self.threshold == 0:
            logger.warning("Threshold is 0. This may not lead to good resutls. Consider running get_threshold first")

        if not score_mode:
            scores = self.get_scores(x, y)
        else:
            scores = x

        labels = (scores > self.threshold).int()
        return labels, scores
    

class NextWindowAnomaly(AnomalyBase):

    """
    This class first predicts the next window of events shiftet by tau from the start of the input window. Anomaly score is the distance between the predicted and the true next window.
    """
    
    def __init__(self, tau: int, model: nn.Module, device='cpu'):
        super().__init__()
        self.model = model
        self.tau = tau
        self.device = device

    def set_threshold(self, distance: torch.Tensor, eval_distance: torch.Tensor = None, IQR = True):
        if not IQR:
            raise ValueError("cannot calculate threshold based on the provided data")
        
        if IQR:
            q1 = torch.quantile(distance, 0.25)
            q3 = torch.quantile(distance, 0.75)
            iqr = q3 - q1
            self.threshold = (q3 + 1.5 * iqr).item()
            return self.threshold
        else:
            raise NotImplementedError("So far only IQR supported")
        
    def get_scores(self, x, y):
        shift = self.tau - 1
        mask = nn.Transformer.generate_square_subsequent_mask(x.shape[1])
        if shift != 0:
            mask = torch.roll(mask, -shift, dims=0)
            mask[-shift:, :] = 0
        mask = mask.to(device=self.device)

        with torch.no_grad():
            y_pred = self.model(x, mask).cpu()

        scores = torch.sqrt(torch.sum((y - y_pred) ** 2, dim=(1, 2)))
        return scores
    
    def classify(self, x: torch.Tensor, y: torch.Tensor = None, score_mode=False):
        # If score_mode=True assume that x is a np.array of scores
        logger.debug(f"Calculating predictions and anomaly score")
        if self.threshold == 0:
            logger.warning("Threshold is 0. This may not lead to good resutls. Consider running get_threshold first")

        if not score_mode:
            scores = self.get_scores(x, y)
        else:
            scores = x

        labels = (scores > self.threshold).int()
        return labels, scores
    

class ReconstructionAnomaly(AnomalyBase):
    """
    This class calculates the anomaly score as the distance between the input and the reconstructed input.
    """
    
    def __init__(self, model: nn.Module, device='cpu'):
        super().__init__()
        self.model = model
        self.model.to(device=device)
        self.device = device

    def set_threshold(self, distance: torch.Tensor, mal_distance: torch.Tensor = None, IQR = True):
        if not IQR and mal_distance is None:
            raise ValueError("cannot calculate threshold based on the provided data")
        
        if IQR:
            q1 = torch.quantile(distance, 0.25)
            q3 = torch.quantile(distance, 0.75)
            iqr = q3 - q1
            self.threshold = (q3 + 1.5 * iqr).item()
            return self.threshold
        else:
            max_score = max([distance.max().item(), mal_distance.max().item()])
            distance = distance.to(self.device)
            mal_distance = mal_distance.to(self.device)

            thresholds = torch.tensor(np.linspace(0, max_score, 1000)[:-2], device=self.device)

            tp = mal_distance[:, None] > thresholds
            fp = distance[:, None] > thresholds
            fn = mal_distance[:, None] < thresholds
            tp = torch.sum(tp, dim=0).cpu()
            fp = torch.sum(fp, dim=0).cpu()
            fn = torch.sum(fn, dim=0).cpu()

            f1s = f1_score(tp, fp, fn)

            self.threshold = thresholds[torch.argmax(f1s)].item()

            return self.threshold, f1s, thresholds
        
    def get_scores(self, x):
        with torch.no_grad():
            x = x.to(device=self.device)
            x_pred = self.model(x)
            if isinstance(x_pred, tuple):
                x_pred = x_pred[0]
        
        scores = torch.mean(((x - x_pred) ** 2).view(x.size(0), -1), dim=1)
        return scores
    
    def classify(self, x: torch.Tensor, score_mode=False):
        # If score_mode=True assume that x is a np.array of scores
        logger.debug(f"Calculating predictions and anomaly score")
        if self.threshold == 0:
            logger.warning("Threshold is 0. This may not lead to good resutls. Consider running get_threshold first")

        if not score_mode:
            scores = self.get_scores(x)
        else:
            scores = x

        labels = (scores > self.threshold).int()
        return labels, scores
