from sklearn import metrics
import torch
from torch.nn.functional import cross_entropy, mse_loss

EPS = 1e-9

def precision(tp ,fp):
    return tp / (tp + fp + EPS)

def recall(tp, fn):
    return tp / (tp + fn + EPS)

def fpr(fp, tn):
    return fp / (fp + tn + EPS)

def f1_score(tp, fp, fn):
    p = precision(tp, fp)
    r = recall(tp, fn)

    return 2 * p * r / (p +r + EPS)

def auc_score(y_true, y_score):
    return metrics.roc_auc_score(y_true, y_score)


def top1_accuracy(y_hat: torch.Tensor, y: torch.Tensor):
    pred = torch.argmax(y_hat, dim=-1)
    correct = torch.sum(pred == y)
    return correct / y_hat.shape[0]

def combined_loss(y_hat: torch.Tensor, y_hat_network: torch.Tensor, y: torch.Tensor, alpha: float = 0.):
    network = cross_entropy(y_hat_network, y)
    agg = cross_entropy(y_hat, y)

    return alpha * network + (1 - alpha) * agg

def inverse_mse(x_hat, x, y, gamma=1.):
    mse = mse_loss(x_hat, x, reduction="none")
    mse = torch.mean(mse, dim=tuple(range(1, mse.dim())))
    mse = (1 + (gamma-1) * y) * (1.-2*y) * mse
    return torch.mean(mse)

def reciprocal_mse(x_hat, x, y, gamma=1.):
    mse = mse_loss(x_hat, x, reduction="none")
    mse = torch.mean(mse, dim=tuple(range(1, mse.dim())))
    mse = (1 + (gamma-1) * y) * (mse ** (1.-2*y))
    return torch.mean(mse)

def mcc(y_true, y_pred):
    return metrics.matthews_corrcoef(y_true, y_pred)

def mcc_from_confusion_matrix(confusion_matrix):
    tn, fp, fn, tp = confusion_matrix[1, 1], confusion_matrix[1, 0], confusion_matrix[0, 1], confusion_matrix[0, 0]
    return (tp * tn - fp * fn) / ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5

def f1_from_confusion_matrix(confusion_matrix):
    tn, fp, fn, tp = confusion_matrix[1, 1], confusion_matrix[1, 0], confusion_matrix[0, 1], confusion_matrix[0, 0]
    return f1_score(tp, fp, fn)

def precision_from_confusion_matrix(confusion_matrix):
    tn, fp, fn, tp = confusion_matrix[1, 1], confusion_matrix[1, 0], confusion_matrix[0, 1], confusion_matrix[0, 0]
    return precision(tp, fp)

def recall_from_confusion_matrix(confusion_matrix):
    tn, fp, fn, tp = confusion_matrix[1, 1], confusion_matrix[1, 0], confusion_matrix[0, 1], confusion_matrix[0, 0]
    return recall(tp, fn)