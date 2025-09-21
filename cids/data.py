import h5py
import logging
import pickle as pkl
import os
import random
from typing import Any, Iterator

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

from .util import optc_util as opil
from .util import misc_funcs as misc

logger = logging.getLogger(__name__)

SCVIC_CIDS_CLASSES = {
    'Benign': 0,
    'Bot': 1,
    'DoS-SlowHTTPTest': 2,
    'DoS-Hulk': 3,
    'Brute Force -Web': 4,
    'Brute Force -XSS': 5,
    'SQL Injection': 6,
    'Infiltration': 7,
    'DoS-GoldenEye': 8,
    'DoS-Slowloris': 9,
    'DDOS-LOIC-HTTP': 10,
    'DDOS-LOIC-UDP': 11,
    'FTP-BruteForce': 12,
    'SSH-Bruteforce': 13,
    'DDOS-HOIC': 14}

SCVIC_CIDS_CLASSES_INV = {v: k for k, v in SCVIC_CIDS_CLASSES.items()}


class OpTCDataset(torch.utils.data.IterableDataset):

    def __init__(self, ds_name, parts=1, window_size=10, shuffle=False, eval_mode=False, only_benign=True, features: list[str] = None, stage:int = -1, load_all=False) -> None:
        super(OpTCDataset).__init__()
        self.ds_name = ds_name
        self.parts = parts
        self.window_size = window_size
        self.shuffle = shuffle
        self.eval_mode = eval_mode
        self.features = features
        self.stage = stage
        self.only_benign = only_benign
        self.load_all = load_all

        self.currently_loaded_idx = None
        self.partrange = np.arange(0, parts)
        if shuffle and not load_all:
            self.partorder = self.partrange[torch.randperm(len(self.partrange)).tolist()]
        else:
            self.partorder = self.partrange

        self.backup_prev = {}
        self.backup_next = {}

        if self.load_all:
            logger.info("Loading all parts of the dataset in advance")
            self.full_dataset = opil.load_preprocessed_data_full(self.ds_name, stage=self.stage)
            logger.info(f"Loaded full dataset with {len(self.full_dataset)} samples")

    def __iter__(self) -> Iterator:
        if self.load_all:
            # Handle the case where the full dataset is loaded
            currently_loaded = self.full_dataset
            length_part = len(currently_loaded)

            # Filter out start indexes that would result in invalid windows
            start_new_process = torch.arange(len(currently_loaded))[(currently_loaded['new_process'] == 1).tolist()]
            to_drop = []
            for p in start_new_process:
                for i in range(1, self.window_size):
                    if p-i < 0 or p-i > length_part - self.window_size:
                        continue
                    to_drop.append(p-i)

            idxs = torch.arange(length_part - self.window_size + 1)[torch.zeros((length_part - self.window_size + 1)).scatter_(0, torch.tensor(to_drop).squeeze_(), 1) == 0]
            del to_drop, start_new_process

            # Drop last helper columns and convert to torch tensor
            label = torch.from_numpy(currently_loaded['label'].to_numpy(dtype=int))
            currently_loaded = currently_loaded.drop(columns=['row_idx', 'new_process', 'label'])
            if self.features is not None:
                # Add missing features with all elements set to 0
                for feature in self.features:
                    if feature not in currently_loaded.columns:
                        currently_loaded[feature] = 0
                # Reorder columns to match self.features
                currently_loaded = currently_loaded[self.features]

            currently_loaded = torch.from_numpy(currently_loaded.to_numpy(dtype=np.float32))

            if self.shuffle:
                idxs = idxs[torch.randperm(len(idxs))]

            if not self.only_benign:
                if self.eval_mode:
                    for idx in idxs.tolist():
                        yield currently_loaded[idx:(idx+self.window_size), :], label[idx]
                else:
                    for idx in idxs.tolist():
                        yield currently_loaded[idx:(idx+self.window_size), :]
            else:
                if self.eval_mode:
                    for idx in idxs.tolist():
                        if label[idx] != 0:
                            continue
                        yield currently_loaded[idx:(idx+self.window_size), :], label[idx]
                else:
                    for idx in idxs.tolist():
                        if label[idx] != 0:
                            continue
                        yield currently_loaded[idx:(idx+self.window_size), :]
        else:
            # Handle the case where parts are loaded one by one
            for part in self.partorder:
                currently_loaded = opil.load_preprocessed_data_part(self.ds_name, stage=self.stage, partnum=part)

                # make sure that windows run over the partition edges. If mult workers are used the worker edges must be loaded beforehand
                if part in self.backup_prev:
                    currently_loaded = pd.concat([self.backup_prev.pop(part), currently_loaded])
                elif part != 0:
                    self.backup_next[part-1] = currently_loaded.iloc[:self.window_size-1]
                if part in self.backup_next:
                    currently_loaded = pd.concat([currently_loaded, self.backup_next.pop(part)])
                elif part != self.parts-1:
                    self.backup_prev[part+1] = currently_loaded.iloc[len(currently_loaded) - self.window_size + 1:]
                currently_loaded.reset_index()
                
                length_part = len(currently_loaded)
                # filter out start indexes that would result in windows lapping between different or would result in window end in next part
                start_new_process = torch.arange(len(currently_loaded))[(currently_loaded['new_process'] == 1).tolist()]
                to_drop = []
                for p in start_new_process:
                    for i in range(1, self.window_size):
                        if p-i < 0:
                            continue
                        if p-i > length_part - self.window_size:
                            continue
                        to_drop.append(p-i)
                
                idxs = torch.arange(length_part - self.window_size + 1)[torch.zeros((length_part - self.window_size + 1)).scatter_(0, torch.tensor(to_drop).squeeze_(), 1) == 0]
                del to_drop, start_new_process
                
                # drop last helper columns and convert to torch tensor
                label = torch.from_numpy(currently_loaded['label'].to_numpy(dtype=int))
                
                currently_loaded = currently_loaded.drop(columns=['row_idx', 'new_process', 'label'])
                if self.features is not None:
                    # Add missing features with all elements set to 0
                    for feature in self.features:
                        if feature not in currently_loaded.columns:
                            currently_loaded[feature] = 0
                    
                    # Reorder columns to match self.features
                    currently_loaded = currently_loaded[self.features]
                    
                currently_loaded = torch.from_numpy(currently_loaded.to_numpy(dtype=np.float32))
                
                # shuffle idxs
                if self.shuffle:
                    idxs = idxs[torch.randperm(len(idxs))]
                
                if not self.only_benign:
                    if self.eval_mode:
                        for idx in idxs.tolist():
                            yield currently_loaded[idx:(idx+self.window_size), :], label[idx]
                    else:
                        for idx in idxs.tolist():
                            yield currently_loaded[idx:(idx+self.window_size), :]
                else:
                    if self.eval_mode:
                        for idx in idxs.tolist():
                            if label[idx] != 0:
                                continue
                            yield currently_loaded[idx:(idx+self.window_size), :], label[idx]
                    else:
                        for idx in idxs.tolist():
                            if label[idx] != 0:
                                continue
                            yield currently_loaded[idx:(idx+self.window_size), :]



class OpTCDatasetWithIndexing(torch.utils.data.IterableDataset):
    def __init__(self, ds_name, parts: list, last_part: int, window_size=10, shuffle=False, eval_mode=False, stage=-1, only_benign=False) -> None:
        super(OpTCDataset).__init__()
        self.ds_name = ds_name
        self.parts = len(parts)
        self.last_part = last_part
        self.window_size = window_size
        self.shuffle = shuffle
        self.eval_mode = eval_mode
        self.stage = stage
        self.only_benign = only_benign

        self.currently_loaded_idx = None
        if shuffle:
            self.partorder = np.asarray(parts)[torch.randperm(self.parts).tolist()]
        else:
            self.partorder = np.asarray(parts)

        self.backup_prev = {}
        self.backup_next = {}

    def __iter__(self) -> Iterator:
        for part in self.partorder:
            currently_loaded = opil.load_preprocessed_data_part(self.ds_name, stage=self.stage, partnum=part)

            # make sure that windows run over the partition edges. If mult workers are used the worker edges must be loaded beforehand
            if part in self.backup_prev:
                currently_loaded = pd.concat([self.backup_prev.pop(part), currently_loaded])
            elif part != 0:
                self.backup_next[part-1] = currently_loaded.iloc[:self.window_size-1]
            if part in self.backup_next:
                currently_loaded = pd.concat([currently_loaded, self.backup_next.pop(part)])
            elif part != self.last_part-1:
                self.backup_prev[part+1] = currently_loaded.iloc[len(currently_loaded) - self.window_size + 1:]
            currently_loaded.reset_index()
            
            length_part = len(currently_loaded)
            # filter out start indexes that would result in windows lapping between different or would result in window end in next part
            start_new_process = torch.arange(len(currently_loaded))[(currently_loaded['new_process'] == 1).tolist()]
            to_drop = []
            for p in start_new_process:
                for i in range(1, self.window_size):
                    if p-i < 0:
                        continue
                    if p-i > length_part - self.window_size:
                        continue
                    to_drop.append(p-i)
            
            idxs = torch.arange(length_part - self.window_size + 1)[torch.zeros((length_part - self.window_size + 1)).scatter_(0, torch.tensor(to_drop).squeeze_(), 1) == 0]
            del to_drop, start_new_process
            
            # drop last helper columns and convert to torch tensor
            label = torch.from_numpy(currently_loaded['label'].to_numpy(dtype=int))
            
            currently_loaded = currently_loaded.drop(columns=['row_idx', 'new_process', 'label'])
            currently_loaded = torch.from_numpy(currently_loaded.to_numpy(dtype=np.float32))
            
            # shuffle idxs
            if self.shuffle:
                idxs = idxs[torch.randperm(len(idxs))]

            if not self.only_benign:
                if self.eval_mode:
                    for idx in idxs.tolist():
                        yield currently_loaded[idx:(idx+self.window_size), :], label[idx]
                else:
                    for idx in idxs.tolist():
                        yield currently_loaded[idx:(idx+self.window_size), :]
            else:
                if self.eval_mode:
                    for idx in idxs.tolist():
                        if label[idx] != 0:
                            continue
                        yield currently_loaded[idx:(idx+self.window_size), :], label[idx]
                else:
                    for idx in idxs.tolist():
                        if label[idx] != 0:
                            continue
                        yield currently_loaded[idx:(idx+self.window_size), :]

class OpTCDatasetUndersampling(torch.utils.data.IterableDataset):
        
    def __init__(self, ds_name, target_ratio=0.5, parts=1, window_size=10, shuffle=False, features: list[str] = None, stage:int = -1) -> None:
        # could theoretially be used as mapped dataset, but to keep compatibility, we keep it as iterable dataset
        super(OpTCDataset).__init__()
        self.ds_name = ds_name
        self.parts = parts
        self.window_size = window_size
        self.shuffle = shuffle
        self.target_ratio = target_ratio
        self.features = features
        self.stage = stage
        logger.info("Loading all parts of the dataset in advance")
        full_dataset = opil.load_preprocessed_data_full(self.ds_name, stage=self.stage)
        logger.info(f"Loaded full dataset with {len(full_dataset)} samples")

        length = len(full_dataset)

        # Filter out start indexes that would result in invalid windows
        start_new_process = torch.arange(length)[(full_dataset['new_process'] == 1).tolist()]
        to_drop = []
        for p in start_new_process:
            for i in range(1, self.window_size):
                if p-i < 0 or p-i > length - self.window_size:
                    continue
                to_drop.append(p-i)

        self.idxs = torch.arange(length - self.window_size + 1)[torch.zeros((length - self.window_size + 1)).scatter_(0, torch.tensor(to_drop).squeeze_(), 1) == 0]
        del to_drop, start_new_process
        self.label = torch.from_numpy(full_dataset['label'].to_numpy(dtype=int))[self.idxs]
        full_dataset = full_dataset.drop(columns=['row_idx', 'new_process', 'label'])

        if self.features is not None:
            # Add missing features with all elements set to 0
            for feature in self.features:
                if feature not in full_dataset.columns:
                    full_dataset[feature] = 0
            # Reorder columns to match self.features
            full_dataset = full_dataset[self.features]

        self.full_dataset = torch.from_numpy(full_dataset.to_numpy(dtype=np.float32))

    def __iter__(self) -> Iterator:
        mal_idxs = self.idxs[self.label != 0]
        n_idxs = len(mal_idxs) * ((1-self.target_ratio) / self.target_ratio)
        benign_idxs = self.idxs[self.label == 0]
        undersampled_benign_idxs = benign_idxs[torch.randperm(len(benign_idxs))[:int(n_idxs)]]
        idxs = torch.cat([mal_idxs, undersampled_benign_idxs])
        label = torch.cat([torch.ones(len(mal_idxs)), torch.zeros(len(undersampled_benign_idxs))])

        if self.shuffle:
            shuffle = torch.randperm(len(idxs))
            idxs = idxs[shuffle]
            label = label[shuffle]

        for i, idx in enumerate(idxs.tolist()):
            yield self.full_dataset[idx:(idx+self.window_size), :], label[i]
        


class SCVICCIDSDataset(torch.utils.data.Dataset):

    def __init__(self,path=None, data=None, min_samples_per_class: int = 0, exclude_classes: list = None, subset: list = None,
                 network_data=True, host_data=True, host_embeddings=True):
        super().__init__()
        if data is None and path is None:
            raise ValueError("Either data or path must be given")
        if data is not None and path is not None:
            raise ValueError("Only one of data or path must be given")
        
        self.min_samples_per_class = min_samples_per_class
        self.network_minmax = None
        self.host_minmax = None
        minmax = torch.load(os.path.join(misc.data_raw(scvic=True), "minmax.pt"), weights_only=True)
        if network_data:
            self.network_minmax = minmax["network_minmax"]
        if host_data:
            self.host_minmax = minmax["host_minmax"]

        del minmax
        self.embeddings = None
        if host_embeddings:
            logger.debug("Start loading embeddings")
            if os.path.exists(os.path.join(misc.data_raw(scvic=True), "Msg2Vec_Bert_100.h5")):
                with h5py.File(os.path.join(misc.data_raw(scvic=True), "Msg2Vec_Bert_100.h5"), 'r') as f:
                    msgs = f["logs"][:]
                    vec = f["embeddings"][:]
                msgs = [m.decode("utf-8") for m in msgs]
                vec = torch.tensor(vec)
            else:
                with open(os.path.join(misc.data_raw(scvic=True), "Msg2Vec_Bert_100.pkl"), 'rb') as f:
                    msgs, vec = pkl.load(f)
            self.embeddings = vec

        if data is not None:

            if exclude_classes is None:
                exclude_classes = []
            if min_samples_per_class > 0:
            
                occurences = {}
                for d in self.data:
                    label = d[3]
                    if label not in occurences:
                        occurences[label] = 0
                    occurences[label] += 1            
            
                for label, occ in occurences.items():
                    if occ < min_samples_per_class:
                        exclude_classes.append(label)

            self.data = []
            subset = subset if subset is not None else list(range(len(data)))
            for i in subset:
                point = data[i]
                if exclude_classes is not None and point[-1] in exclude_classes:
                    continue  
                self.data.append(point)        

        else :         
            logger.info("Loading SCVIC-CIDS dataset")
            logger.debug("Loading normal")
            
            if os.path.exists(os.path.join(path, "CidsSampleNormal_Tensor_28.h5")):
                with h5py.File(os.path.join(path, 'CidsSampleNormal_Tensor_28.h5'), 'r') as f:
                    network = f['network'][:]
                    host = f['host'][:]
                    
                    # Load the variable-length strings
                    logs =  f['logs'][:]
                    labels = f['label'][:]
                logs = [s.decode('utf-8') for s in logs]
                labels = [s.decode('utf-8') for s in labels]

                normal_data = []
                for i in range(len(network)):
                    tensor1 = torch.tensor(network[i])
                    tensor2 = torch.tensor(host[i])
                    normal_data.append((tensor1, tensor2, logs[i], labels[i]))
            else:
                with open(os.path.join(path, "CidsSampleNormal_Tensor_28.pkl"), 'rb') as f:
                    normal_data = pkl.load(f)
            logger.debug("Loading malicious")
            if os.path.exists(os.path.join(path, "CidsSampleAttack_Tensor_28.h5")):
                with h5py.File(os.path.join(path, 'CidsSampleAttack_Tensor_28.h5'), 'r') as f:
                    network = f['network'][:]
                    host = f['host'][:]
                    
                    # Load the variable-length strings
                    logs =  f['logs'][:]
                    labels = f['label'][:]
                logs = [s.decode('utf-8') for s in logs]
                labels = [s.decode('utf-8') for s in labels]

                malicious_data = []
                for i in range(len(network)):
                    tensor1 = torch.tensor(network[i])
                    tensor2 = torch.tensor(host[i])
                    malicious_data.append((tensor1, tensor2, logs[i], labels[i]))
            else:
                with open(os.path.join(path, "CidsSampleAttack_Tensor_28.pkl"), 'rb') as f:
                    malicious_data = pkl.load(f)
            logger.info("Apply preprocessing")
            if exclude_classes is None:
                exclude_classes = []
            if min_samples_per_class > 0:
                logger.debug("Load occurences")
                occurences = {}
                for d in malicious_data:
                    label = d[3]
                    if label not in occurences:
                        occurences[label] = 0
                    occurences[label] += 1
                for label, occ in occurences.items():
                    if occ < min_samples_per_class:
                        exclude_classes.append(label)
            
            data = normal_data + malicious_data
            del normal_data, malicious_data

            subset = subset if subset is not None else list(range(len(data)))
            self.classes = [c for c in SCVIC_CIDS_CLASSES.keys() if c not in exclude_classes]
            classes_map = {c: i for i, c in enumerate(self.classes)}
            self.data = []

            logger.debug(f"Go through subset of len {len(subset)} and convert msg to embedding")
            for i in subset:
                # logger.debug(f"Read idx {i}")
                point = list(data[i])
                if exclude_classes is not None and point[3] in exclude_classes:
                    continue
                if host_embeddings:
                    msg = point[2]
                    if isinstance(msg, str):
                        idx = msgs.index(msg)
                        point[2] = idx
                if isinstance(point[3], str):
                    point[3] = classes_map[point[3]]
                
                if self.network_minmax is not None:
                    point[0] = (point[0] - self.network_minmax[0]) / (self.network_minmax[1] - self.network_minmax[0] + 1e-16)
                if self.host_minmax is not None:
                    point[1] = (point[1] - self.host_minmax[0]) / (self.host_minmax[1] - self.host_minmax[0] + 1e-16)

                if not network_data:
                    point.pop(0)
                if not host_data:
                    point.pop(-3)
                if not host_embeddings:
                    point.pop(-2)
                self.data.append(tuple(point))
        logger.info("Done intializing dataset")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = list(self.data[idx])
        if self.embeddings is not None:
            data[-2] = self.embeddings[data[-2]].squeeze()
        return tuple(data)    
    

class SCVICCIDSDatasetBinary(SCVICCIDSDataset): 

    def __init__(self,path=None, data=None, min_samples_per_class: int = 0, exclude_classes: list = None, subset: list = None, only_benign: bool = False,
                 network_data=True, host_data=True, host_embeddings=True):
        
        if only_benign:
            if exclude_classes is not None:
                logger.warning("only_benign is set. Therefore exclude classes is overwritten.")
            exclude_classes = list(SCVIC_CIDS_CLASSES.keys())
            exclude_classes.remove("Benign")
        
        super().__init__(path=path, data=data, min_samples_per_class=min_samples_per_class,
                         exclude_classes=exclude_classes, subset=subset,
                         network_data=network_data, host_data=host_data, host_embeddings=host_embeddings)

    def __getitem__(self, idx):
        data = list(self.data[idx])
        
        data[-1] = 0 if data[-1] == 0 else 1
        if self.embeddings is not None:
            data[-2] = self.embeddings[data[-2]].squeeze()
        return tuple(data) 
    

class Undersampling(torch.utils.data.IterableDataset):
    """
    Undersampling using rejection sampling. Code for online-version extracted from https://github.com/MaxHalford/pytorch-resample/blob/master/pytorch_resample/under.py
    """

    def __init__(self, dataset: torch.utils.data.IterableDataset, target_dist: dict, seed: int = None, online=True):
        super().__init__()
        self.dataset = dataset
        self.target_dist = target_dist
        self.seed = seed
        self.online = online

        self.actual_dist = {k: 0 for k in target_dist.keys()}
        if not online:
            self.ratio = {k: 0 for k in target_dist.keys()}
            total = 0
            for _, y in dataset:
                self.actual_dist[y.item()] += 1
                total += 1
            for k in self.actual_dist.keys():
                self.actual_dist[k] = self.actual_dist[k] / total 
            self.raised = False
            try:
                M = max(self.target_dist[k] / self.actual_dist[k] for k in self.target_dist.keys())
            except ZeroDivisionError as e:
                logger.warning("No Malicious samples in dataset")
                self.raised = True
            if not self.raised:
                self.ratio = {k: self.target_dist[k] / (M * self.actual_dist[k]) for k in self.target_dist.keys()}

        self._pivot = None
        

    def __iter__(self):
        # To ease notation
        f = self.target_dist
        g = self.actual_dist

        if not self.online:
            if self.raised:
                for x, y in self.dataset:
                    yield x, y
            else:
                for x, y_tensor in self.dataset:
                    ratio = self.ratio[int(y_tensor)]
                    if ratio <= 1 and random.random() < ratio:
                        yield x, y_tensor

        else:
            for x, y_tensor in self.dataset:
                y = int(y_tensor)

                self.actual_dist[y] += 1

                # Check if the pivot needs to be changed

                if y != self._pivot:
                    self._pivot = max(g.keys(), key=lambda y: f[y] / g[y])
                else:
                    yield x, y_tensor
                    continue

                # Determine the sampling ratio if the observed label is not the pivot
                M = f[self._pivot] / g[self._pivot]
                ratio = f[y] / (M * g[y])

                if ratio <= 1 and random.random() < ratio:
                    yield x, y_tensor

    @staticmethod
    def expected_size(n, desired_dist, actual_dist):
        M = max(
            desired_dist.get(k) / actual_dist.get(k)
            for k in set(desired_dist) | set(actual_dist)
        )
        return int(n / M)


def worker_init(worker_id):
    """
    If we want to initialize the dataset using a dataloader and multiple workers, each worker will get a specific set of parts to generate the data from.
    Therefore we already need to load the data and the boarders between workers in advance. Note this function only works if the total number of parts is divisble by 
    the number of workers 
    """
    logger.debug(f"start INIT worker {worker_id}")
    worker_info = torch.utils.data.get_worker_info()

    num_parts = worker_info.dataset.parts // worker_info.num_workers

    worker_info.dataset.partrange = np.arange(num_parts * worker_id, num_parts * worker_id + num_parts)
    if worker_info.dataset.shuffle:
        worker_info.dataset.partorder = worker_info.dataset.partrange[torch.randperm(len(worker_info.dataset.partrange)).tolist()]
    else:
        worker_info.dataset.partorder = worker_info.dataset.partrange

    load_part = num_parts * worker_id - 1
    if not load_part < 0:
        loaded = opil.load_preprocessed_data_part(worker_info.dataset.ds_name, stage=-1, partnum=load_part)
        worker_info.dataset.backup_prev[load_part + 1] = loaded.iloc[len(loaded) - worker_info.dataset.window_size + 1:]
    
    
    logger.debug(f"Initalized dataset for worker {worker_id}. Will load parts {worker_info.dataset.partrange} in order {worker_info.dataset.partorder}")

def worker_init_wrapped(worker_id):
    """
    If we want to initialize the dataset using a dataloader and multiple workers, each worker will get a specific set of parts to generate the data from.
    Therefore we already need to load the data and the boarders between workers in advance. Note this function only works if the total number of parts is divisble by 
    the number of workers 
    """
    logger.debug(f"start INIT worker {worker_id}")
    worker_info = torch.utils.data.get_worker_info()

    num_parts = worker_info.dataset.dataset.parts // worker_info.num_workers

    worker_info.dataset.dataset.partrange = np.arange(num_parts * worker_id, num_parts * worker_id + num_parts)
    if worker_info.dataset.dataset.shuffle:
        worker_info.dataset.dataset.partorder = worker_info.dataset.dataset.partrange[torch.randperm(len(worker_info.dataset.dataset.partrange)).tolist()]
    else:
        worker_info.dataset.dataset.partrange = worker_info.dataset.dataset.partrange

    load_part = num_parts * worker_id - 1
    if not load_part < 0:
        loaded = opil.load_preprocessed_data_part(worker_info.dataset.dataset.ds_name, stage=-1, partnum=load_part)
        worker_info.dataset.dataset.backup_prev[load_part + 1] = loaded.iloc[len(loaded) - worker_info.dataset.dataset.window_size + 1:]
    
    
    logger.debug(f"Initalized dataset for worker {worker_id}. Will load parts {worker_info.dataset.dataset.partrange} in order {worker_info.dataset.dataset.partorder}")

def get_SCVIC_dataloader(batch_size, num_workers, exclude=None, validation=True, test=False, random_state=42, network_data=True, host_data=True, host_embeddings=True):
    """
    Get the SCVIC dataloader. If exclude is given, the classes in the exclude list will be removed from the dataset. If validation is True, the validation set will be returned additionaly
    the train set. The split between train and validation is 80/20 and randomly generated using the random_state If test is true the test set will be returned.
    """
    if test:
        if validation:
            raise ValueError("Can't return validation and test set at the same time")
        logger.info("Load test loader")
        idxs = np.loadtxt(os.path.join(misc.data(), "scvic/test_indices.txt"), dtype=int)
        dataset = SCVICCIDSDataset(misc.data_raw(scvic=True), exclude_classes=exclude, subset=idxs, network_data=network_data, host_data=host_data, host_embeddings=host_embeddings)
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
    
    logger.info(f"load train_dataloader based on {os.path.join(misc.data(), "scvic/train_indices.txt")}")
    idxs = np.loadtxt(os.path.join(misc.data(), "scvic/train_indices.txt"), dtype=int)
    dataset = SCVICCIDSDataset(misc.data_raw(scvic=True), exclude_classes=exclude, subset=idxs, network_data=network_data, host_data=host_data, host_embeddings=host_embeddings)
    
    if not validation:
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    
    # split into train and validation
    logger.info("Split train into train/val split using 0.8/0.2 split")
    train_idxs, val_idxs = next(StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=random_state).split(np.zeros(len(dataset)), [d[-1] for d in dataset]))
    train_dataset = SCVICCIDSDataset(data=dataset.data, subset=train_idxs, network_data=network_data, host_data=host_data, host_embeddings=host_embeddings)
    val_dataset = SCVICCIDSDataset(data=dataset.data, subset=val_idxs, network_data=network_data, host_data=host_data, host_embeddings=host_embeddings)
    return torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True), torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers)

def get_SCVIC_dataloader_binary(batch_size, num_workers, exclude=None, validation=True, test=False, random_state=42, network_data=True, host_data=True, host_embeddings=True):
    """
    Get the SCVIC dataloader. If exclude is given, the classes in the exclude list will be removed from the dataset. If validation is True, the validation set will be returned additionaly
    the train set. The split between train and validation is 80/20 and randomly generated using the random_state If test is true the test set will be returned.
    """
    if test:
        if validation:
            raise ValueError("Can't return validation and test set at the same time")
        logger.info("Load test loader")
        idxs = np.loadtxt(os.path.join(misc.data(), "scvic/test_indices.txt"), dtype=int)
        dataset = SCVICCIDSDatasetBinary(misc.data_raw(scvic=True), exclude_classes=exclude, subset=idxs, network_data=network_data, host_data=host_data, host_embeddings=host_embeddings)
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
    
    logger.info(f"load train_dataloader based on {os.path.join(misc.data(), "scvic/train_indices.txt")}")
    idxs = np.loadtxt(os.path.join(misc.data(), "scvic/train_indices.txt"), dtype=int)
    dataset = SCVICCIDSDataset(misc.data_raw(scvic=True), exclude_classes=exclude, subset=idxs, network_data=network_data, host_data=host_data, host_embeddings=host_embeddings)
    
    if not validation:
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    
    # split into train and validation
    logger.info("Split train into train/val split using 0.8/0.2 split")
    train_idxs, val_idxs = next(StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=random_state).split(np.zeros(len(dataset)), [d[-1] for d in dataset]))
    train_dataset = SCVICCIDSDatasetBinary(data=dataset.data, subset=train_idxs, network_data=network_data, host_data=host_data, host_embeddings=host_embeddings)
    val_dataset = SCVICCIDSDatasetBinary(data=dataset.data, subset=val_idxs, network_data=network_data, host_data=host_data, host_embeddings=host_embeddings)
    return torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True), torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers)

def get_SCVIC_Dataset(exclude=None, train=True, validation=True, test=False, random_state=42, network_data=True, host_data=True, host_embeddings=True):

    dss = {}
    if train:
        idxs = np.loadtxt(os.path.join(misc.data(), "scvic/train_indices.txt"), dtype=int)
        dataset = SCVICCIDSDataset(misc.data_raw(scvic=True), exclude_classes=exclude, subset=idxs, network_data=network_data, host_data=host_data, host_embeddings=host_embeddings)

        if validation:
            train_idxs, val_idxs = next(StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=random_state).split(np.zeros(len(dataset)), [d[-1] for d in dataset]))
            train_dataset = SCVICCIDSDataset(data=dataset.data, subset=train_idxs, network_data=network_data, host_data=host_data, host_embeddings=host_embeddings)
            val_dataset = SCVICCIDSDataset(data=dataset.data, subset=val_idxs, network_data=network_data, host_data=host_data, host_embeddings=host_embeddings)

            dss["train"] = train_dataset
            dss["val"] = val_dataset
        else:
            dss["train"] = dataset
    if test:
        idxs = np.loadtxt(os.path.join(misc.data(), "scvic/test_indices.txt"), dtype=int)
        dataset = SCVICCIDSDataset(misc.data_raw(scvic=True), exclude_classes=exclude, subset=idxs, network_data=network_data, host_data=host_data, host_embeddings=host_embeddings)
        dss["test"] =dataset

    return dss

def get_SCVIC_DatasetBinary(exclude=None, train=True, validation=True, test=False, random_state=42, network_data=True, host_data=True, host_embeddings=True):

    dss = {}
    if train:
        idxs = np.loadtxt(os.path.join(misc.data(), "scvic/train_indices.txt"), dtype=int)
        dataset = SCVICCIDSDatasetBinary(path=misc.data_raw(scvic=True), exclude_classes=exclude, subset=idxs, network_data=network_data, host_data=host_data, host_embeddings=host_embeddings)

        if validation:
            train_idxs, val_idxs = next(StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=random_state).split(np.zeros(len(dataset)), [d[-1] for d in dataset]))
            train_dataset = SCVICCIDSDatasetBinary(data=dataset.data, subset=train_idxs, network_data=network_data, host_data=host_data, host_embeddings=host_embeddings)
            val_dataset = SCVICCIDSDatasetBinary(data=dataset.data, subset=val_idxs, network_data=network_data, host_data=host_data, host_embeddings=host_embeddings)

            dss["train"] = train_dataset
            dss["val"] = val_dataset
        else:
            dss["train"] = dataset
    if test:
        idxs = np.loadtxt(os.path.join(misc.data(), "scvic/test_indices.txt"), dtype=int)
        dataset = SCVICCIDSDatasetBinary(path=misc.data_raw(scvic=True), exclude_classes=exclude, subset=idxs, network_data=network_data, host_data=host_data, host_embeddings=host_embeddings)
        dss["test"] =dataset

    return dss
