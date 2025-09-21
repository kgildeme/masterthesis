from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parents[1]))

import argparse
import json
import logging
import os

from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np

from cids.util import optc_util as opil
from cids.util import misc_funcs as misc
import util

STAGE = 5

def generate_split_stratified(labels: list, new_process: list):
    # First, build mapping from process id -> list of event indices 
    # and record a label for each process.
    process_id_map = {}
    process_label = []
    pid = -1
    
    for i, p in enumerate(new_process):
        if p == 1:
            pid += 1 
            process_id_map[pid] = [i] 
            process_label.append(labels[i]) 
        else: 
            process_id_map[pid].append(i)

    # Group processes by label
    label_to_pids = {}
    for pid, lab in enumerate(process_label):
        label_to_pids.setdefault(lab, []).append(pid)
    logger.info(f"Label Groups: {list(label_to_pids.keys())}")
    # For reproducible shuffling
    np.random.seed(42)

    train_idxs = []
    val_idxs = []

    # For each label group, assign processes so that about 80% of the events go to training.
    for lab, pids in label_to_pids.items():
        np.random.shuffle(pids)
        
        # Calculate total events for this label group
        total_events = sum(len(process_id_map[pid]) for pid in pids)
        logger.info(f"found {total_events} events for label {lab}")
        train_target = 0.8 * total_events
        train_events = 0
        
        # Loop over the processes in this label group
        for pid in pids:
            process_events = len(process_id_map[pid])
            if train_events < train_target:
                train_idxs.extend(process_id_map[pid])
                train_events += process_events
            else:
                val_idxs.extend(process_id_map[pid])

    labels = np.asarray(labels)
    logger.info(f"Generated split is: {len(train_idxs) / len(labels)} / {len(val_idxs) / len(labels)}. \n With the number split of malicious samples beeing: {sum(labels[train_idxs])} / {sum(labels[val_idxs])}")

    return train_idxs, val_idxs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("ds_name", help="Name of the dataset (e.g. hids-v5_201_train) or comma-separated list", type=str)
    parser.add_argument("--remove", help="Remove old dataset files", action="store_true")
    args = parser.parse_args()

    logger = util.setup_logger(os.path.join(misc.root(), f"logs/preprocessing/{'-'.join(args.ds_name.split(','))}-STAGE{STAGE}.log"), level=logging.INFO)

    for ds_name in args.ds_name.split(','):
        if args.remove:
            try:
                logger.info("Removing old data")
                opil.remove_preprocessed_data(ds_name, STAGE)
            except Exception as e:
                logger.exception("An unexpected error occured, while deleting old datasets")
                raise e
            
        try:
            df = opil.load_preprocessed_data_full(ds_name, STAGE - 1)
            meta = opil.load_preprocessed_data_full(ds_name + "_meta", STAGE - 1)
            opil.timing("Loaded")
        except Exception as e:
            logger.exception("An unexpected error occured while loading previous stage")
            raise e
        
        logger.debug(f"Loaded df contains the following columns: {df.columns}")
        random_state = 42
        logger.debug(f"Splitting dataset using random state {random_state}")

        # Generate train and validation indices using the updated generate_split function
        labels = df["label"].tolist()
        new_process = df["new_process"].tolist()
        train_idxs, val_idxs = generate_split_stratified(labels, new_process)

        df_train = df.iloc[train_idxs]
        df_val = df.iloc[val_idxs]

        meta_train = meta.iloc[train_idxs]
        meta_val = meta.iloc[val_idxs]

        opil.timing("Split dataset")

        opil.save_preprocessed_data(df_train, ds_name + "_train", stage=STAGE, parts=8)
        opil.save_preprocessed_data(meta_train, ds_name + "_train_meta", stage=STAGE)

        opil.save_preprocessed_data(df_val, ds_name + "_eval", stage=STAGE, parts=8)
        opil.save_preprocessed_data(meta_val, ds_name + "_eval_meta", stage=STAGE)

        opil.timing("Saving")

    opil.timing_overall()
