"""
Load ecar-bro data into dataframe
Author: Lars Janssen
"""
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parents[1]))

import json
import gzip
import glob
import os
import argparse
import time
import logging
from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

import pandas as pd
import numpy as np

from cids.util import optc_util as opil
from cids.util import misc_funcs as misc

STAGE = 1
pd.set_option('display.precision', 12)
logging.basicConfig(level=logging.INFO)
## Process raw eCAR-bro mapping data to a single dataframe

parser = argparse.ArgumentParser()
parser.add_argument("ds_name", help="Name of the dataset (e.g. ecarbro_201_complete) or comma-separated list", type=str)
parser.add_argument("--remove", help="Remove old dataset files", action="store_true")
args = parser.parse_args()

cols = ["timestamp", "id", "hostname", "objectID", "object", "action", "actorID", "pid", "ppid", "tid", "principal", ]
properties_cols = ["acuity_level", "bro_uid", "dest_ip", "dest_port", "direction", "image_path", "l4protocol", "src_ip", "src_port"]

type_folder_lookup = {"train": "benign", "eval": "evaluation", "complete": "*"}

def label(df, host):
    df.loc[:, "label"] = np.where(df["actorID"].isin(opil.mal_processes(host)), 1, 0)
    return df

def save_json_as_df(filename, host, ds_name, use_all_objects=False):
    hoststring = f'SysClient{int(host):04d}.systemia.com'
    events = []
    _file_size = os.path.getsize(filename)
    with open(filename, "rb") as cf, gzip.GzipFile(fileobj=cf) as f, \
        tqdm(total=_file_size, unit='B', unit_scale=True, position=1, leave=False) as pbar, logging_redirect_tqdm():
        ctr = 0
        succ = 0
        hostname_mask = f'"hostname":"'
        for line in f:
            line = line.decode("utf-8")
            pbar.update(cf.tell() - pbar.n)
            if ctr % int(1e7) == 0 and ctr > 0:
                logging.info(f"Processed {ctr} lines, {succ:,} extracted, {int(time.time() - opil.overall_start_time)}s elapsed")
            ctr += 1

            host_value = line[line.index(hostname_mask) + len(hostname_mask):]
            host_value = host_value[:host_value.index("\"")]
            if not host_value == hoststring:
                continue
            if not use_all_objects: # Only keep Process, File, Module, Shell and Flow events
                if not opil.extract_json_attribute(line, "object") in ["PROCESS", "FILE", "FLOW", "MODULE", "SHELL"]:
                    continue
            doc = json.loads(line)
            lst = [doc[col] for col in cols]
            for key in doc.keys():
                if key not in cols + ['properties']:
                    raise Exception(f"Found new key: {key}")
            properties = doc['properties']
            for col in properties.keys():
                if not col in properties_cols:
                    raise Exception(f"Found new property: {col}")
            for col in properties_cols:
                lst.append(properties[col] if col in properties else None)
                           
            if len(lst) != len(cols) + len(properties_cols):
                logging.critical(f"Length mismatch {len(lst)} != {len(cols) + len(properties_cols)}")
                exit()

            events.append(lst)
            succ += 1
    return events

for ds_name in args.ds_name.split(","):
    if args.remove:
        logging.info("Removing old data...")
        opil.remove_preprocessed_data(ds_name, STAGE)
    
    host = ds_name.split("_")[1]
    aia_folder = opil.host_num_to_aia(host)

    ds_type = ds_name.split("_")[2]
    if ds_type != "complete" or ds_type not in type_folder_lookup:
        raise ValueError(f"Only complete ecar-bro datasets are supported, not {ds_type}")
    
    type_folder = type_folder_lookup[ds_type]
    
    files_mask = os.path.join(misc.data_raw(), "ecar-bro", type_folder, "*", aia_folder, "*.json.gz")
    files = glob.glob(files_mask)
    files.sort()
    logging.info(f"Found {len(files)} files for ds {ds_name}")

    os.makedirs(os.path.join(misc.root(), f"data/preprocessed/stage{STAGE}", ds_name), exist_ok=True)
    events = []
    with logging_redirect_tqdm():
        for i, filename in enumerate(tqdm(files, desc="Processing files:", total=len(files), position=0)):
            logging.info(f"Processing ({filename})")
            events += save_json_as_df(filename, host, ds_name, use_all_objects="-ao" in ds_name)
    
    # Convert to dataframe
    df = pd.DataFrame(data=events, columns=cols+properties_cols)
    # Label
    df = label(df, host)
    # Save
    opil.save_preprocessed_data(df, ds_name, STAGE, remove_old_data=False)
opil.timing_overall()