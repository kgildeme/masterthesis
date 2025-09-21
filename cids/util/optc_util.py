"""
Useful functions for handling data from the OpTC dataset
Most functions copied from CIDS-Implementation by Lars Janssen
"""

import gzip
import json
import pandas as pd
import time
import pickle
import os
import glob
import logging
import matplotlib as mpl
from datetime import datetime
from . import misc_funcs as misc
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)
start_time = time.time()
overall_start_time = start_time

lookup_list = ['SERVICE-CREATE', 'HOST-START', 'FILE-CREATE', 'FLOW-START', 'THREAD-TERMINATE', 'PROCESS-OPEN', 'THREAD-CREATE', 'FILE-MODIFY', 'FLOW-MESSAGE',\
    'THREAD-REMOTE_CREATE', 'PROCESS-CREATE', 'PROCESS-TERMINATE', 'MODULE-LOAD', 'FILE-WRITE', 'FILE-READ', 'FILE-RENAME', 'REGISTRY-EDIT', 'FLOW-OPEN', \
    'REGISTRY-ADD', 'REGISTRY-REMOVE', 'FILE-DELETE', 'TASK-MODIFY', 'USER_SESSION-LOGIN', 'USER_SESSION-GRANT', 'TASK-START', 'TASK-DELETE', 'TASK-CREATE',\
    'USER_SESSION-REMOTE', 'USER_SESSION-LOGOUT', 'USER_SESSION-INTERACTIVE', 'USER_SESSION-UNLOCK', 'SHELL-COMMAND']

lookup_list.sort()
lookup_table = {k: i for i, k in enumerate(lookup_list)}

top_ecar_features = ["image_path_filename", "parent_image_path_filename", "info_class", "command_line_filename", "module_path_filename"]
top_bro_features = ["server_name", "host", "subject", "certificate.subject", "referrer.domain", "method", "curve", "cipher", "version", 
                    "resp_mime_types", "mime_type", "conn_state", "analyzers", "status_msg", "service", "status_code", "history"]
top_feature_amount = 5

LAST_STAGE = 4

blue_cmap = mpl.colors.LinearSegmentedColormap.from_list("custom", [(1, 1, 1), (0.2, 0.2, 1)], N=200)
orange_cmap = mpl.colors.LinearSegmentedColormap.from_list("custom", [(1, 1, 1), (1, 0.5, 0.1)], N=200)

train_huge_num = 10
eval_huge_num = 5

train_host_num = 10
eval_host_num = 10

all_hosts = [50 * i + j for i in range(1,20) for j in range(1, 26)]

def get_host_ips(hostnum, throw=True):
    ips = []
    files = [os.path.join(misc.root(), "preprocessing/labeling/ip_lookup_142.txt"), os.path.join(misc.root(), "preprocessing/labeling/ip_lookup_10.txt")]
    for filename in files:
        with open(filename, "r") as f:
            hosts = {line.split(" ")[0]: line.split(" ")[1] for line in f.readlines()}
        key = f"SysClient{hostnum:04}.systemia.com"
        if key in hosts:
            ips.append(hosts[key].strip())
        else:
            if throw:
                raise Exception(f"Error: Host {hostnum} not found")
            print(f"Error: Host {hostnum} not found")
    return ips

def ip_to_host_num(ip):
    if ip[:7] == "142.20.":
        parts = ip[7:].split(".")
        host =  256 * (int(parts[0]) - 56) + int(parts[1]) - 1
        if host >= 0 and host <= 1000:
            return host
    return -1

def host_num_to_aia(hostnum):
    return f"AIA-{25*int(int(hostnum) / 25)+1}-{25*int(int(hostnum) / 25)+25}"

def convert_timestamp(ts):
    ts = ts.replace("-04:00", "-0400")
    if "." in ts:
        dt = datetime.strptime(ts, "%Y-%m-%dT%H:%M:%S.%f%z")
    else:
        dt = datetime.strptime(ts, "%Y-%m-%dT%H:%M:%S%z")
    return dt.timestamp()

def extract_json_attribute(string, key, nostring=False):
    # Extract key by hand in contrast to using the json library to save time
    mask = f"\"{key}\":\""
    if nostring:
        mask = mask[:-1]
    if not mask in string:
        return None
    value = string[string.index(mask) + len(mask):]
    if nostring:
        value = value[:value.index(",")]
    else:
        value = value[:value.index("\"")]
    return value

def determine_top_feature_value_counts(df, percentage=0.02):
    top_features = [feature for feature in df.columns if "top_" in feature]

    out_features = {}
    total_num_features = 0
    for feature in top_features:
        val_counts = df[feature].value_counts()
        feature_len = len(df[~df[feature].isnull()])
        total_len = len(df)
        cum_len = 0
        num_features = 0
        top_values = []
        for i in range(len(val_counts)):
            cum_len += val_counts.iloc[i]
            num_features += 1
            total_num_features += 1
            top_values.append(val_counts.index[i])
            if val_counts.iloc[i] < total_len * percentage:
                break
        logger.info(f"{feature}: {num_features} / {len(val_counts)} ({feature_len})")
        #top_values = top_features_df[feature].value_counts()[:opil.get_top_feature_amount()].index.tolist()
        for i, value in enumerate(top_values):
            out_features[f"{feature}_top_{i+1}"] = value
    logger.info(f"Total num features: {total_num_features}")

    print(json.dumps(out_features, indent=2))

    return out_features

def get_lookup_table():
    return lookup_table

def timing(label):
    global start_time
    logger.info(f"- {time.time() - start_time:5.2f}s - {label} -")
    start_time = time.time()

def timing_overall():
    print(f"- {time.time() - overall_start_time:05.2f}s - TOTAL -", flush=True)

def mal_processes(host, dict=False):
    mal_ids = {} if dict else []
    mal_file = os.path.join(misc.root(), f"preprocessing/labeling/malicious_processes/malicious_processes_{host}")
    if not os.path.exists(mal_file):
        raise Exception(f"Warning: This host ({host}) has no malicious processes. Is this intended? ({mal_file})")
    with open(mal_file, "r") as f:
        lines = f.readlines()
        for line in lines:
            if len(line.strip()) > 0:
                if dict:
                    mal_ids[line.strip()] = True
                else:
                    mal_ids.append(line.strip())
    return mal_ids

def objaction_lookup(value):
    if type(value) == int:
        return lookup_list[value]
    else:
        return lookup_table[value]

def get_top_ecar_features():
    return top_ecar_features

def get_top_bro_features():
    return top_bro_features

def get_top_feature_amount():
    return top_feature_amount


def save_pkl(filename, elem):
    if not filename.endswith(".pkl"):
        filename += ".pkl"
    if not filename.startswith("/"):
        raise Exception(f"Error: Filename {filename} is relative. Is this intended?")
    with open(f"{filename}", "wb") as f:
        pickle.dump(elem, f)

def load_pkl(filename):
    if not filename.endswith(".pkl"):
        filename += ".pkl"
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Error: File {filename} not found")
    with open(f"{filename}", "rb") as f:
        elem = pickle.load(f)
    return elem

def load_ds_metadata(ds_name, stage):
    stage = LAST_STAGE if stage == -1 else stage
    parts = glob.glob(os.path.join(misc.root(), f"data/preprocessed/stage{stage}/{ds_name}/*.parquet"))
    if len(parts) < 1:
        print(f"Warning: No data found for {ds_name} in stage {stage}")
        return None

    return pq.read_metadata(parts[0])

def get_preprocessed_data_num_parts(ds_name, stage):
    stage = LAST_STAGE if stage == -1 else stage
    parts = glob.glob(os.path.join(misc.root(), f"data/preprocessed/stage{stage}/{ds_name}/*.parquet"))
    return len(parts)

def load_preprocessed_data_full(ds_name, stage, silent=False):
    stage = LAST_STAGE if stage == -1 else stage
    parts = glob.glob(os.path.join(misc.data(), f"preprocessed/stage{stage}/{ds_name}/*.parquet"))
    if len(parts) < 1:
        raise RuntimeError(f"Error: No data found for {ds_name} in stage {stage}")
    
    parts.sort(key=lambda x: int(x.split(os.path.sep)[-1].split(".parquet")[0].split("part_")[1]))
    if not silent:
        logger.info(f"Load following parts: {parts}")

    sub_df_list = []
    for file in parts:        
        sub_df_list.append(pd.read_parquet(file, engine="pyarrow"))
        if not silent:
            part_num = int(file.split(os.path.sep)[-1].split(".parquet")[0].split("part_")[1])
            logger.info(f"Loaded part {part_num}")
    if len(sub_df_list) == 1:
        return sub_df_list[0]
    elif len(sub_df_list) == 0:
        raise ValueError(f"Weird Error: No data found for {ds_name} in stage {stage}")
    return pd.concat(sub_df_list)

def load_preprocessed_data_part(ds_name, stage, partnum):
    stage = LAST_STAGE if stage == -1 else stage
    part = os.path.join(misc.root(), f"data/preprocessed/stage{stage}/{ds_name}/part_{partnum}.parquet")
    logger.debug(f"loaded part: {part}")
    if not os.path.exists(part):
        parts = glob.glob(os.path.join(misc.root(), f"data/preprocessed/stage{stage}/{ds_name}/*.parquet"))
        parts.sort()
        if len(parts) < 1:
            raise Exception(f"Error: No data found for {ds_name} in stage {stage}")
        raise Exception(f"Error: Part {partnum} not found for {ds_name} in stage {stage}. Found {len(parts)} parts")
    return pd.read_parquet(part, engine="pyarrow")

"""
class DataLoader(object):
    def __init__(self, ds_name, stage):
        self._ds_name = ds_name
        self._stage = LAST_STAGE if stage == -1 else stage
        self._length = len(glob.glob(os.path.join(misc.root(), f"data/preprocessed/stage{self._stage}/{ds_name}/*.parquet")))

    def __iter__(self):
        filenames = glob.glob(os.path.join(misc.root(), f"data/preprocessed/stage{self._stage}/{self._ds_name}/*.parquet"))
        filenames.sort(key=lambda x: int(x.split(os.path.sep)[-1].split(".parquet")[0].split("part_")[1]))
        for filename in filenames:
            data = pd.read_parquet(filename, engine="pyarrow")
            yield data, int(filename.split(os.path.sep)[-1].split(".parquet")[0].split("part_")[1])

    def __len__(self):
        return self._length
"""
def remove_preprocessed_data(ds_name, stage):
    stage = LAST_STAGE if stage == -1 else stage
    filenames = glob.glob(os.path.join(misc.root(), f"data/preprocessed/stage{stage}/{ds_name}/*.parquet"))
    for filename in filenames:
        os.remove(filename)

def save_preprocessed_data(data, ds_name, stage, partnum=0, rows_per_part=0, parts=0, remove_old_data=True):
    stage = LAST_STAGE if stage == -1 else stage
    assert not (rows_per_part > 0 and parts > 0), "Cannot set rows_per_part and parts"

    out_dir = os.path.join(misc.data(), f"preprocessed/stage{stage}/{ds_name}")
    if remove_old_data:
        remove_preprocessed_data(ds_name, stage)
    if isinstance(data, pd.Series):
        data = pd.DataFrame(data)
    os.makedirs(out_dir, exist_ok=True)
    if rows_per_part == 0 and parts == 0:
        data.to_parquet(os.path.join(out_dir, f"part_{partnum}.parquet"), compression="zstd", engine="pyarrow")
        logger.info(f"Written to {out_dir}/part_{partnum}.parquet")
    else:
        if "actorID" in data.columns:
            raise Exception("Warning: Don't use this for the last stage of preprocessing, as the order will get changed during this split. " + 
                                "Either remove the whole actorID split, or remove this warning if safe.")
        else:
            logger.warning("Warning: No actorID column found. Using simple split")
            # Simple split
            if parts != 0:
                for i in range(0, len(data), len(data) // parts):
                    if ((i+ (len(data) // parts)) < len(data)):
                        data_part = data.iloc[i:i+(len(data) // parts)]
                    else:
                        data_part = data.iloc[i:]
                    
                    logger.debug(data_part.head().to_json())
                    data_part.to_parquet(os.path.join(out_dir, f"part_{partnum}.parquet"), compression="zstd", engine="pyarrow")
                    partnum += 1
            else:
                for i in range(0, len(data), rows_per_part):
                    data_part = data.iloc[i:i+rows_per_part]
                    data_part.to_parquet(os.path.join(out_dir, f"part_{partnum}.parquet"), compression="zstd", engine="pyarrow")
                    partnum += 1

def feature_file_exists(ds_name, filename, stage):
    stage = LAST_STAGE if stage == -1 else stage
    in_dir = os.path.join(misc.root(), f"data/preprocessed/stage{stage}/{ds_name}")
    return os.path.exists(os.path.join(in_dir, filename + ".pkl"))

def load_features_file(ds_name, filename, stage):
    stage = LAST_STAGE if stage == -1 else stage
    in_dir = os.path.join(misc.root(), f"data/preprocessed/stage{stage}/{ds_name}")
    logger.info(f"loading features file from: {os.path.join(in_dir, filename)}")
    return load_pkl(os.path.join(in_dir, filename))

def save_features_file(data, ds_name, filename, stage):
    stage = LAST_STAGE if stage == -1 else stage
    out_dir = os.path.join(misc.data(), f"preprocessed/stage{stage}/{ds_name}")
    os.makedirs(out_dir, exist_ok=True)
    save_pkl(os.path.join(out_dir, filename), data)

def str_to_bool(s):
    if s == "True":
        return True
    elif s == "False":
        return False
    raise Exception(f"Unknown bool value: {s}")

def open_gz(filename, mode):
    if filename.endswith(".gz"):
        return gzip.open(filename, mode)
    else:
        return open(filename, mode)

def intersection(x, y_1, y_2):
    distances = [abs(val_1 - val_2) for val_1, val_2 in zip(y_1, y_2)]
    t = x[distances.index(min(distances))]
    t_y = y_1[distances.index(min(distances))]
    return t, t_y

precision_recall_error_printed = False
def calc_metrics(tp, fp, fn, tn):
    global precision_recall_error_printed
    if tp + fp == 0:
        precision = 1
    else:
        precision = tp / (tp + fp)
    if tp + fn == 0:
        recall = 1
    else:
        recall = tp / (tp + fn)
    if tn + fp == 0:
        tnr = 1
    else:
        tnr = tn / (tn + fp)
    if precision + recall == 0:
        if not precision_recall_error_printed:
            print(f"Warning: Precision + recall = 0")
            precision_recall_error_printed = True
        f1 = 0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)
    return f1, precision, recall, tnr

def plot_folder():
    return os.path.join(misc.root(), "plots")