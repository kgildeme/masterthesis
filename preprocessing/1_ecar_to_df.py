"""
Load ecar data into dataframe
Author: Lars Janssen
"""
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parents[1]))

import json
import gzip
import glob
import gc
import os
import argparse
import time
import logging
from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

import pandas as pd

from cids.util import optc_util as opil
from cids.util import misc_funcs as misc

STAGE = 1
pd.set_option('display.precision', 12)
logging.basicConfig(level=logging.INFO)
 ## Process raw eCAR data to a single dataframe 
  
parser = argparse.ArgumentParser()
parser.add_argument("ds_name", help="Name of the dataset (e.g. hids_201_train) or comma-separated list", type=str)
parser.add_argument("--remove", help="Remove old dataset files", action="store_true")
args = parser.parse_args()

cols = ['object', 'action', 'actorID', 'objectID', 'principal', 'tid', 'timestamp', 'id']
unused_cols = ['hostname', 'pid', 'ppid'] # id
properties_cols = ['acuity_level', 'image_path', 'dest_ip', 'dest_port', 'direction', 'l4protocol', 'src_ip', 'src_port', 'end_time',\
                    'start_time', 'size', 'info_class', 'file_path', 'new_path', 'sid', 'user', 'command_line', 'parent_image_path', \
                    'module_path', 'base_address', 'context_info', 'payload']
type_folder_lookup = {"train": "benign", "eval": "evaluation", "complete": "*"}
minimal_events = 10

events_per_actor = {}
written_actors = []

buffer = []
buffer_length = 0
output_file_index = 0

def write_actor_to_parquet(ds_name, events, force=False):
    global buffer, buffer_length, output_file_index
    len_events = len(events)
    if len_events < minimal_events and not force:
        return
    buffer += events
    buffer_length += len_events
    if buffer_length > 2e6 or force:
        logging.info(f"\nWrite! (buffer length {buffer_length})")
        df = pd.DataFrame(data=buffer, columns=cols+properties_cols)
        opil.save_preprocessed_data(df, ds_name, STAGE, partnum=output_file_index, remove_old_data=False)
        output_file_index += 1
        buffer = []
        buffer_length = 0

def save_json_as_df(filename, host, ds_name, use_all_objects=False):
    global events_per_actor, written_actors
    hoststring = f'SysClient{int(host):04d}.systemia.com'
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
                if "-ptstage1" in ds_name:
                    logging.info(f"Processed {ctr} lines, {succ:,} extracted, {len(events_per_actor)} PIDs non terminated, {len(written_actors)} PIDs terminated, buffer length {buffer_length}, {output_file_index} parts written, {int(time.time() - opil.overall_start_time)}s elapsed")
                else:
                    logging.info(f"Processed {ctr} lines, {succ:,} extracted, {len(events_per_actor)} PIDs, buffer length {buffer_length}, {output_file_index} parts written, {int(time.time() - opil.overall_start_time)}s elapsed")
                # exit()
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
                if key not in cols + unused_cols + ['properties']:
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

            actorID = doc["actorID"]
            if actorID in written_actors: # This process is already terminated
                continue
            if actorID not in events_per_actor:
                events_per_actor[actorID] = []

            events_per_actor[actorID].append(lst)
            succ += 1
        
        keys = list(events_per_actor.keys())
        for actor in keys:
            write_actor_to_parquet(ds_name, events_per_actor[actor])
            del events_per_actor[actor]
        gc.collect()
        if len(events_per_actor) > 0:
            logging.error(f"Error: {len(events_per_actor)} actors remaining")

if __name__ == '__main__':
    for ds_name in args.ds_name.split(","):
        if args.remove:
            logging.info("Removing old data...")
            opil.remove_preprocessed_data(ds_name, STAGE)

        host = ds_name.split("_")[1]
        aia_folder = opil.host_num_to_aia(host)

        ds_type = ds_name.split("_")[2]
        if ds_type not in type_folder_lookup:
            raise ValueError(f"Unknown ds type {ds_type}")

        type_folder = type_folder_lookup[ds_type]

        day_folder = "*" if not "-23night" in ds_name else "23Sep-night"
        files_mask = os.path.join(misc.data_raw(), "ecar", type_folder, day_folder, aia_folder, "*.json.gz")
        files = glob.glob(files_mask)
        files.sort()
        logging.info(f"Found {len(files)} files for ds {ds_name}")

        os.makedirs(os.path.join(misc.root(), f"data/preprocessed/stage{STAGE}", ds_name), exist_ok=True)
        with logging_redirect_tqdm():
            for i, filename in enumerate(tqdm(files, desc="Processing files:", total=len(files), position=0)):
                logging.info(f"Processing ({filename})")
                save_json_as_df(filename, host, ds_name, use_all_objects="-ao" in ds_name)

        # Save remaining
        logging.info(f"{len(events_per_actor)} PIDs remaining")
        for actorID in events_per_actor:
            write_actor_to_parquet(ds_name, events_per_actor[actorID])
        write_actor_to_parquet(ds_name, [], force=True)
    opil.timing_overall()
