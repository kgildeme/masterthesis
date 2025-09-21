from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parents[1]))

import argparse
import logging
import os
import pandas as pd
from cids.util import optc_util as opil
from cids.util import misc_funcs as misc
import util

STAGE = 5

def extract_malicous_processes(ds_name, stage=5):
    try:
        df = opil.load_preprocessed_data_full(ds_name, stage)
        meta = opil.load_preprocessed_data_full(ds_name + "_meta", stage)
    except Exception as e:
            logger.exception("An unexpected error occured while loading previous stage")
            raise e
    
    idx = df.index[df["label"] == 1]
    logger.info(f"Found {len(idx)} malicious samples in {ds_name}")

    return df.loc[idx], meta.loc[idx]


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("base_ds", help="Name of the base dataset to which malicious samples should be added")
    parser.add_argument("add_on_dss", help="ds or comma seperated list of ds which malicious samples should be added")
    parser.add_argument("--remove", action="store_true")
    parser.add_argument("--stage", type=int, default=5)
    parser.add_argument("--stage_add", type=int, default=5)
    args = parser.parse_args()

    hosts = []
    for add_on in args.add_on_dss.split(","):
         host = add_on.split("_")[1]
         hosts.append(host)

    logger = util.setup_logger(os.path.join(misc.root(), f"logs/preprocessing/{args.base_ds}-{'-'.join(hosts)}-STAGE{STAGE}.log"), level=logging.INFO)

    final_ds_name = f"{args.base_ds}--{'-'.join(hosts)}"
    if args.remove:
            try:
                logger.info("Removing old data")
                opil.remove_preprocessed_data(final_ds_name, STAGE)
            except Exception as e:
                logger.exception("An unexpected error occured, while deleting old datasets")
                raise e
            
    # collect malicous processes
    malicious_data = []
    malicious_meta = []
    total_length = 0
    
    try:
        base_df = [opil.load_preprocessed_data_full(args.base_ds, stage=args.stage)]
        base_meta = [opil.load_preprocessed_data_full(args.base_ds + "_meta", stage=args.stage)]

    except Exception as e:
            logger.exception("An unexpected error occured while loading previous stage")
            raise e

    for ds in args.add_on_dss.split(","):
        md = extract_malicous_processes(ds, stage=args.stage_add)

        columns = [c for c in md[0].columns if c in base_df[0].columns]
        columns_not = [c for c in md[0].columns if c not in base_df[0].columns]
        logger.info(f"columns: {columns}")
        malicious_data.append(md[0][columns])
        malicious_meta.append(md[1][base_meta[0].columns])

        total_length += len(md[0])
    
    original_length = len(base_df[0])
    logger.info(f"Original lenght is {original_length}")
    base_df.extend(malicious_data)
    base_meta.extend(malicious_meta)
    # Combine malicious data with base data
    base_df = pd.concat(base_df, ignore_index=True)
    base_df.fillna(0, inplace=True)
    base_meta = pd.concat(base_meta, ignore_index=True)

    logger.info(f"Added {len(base_df)-original_length} / {len(base_df)} samples to {args.base_ds}")

    # Save the updated data
    try:
        opil.save_preprocessed_data(base_df, final_ds_name, stage=STAGE, parts=4)
        opil.save_preprocessed_data(base_meta, final_ds_name + "_meta", stage=STAGE)
        logger.info(f"Successfully saved updated dataset: {final_ds_name}")
    except Exception as e:
        logger.exception("An unexpected error occurred while saving the updated dataset")
        raise e