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

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("base_ds", help="Name of the base dataset from which malicious samples should be extracted")
    parser.add_argument("--remove", action="store_true")
    parser.add_argument("--stage", type=int, default=5)
    parser.add_argument("--malicious", action="store_true")
    args = parser.parse_args()
    logger = util.setup_logger(os.path.join(misc.root(), f"logs/preprocessing/{args.base_ds}-bening-STAGE{STAGE}.log"), level=logging.INFO)

    final_ds_name = f"{args.base_ds}--benign_only" if not args.malicious else f"{args.base_ds}--malicious_only"
    if args.remove:
            try:
                logger.info("Removing old data")
                opil.remove_preprocessed_data(final_ds_name, STAGE)
            except Exception as e:
                logger.exception("An unexpected error occured, while deleting old datasets")
                raise e

    try:
        base_df = opil.load_preprocessed_data_full(args.base_ds, stage=args.stage)
        base_meta = opil.load_preprocessed_data_full(args.base_ds + "_meta", stage=args.stage)

    except Exception as e:
            logger.exception("An unexpected error occured while loading previous stage")
            raise e
    
    if args.malicious:
         idx = base_df.index[base_df["label"] == 1]
    else:
        idx = base_df.index[base_df["label"] == 0]

    logger.info(f"Found {len(idx)} benign samples in {args.base_ds}")

    base_df = base_df.loc[idx]
    base_meta = base_meta.loc[idx]

    # Save the updated data
    try:
        opil.save_preprocessed_data(base_df, final_ds_name, stage=STAGE, parts=4)
        opil.save_preprocessed_data(base_meta, final_ds_name + "_meta", stage=STAGE)
        logger.info(f"Successfully saved updated dataset: {final_ds_name}")
    except Exception as e:
        logger.exception("An unexpected error occurred while saving the updated dataset")
        raise e