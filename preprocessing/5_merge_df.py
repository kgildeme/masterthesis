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
    parser.add_argument("ds", help="Name of the base dataset from which malicious samples should be extracted")
    parser.add_argument("--remove", action="store_true")
    parser.add_argument("--stage", type=int, default=5)
    args = parser.parse_args()
    logger = util.setup_logger(os.path.join(misc.root(), f"logs/preprocessing/{args.ds}-merge-STAGE{STAGE}.log"), level=logging.INFO)


    final_ds_name = "--".join(ds for ds in args.ds.split(","))

    if args.remove:
            try:
                logger.info("Removing old data")
                opil.remove_preprocessed_data(final_ds_name, STAGE)
            except Exception as e:
                logger.exception("An unexpected error occured, while deleting old datasets")
                raise e
    dss = []
    dss_meta = []
    try:
        for ds in args.ds.split(","):
            dss.append(opil.load_preprocessed_data_full(ds, stage=args.stage))
            dss_meta.append(opil.load_preprocessed_data_full(ds + "_meta", stage=args.stage))

    except Exception as e:
            logger.exception("An unexpected error occured while loading previous stage")
            raise e

    full_df = pd.concat(dss, axis=0)
    full_meta = pd.concat(dss_meta, axis=0)
    # Save the updated data
    try:
        opil.save_preprocessed_data(full_df, final_ds_name, stage=STAGE, parts=4)
        opil.save_preprocessed_data(full_meta, final_ds_name + "_meta", stage=STAGE)
        logger.info(f"Successfully saved updated dataset: {final_ds_name}")
    except Exception as e:
        logger.exception("An unexpected error occurred while saving the updated dataset")
        raise e