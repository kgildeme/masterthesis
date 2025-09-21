"""
Clean-Up the dat and index each row such that we are able to use a dataloader for iteration over the data
Author: Lars Janssen
Refactoring: Kilian Gildemeister
"""
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parents[1]))

import util
import os
import argparse

import numpy as np

from cids.util import misc_funcs as misc
from cids.util import optc_util as opil

STAGE = 3

def bro_append_objectID(bro_ds_name, ecarbro_ds_name):

    bro = opil.load_preprocessed_data_full(bro_ds_name, 3)
    ecarbro = opil.load_preprocessed_data_full(ecarbro_ds_name, 1)
    ecarbro = ecarbro[["bro_uid", "objectID"]]

    ## Merge bro and ecarbro

    logger.info(f"{len(bro) - bro['bro_uid'].nunique()}/{len(bro)} events with reused uid (http stuff)")
    merged = bro.merge(ecarbro, left_on="bro_uid", right_on="bro_uid", how="left")
    del bro, ecarbro

    logger.info(f"{len(merged[merged['objectID'].isna()])}/{len(merged)} zeek events !without! ecarbro matching")
    logger.info(f"{np.count_nonzero(merged[merged['label'].isna()]['timestamp'] < 1568674800)} of those are before the 17th")
    
     # merged = merged[~merged["objectID"].isna()]
    merged = merged.dropna(subset=["objectID"])
    merged = merged.reset_index(drop=True)

    # add column with increasing counter
    merged.loc[:, "nids_idx"] = np.arange(len(merged))

    return merged 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("ds_name", help="Name of the cids dataset (e.g. cids-v5_201_train) or comma seperated list", type=str)
    parser.add_argument("--hids_ds_name", help="Name of the hids dataset (e.g. hids-v5_201_train)", type=str, default=None)
    parser.add_argument("--nids_ds_name", help="Name of the nids dataset (e.g. nids-v5_201_train)", type=str, default=None)
    parser.add_argument("--remove", help="Remove old dataset files", action="store_true")
    args = parser.parse_args()
    
    if ',' in args.ds_name and (args.hids_ds_name is not None or args.nids_ds_name is not None):
        raise Exception("If multiple datasets are given, hids_ds_name and nids_ds_name must not be set")
     
    
    logger = util.setup_logger(os.path.join(misc.root(), f"logs/preprocessing/{args.ds_name}-STAGE{STAGE}-merge.log"))

    for ds_name in args.ds_name.split(','):
        logger.info(f"Starting with {ds_name}")

        fill_method = ""
        if "-ff" in args.ds_name:
            fill_method = "ffill"

        
        
        if args.hids_ds_name is None:
            hids_ds_name = ds_name.replace("cids", "hids").replace("-ff", "")
        else:
            hids_ds_name = args.hids_ds_name
        if args.nids_ds_name is None:
            nids_ds_name = ds_name.replace("cids", "nids").replace("-ff", "")
        else:
            nids_ds_name = args.nids_ds_name

        if args.remove:
                try:
                    logger.info("Removing old data")
                    opil.remove_preprocessed_data(ds_name, STAGE)
                except Exception as e:
                    logger.exception("An unexpected error occured, while deleting old datasets")
                    raise e
                
        host = ds_name.split("_")[1]
        bro_ds_name = nids_ds_name
        ecar_ds_name = hids_ds_name
        ecarbro_ds_name = f"ecarbro_{host}_complete"

        bro_merged = bro_append_objectID(bro_ds_name, ecarbro_ds_name)
        logger.info("Appended ObjectID to bro df.")

        ecar = opil.load_preprocessed_data_full(ecar_ds_name, 3)

        # Drop all features that are problematic
        redundant_features_bro = opil.load_features_file("_".join(bro_ds_name.split("_")[:2]) + "_train", "redundant_features", 3)
        redundant_features_ecar = opil.load_features_file("_".join(ecar_ds_name.split("_")[:2]) + "_train", "redundant_features", 3)
        bro_merged = bro_merged.drop(columns=redundant_features_bro + ["timestamp", "actorID"])
        ecar = ecar.drop(columns=redundant_features_ecar)
        logger.info("Dropped Redundant features")

        bro_features = ["feature_duration_bro" if col == "feature_duration" else col for col in bro_merged.columns if "feature_" in col]
        ecar_features = ["feature_duration_ecar" if col == "feature_duration" else col for col in ecar.columns if "feature_" in col]

        ecar = ecar.reset_index(drop=True)
        ecar.loc[:,"hids_idx"] = np.arange(len(ecar))
        len_ecar_before_merge = len(ecar)

        ecar_merge = ecar.merge(bro_merged, left_on="objectID", right_on="objectID", how="left")
        del ecar

        ecar_merge = ecar_merge.reset_index(drop=True)
        # ecar_merge_bro_part = ecar_merge[~ecar_merge["label_y"].isna()]
        ecar_merge_bro_part = ecar_merge.dropna(subset=["label_y"])
        opil.timing("Merge Done")

        logger.info(f"{len(ecar_merge_bro_part)} / {len(ecar_merge)} events with bro information")
        logger.info(f"{len(ecar_merge_bro_part[ecar_merge_bro_part['label_y'] == 1])} / {len(ecar_merge[ecar_merge['label_x'] == 1])} malicious events with bro information")
        num_matched_ecar_events = len(ecar_merge_bro_part)

        if not ecar_merge_bro_part["label_x"].astype(int).equals(ecar_merge_bro_part["label_y"].astype(int)):
            logger.error("Label mismatch")
            exit()

        ecar_merge.loc[:,"label"] = ecar_merge["label_x"]
        ecar_merge = ecar_merge.drop(columns=["label_x", "label_y"])

        #rename duration_x to duration_ecar and duration_y to duration_bro
        ecar_merge = ecar_merge.rename(columns={"feature_duration_x": "feature_duration_ecar", "feature_duration_y": "feature_duration_bro"})
        for col in ecar_merge.columns:
            if col.endswith("_x"):
                raise Exception(f"Column {col} still has _x suffix")
            
    # ffill
        if fill_method == "ffill":
            num_ffilled_ecar_events = np.count_nonzero(ecar_merge["feature_duration_bro"].isna())#

            actorID_col = ecar_merge["actorID"].copy(deep=True)
            ecar_merge = ecar_merge.groupby("actorID").ffill()
            ecar_merge.loc[:, "actorID"] = actorID_col
            num_ffilled_ecar_events -= np.count_nonzero(ecar_merge["feature_duration_bro"].isna())
            opil.timing(f"Filled {num_ffilled_ecar_events} NaN values")

        ecar_merge = ecar_merge.drop(columns=["hids_idx", "nids_idx"])

        # fill0
        for col in ecar_merge.columns:
            if (not col in bro_merged.columns) and (not col == "feature_duration_bro") and ecar_merge[col].hasnans:
                raise Exception(f"ecar col {col} has NaNs")
        ecar_merge = ecar_merge.fillna(0)
        ecar_merge = ecar_merge.drop(columns=["fuid", "bro_uid"])
        opil.save_preprocessed_data(ecar_merge, ds_name, 3)
        opil.timing(f"Saved {ds_name}")

        print(f"Matched {num_matched_ecar_events} ecar events with bro events")

        misc.export_df_value_counts(ecar_merge, ds_name, 3)

    opil.timing_overall()
