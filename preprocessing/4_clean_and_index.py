"""
Clean-Up the dat and index each row such that we are able to use a dataloader for iteration over the data
Author: Kilian Gildemeister
Inspired by code from: Lars Janssen
"""
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parents[1]))

import argparse
import json
import os

import dask.dataframe as dd

from cids.util import optc_util as opil
from cids.util import misc_funcs as misc
import util

STAGE = 4

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("ds_name", help="Name of the dataset (e.g. hids-v5_201_train) or comma-separated list", type=str)
    parser.add_argument("--remove", help="Remove old dataset files", action="store_true")
    args = parser.parse_args()
    
    logger = util.setup_logger(os.path.join(misc.root(), f"logs/preprocessing/{'-'.join(args.ds_name.split(','))}-STAGE{STAGE}.log"))

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
            opil.timing("Loaded")
        except Exception as e:
            logger.exception("An unexpected error occured while loading previous stage")
            raise e
        
        # Dropping redundant features, for cids-dataset already done
        try:
            feature_file_ds = '_'.join(ds_name.split('_')[:2]) + "_train"
            if opil.feature_file_exists(feature_file_ds, "redundant_features", STAGE - 1):
                redundant_features = opil.load_features_file(feature_file_ds, "redundant_features", STAGE-1)
                df = df.drop(columns=redundant_features)
                opil.timing("Redundant features dropped")
            elif "cids" in ds_name:
                logger.info("SKipping Redundant features as already done for cids")
            else:
                raise FileNotFoundError("Could not find file %s", feature_file_ds + ".pkl")
        except FileNotFoundError as e:
            logger.exception("Redundant feature file not found. Run STAGE 3 on the trainset to create feature file")
            raise e
        except Exception as e:
            logger.exception("An unexpected error occured")
            raise e
        
        feature_cols = df.columns[df.columns.str.startswith("feature_")].tolist()
        feature_df = df[feature_cols + ['label']]
        metadata_df = df.drop(columns=feature_cols)
        metadata_cols = metadata_df.columns
        del df
        opil.timing("Created feature and metadata")

        #Indexing
        row_idx = feature_df.index
        feature_df.loc[:,"row_idx"] = row_idx
        metadata_df.loc[:, "row_idx"] = row_idx
        actors = metadata_df["actorID"]
        new_process = (actors != actors.shift(periods=1)).astype(int).to_numpy() # 1 if new process, 0 if same process
        feature_df.loc[:,"new_process"] = new_process
        opil.timing("Indexed feature frame")

        # Save
        opil.save_preprocessed_data(feature_df, ds_name, STAGE, parts=16)
        opil.save_preprocessed_data(metadata_df, ds_name + "_meta", STAGE)
        #feature_ddf = dd.from_pandas(feature_df, npartitions=3)
        #feature_ddf.to_parquet(os.path.join(misc.data(), f"preprocessed/stage4/{ds_name}/dask"), write_index=True, engine="pyarrow", compression="zstd")
        opil.timing("Saved PD")


        misc.export_df_value_counts(feature_df, ds_name, STAGE)
        misc.export_df_value_counts(metadata_df, ds_name + "_meta", STAGE)
        opil.timing("Saving")

    opil.timing_overall()



        
        

            
