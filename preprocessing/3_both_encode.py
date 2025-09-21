"""
encode features
Author: Lars Janssen
Refactoring by: Kilian Gildemeister
"""
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parents[1]))

import argparse
import json
import os

import pandas as pd
import numpy as np

from cids.util import optc_util as opil
from cids.util import misc_funcs as misc
import util

STAGE = 3

value_counts =  {feature: ["well_known", "registered", "ephemeral"] for feature in ["src_port", "dest_port", "id.orig_p", "id.resp_p"]} |\
                {feature: ["ipv4_priv", "ipv4_pub", "ipv6_priv", "ipv6_pub"] for feature in ["src_ip", "dest_ip", "id.orig_h", "id.resp_h", "rx_hosts", "tx_hosts"]} |\
                {feature: ["ns", "sys", "user", "ls", "host", "other"] for feature in ["user", "principal", "sid"]} |\
                {feature: ["T", "F"] for feature in ["basic_constraints.ca", "local_orig", "is_orig", "timedout", "local_resp", "established", "resumed"]} |\
                {feature: ["R"] for feature in ["rst_bit"]} |\
                {feature: ["inbound", "outbound"] for feature in ["direction"]}

tanh_factors =  {"duration": 1, "trans_depth": 10, "resp_pkts": 100, "orig_pkts": 100} |\
                {feature: 1000 for feature in ["size", "resp_ip_bytes", "resp_bytes", "orig_ip_bytes", "orig_bytes", 
                    "seen_bytes", "total_bytes", "request_body_len", "response_body_len", "missing_bytes", "overflow_bytes", "missed_bytes"]}

def stanh(factor):
    def _stanh(x):
        if x is None:
            return 0
        return np.tanh(float(x) / factor)
    return _stanh

def encode_features(df, ds_name):
    # Load top features value counts
    top_features = opil.load_features_file("_".join(ds_name.split("_")[:2]).replace("-ping", "") + "_train", "top_val_counts", STAGE - 1)
    #top_features = {}
    logger.info(f"top features with value count:\n{json.dumps(top_features)}")

    if "-oldfeatureset" in ds_name:
        dropping = [f"feature_{feature}_{suffix}" for feature in ["command_line", "module_path"] for suffix in ["is_windows_dir", "is_user_dir", "is_program_files_dir", "is_file", "is_folder"]]
        dropping += [f"feature_{feature}_{suffix}" for feature in ["image_path", "parent_image_path"] for suffix in ["is_file", "is_folder"]]
        df = df.drop(columns=dropping)
    else:
        if "categorical_sid" in df.columns:
            df.loc[df["categorical_sid"] == "S-1-5-21", "categorical_sid"] = "other"

    to_drop = []
    to_append = []
    for feature in df.columns:
        if not "_" in feature:
            logger.info(f"Skipping {feature}")
            continue
        f_type = feature.split('_')[0]
        f_name = "_".join(feature.split('_')[1:])
        if f_type == "feature":
            continue
        elif f_type == "categorical":
            encoded = 0
            for key in value_counts.keys():
                if f_name == key:
                    # one-hot encode
                    for value in value_counts[key]:
                        #df[f"feature_{f_name}_{value}"] = (df[feature] == value).astype(int)
                        to_append.append((f"feature_{f_name}_{value}", np.where(df[feature] == value, 1, 0)))
                        if value == "S-1-5-21":
                            logger.warning(f"Encoded value S-1-5-21 in feature_{f_name}_{value}\n \
                              for {np.count_nonzero(df[feature] == value)} instances\n \
                              and set 0 for {np.count_nonzero(df[feature] != value)} incstances\n \
                              and a total of {df[feature].value_counts()}")
                    # check if feature has other values than the ones in value_counts
                    for value in df[feature].unique():
                        if not value in value_counts[key] and not value is None:
                            raise Exception(f"Error: {f_name} has value not in value_counts: {value}, expected values: {value_counts[key]}")
                    encoded += 1
            if encoded == 0:
                raise Exception("Error: No categories defined for categorical feature", feature)
            elif encoded > 1:
                raise Exception(f"Error: Encoded {feature} {encoded} times")
        elif f_type == "tanh":
            if not f_name in tanh_factors:
                logger.warning("Could not encode %s", feature)
                continue
            factor = tanh_factors[f_name]
            df.loc[:, f"feature_{f_name}"] = df[feature].apply(stanh(factor))
        elif f_type == "top":
            amount = len([f for f in top_features if f.startswith(feature + "_top_")])
            if "-ping" in ds_name and "image_path" in f_name:
                amount += 1
            top_values = []
            for i in range(amount):
                if i == amount - 1 and "-ping" in ds_name and "image_path" in f_name:
                    value = "ping.exe"
                else:
                    if f"{feature}_top_{i+1}" not in top_features:
                        raise Exception(f"Error: top feature not found: {feature}_top_{i+1}")
                    value = top_features[f"{feature}_top_{i+1}"]
                top_values.append(value)
                to_append.append((f"feature_{f_name}_top_{i+1}", np.where(df[feature] == value, 1, 0)))
            # make one last feature for all other values
            if amount > 0:
                to_append.append((f"feature_{f_name}_other", np.where(df[feature].notnull() & ~df[feature].isin(top_values), 1, 0)))
        elif feature == "acuity_level":
            df.loc[:,"feature_acuity_level"] = pd.to_numeric(df["acuity_level"]) / 5
        else:
            logger.info("Skipping %s", feature)
            continue
        to_drop.append(feature)

    logger.info(f"Encoded the following features: {to_drop}")
    opil.timing("Encoding done")

    df = df.drop(columns=to_drop)
    opil.timing("Dropped columns")

    concat = [df] + [pd.DataFrame(data, columns=[name]) for name, data in to_append]
    df = pd.concat(concat, axis=1)
    opil.timing("Appended columns")

    return df

def determine_redundant_features(df):
    columns = [feature for feature in df.columns if feature.startswith("feature_")]
    columns.sort(reverse=True)
    errors = 0
    features_total = 0

    cols_1elem = []
    for col in columns:
        if len(df[col].unique()) == 1:
            cols_1elem.append(col)
        if df[col].isnull().sum() > 0:
            logger.error(f"Error: Feature {col} has {df[col].isnull().sum()} null/nan values")
            errors += 1
        if df[col].min() < 0 or df[col].max() > 1:
            logger.error(f"Error: Feature {col} has value range {df[col].min()} - {df[col].max()}.")
            errors += 1
        features_total += 1

    logger.info(f"{len(cols_1elem)} features with only one unique value: {cols_1elem}")

    if errors > 0:
        raise Exception(f"{errors} errors found, during post-processing feature analysis. Exiting.")

    same_groups = []
    same_to_drop = []
    for i, feature1 in enumerate(columns):
        if feature1 in cols_1elem:
            continue
        is_same = False
        for g, group in enumerate(same_groups):
            if df[feature1].equals(group[0]):
                is_same = True
                same_groups[g].append(df[feature1])
        if not is_same:
            same_groups.append([df[feature1]])


    redundant_string = f"Same features:\n"
    for g, group in enumerate(same_groups):
        if len(group) > 1:
            redundant_string += f"Group {g}: {[f.name for f in group]}\n"
            same_to_drop += [f.name for f in group[1:]]
    logger.info(redundant_string)

    # columns that should be dropped
    to_drop = cols_1elem + same_to_drop
    logger.info(f"Dropping {to_drop}")
    logger.info(f"Dropping {len(to_drop)}/{features_total} features")

    return to_drop

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("ds_name", help="Name of the dataset (e.g. hids-v5_201_train) or comma-separated list", type=str)
    parser.add_argument("--remove", help="Remove old dataset files", action="store_true")
    args = parser.parse_args()
    
    logger = util.setup_logger(os.path.join(misc.root(), f"logs/preprocessing/{'-'.join(args.ds_name.split(','))}.log"))

    if "-oldfeatureset" in args.ds_name:
        value_counts["sid"] = ["ns", "sys", "user", "ls", "host", "other", "S-1-5-21"]

    for ds_name in args.ds_name.split(','):
        if args.remove:
            try:
                logger.info("Removing old data...")
                opil.remove_preprocessed_data(ds_name, STAGE)
            except Exception as e:
                logger.exception("An unexpected error occured, while deleting old datasets")
                raise e

        try:
            df = opil.load_preprocessed_data_full(ds_name.replace("-ping", ""), STAGE - 1)
            opil.timing("Loaded")
        except Exception as e:
            logger.exception("An unexpected error occured while loading previous stage")
            raise e
        try:
            df = encode_features(df, ds_name)
        except Exception as e:
            logger.exception("An unexpected error occured while encoding features")
            raise e
        
        try:
            if "_train" in ds_name:
                redundant_features = determine_redundant_features(df)
                opil.save_features_file(redundant_features, ds_name, "redundant_features", STAGE)
                opil.timing("Determined redundant features")
        except Exception as e:
            logger.exception("An unexpected error occured while detecting redundant features")
            raise e

        misc.export_df_value_counts(df, ds_name, STAGE)
        opil.save_preprocessed_data(df, ds_name, STAGE)

    opil.timing_overall()
