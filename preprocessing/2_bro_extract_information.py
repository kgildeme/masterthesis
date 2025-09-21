"""
extract feature information
Author: Lars Janssen
Refactoring by: Kilian Gildemeister
"""

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parents[1]))

import ipaddress
import argparse
import logging
import json

import pandas as pd
import numpy as np

from cids.util import misc_funcs as misc
from cids.util import optc_util as opil
from cids.util import feature_transformation as transform

STAGE = 2
logging.basicConfig(level=logging.INFO)

def extract_information(df):
    ## URL features: server_name, host, (uri), referrer, subject, certificate.subject, (san.dns: removed)

    # Functions used to apply on columns
    def remove_url_prefix(elem):
        if elem is None:
            return None
        if elem.startswith("http://"):
            elem = elem[7:]
        elif elem.startswith("https://"):
            elem = elem[8:]
        if elem.startswith("www."):
            elem = elem[4:]
        return elem
    def extract_tld(elem):
        if elem is None:
            return None
        if "/" in elem:
            elem = elem.split("/")[0]
        parts = elem.split(".")
        return parts[-1]
    def extract_url_from_subject(elem):
        if elem is None:
            return None
        for part in elem.split(","):
            if part.startswith("CN="):
                return part[3:]
        return None
    def extract_domain(elem):
        if elem is None:
            return None
        return elem.split("/")[0]
    def extract_path(elem):
        if elem is None:
            return None
        return "/" + "/".join(elem.split("/")[1:])
    def extract_file_type_from_path(elem):
        orig = elem
        if elem is None:
            return None
        if "?" in elem:
            elem = elem.split("?")[0]
        elem = elem.split("/")[-1]
        if not "." in elem:
            return None
        return elem.split(".")[-1]
    def sort_analyzers(value):
        if value is None:
            return None
        parts = value.split(",")
        parts.sort()
        return ",".join(parts)
    def remove_method_suffix(elem):
        if elem is None:
            return None
        if elem.endswith(".html"):
            return elem[:-5]
        return elem
    
    # Apply functions
    df.loc[:,"server_name"] = df["server_name"].apply(remove_url_prefix).apply(extract_tld)
    df.loc[:,"host"] = df["host"].apply(remove_url_prefix).apply(extract_tld)
    df.loc[:,"referrer"] = df["referrer"].apply(remove_url_prefix)
    df.loc[:,"subject"] = df["subject"].apply(extract_url_from_subject).apply(remove_url_prefix).apply(extract_tld)
    df.loc[:,"certificate.subject"] = df["certificate.subject"].apply(extract_url_from_subject).apply(remove_url_prefix).apply(extract_tld)
    df.loc[:,"referrer.domain"] = df["referrer"].apply(extract_domain).apply(extract_tld)
    df.loc[:,"referrer.path"] = df["referrer"].apply(extract_path).apply(extract_file_type_from_path)
    #df.loc[:,"san.dns"] = df["san.dns"].apply(extract_tld)
    df.loc[:,"uri"] = df["uri"].apply(extract_file_type_from_path)
    df.loc[:,"analyzers"] = df["analyzers"].apply(sort_analyzers)
    df.loc[:,"status_msg"] = df["status_msg"].apply(lambda x: x.upper() if x is not None else None)
    df.loc[:,"method"] = df["method"].apply(remove_method_suffix)
    df = df.drop(columns=["san.dns"])

    # IP address features
    ip_features = ["id.orig_h", "id.resp_h", "rx_hosts", "tx_hosts"]
    for ip_feature in ip_features:
        df.loc[:, ip_feature] = df[ip_feature].apply(transform.convert_ip)

    # Ports
    df.loc[:, "id.orig_p"] = df["id.orig_p"].apply(transform.convert_port)
    df.loc[:, "id.resp_p"] = df["id.resp_p"].apply(transform.convert_port)

    # Duration
    df.loc[:, "duration"] = df["duration"].apply(lambda x: 0 if x is None else x)
    df.loc[:, "duration"] = pd.to_numeric(df["duration"])
    df = df.rename(columns={"duration": "tanh_duration"})

    df.loc[:, "categorical_rst_bit"] = np.where(df["history"].str.contains("R") | df["history"].str.contains("r"), "R", None)
    
    # categorical features
    categorical_features = ["basic_constraints.ca", "local_orig", "is_orig", "timedout", "local_resp", "id.orig_p", \
                            "id.resp_p", "established", "resumed"] + ip_features
    df = df.rename(columns={feature: f"categorical_{feature}" for feature in categorical_features})

    # tanh features
    tanh_features = ["resp_ip_bytes", "resp_bytes", "orig_ip_bytes", "orig_bytes", "seen_bytes", \
                     "total_bytes", "request_body_len", "response_body_len", "resp_pkts", "orig_pkts", "trans_depth", \
                        "missing_bytes", "overflow_bytes", "missed_bytes"]
    df = df.rename(columns={feature: f"tanh_{feature}" for feature in tanh_features})
    
    # Drop features
    df = df.drop(columns=["history", "md5", "sha1", "sha256", "user_agent", "parent_fuid", "orig_fuids", "certificate.not_valid_before", \
                            "certificate.not_valid_after", "certificate.serial"])

    # Top features
    top_features = ["server_name", "host", "subject", "certificate.subject", "referrer.domain", "method", "curve", "cipher", "version", \
                    "resp_mime_types", "mime_type", "conn_state", "analyzers", "status_msg", "service", "status_code", "history", \
                    "issuer", "uri", "referrer", "referrer.path", "last_alert", "next_protocol", "client_subject", "client_issuer", \
                    "certificate.version", "certificate.issuer", "certificate.key_alg", "certificate.sig_alg", "certificate.key_type", \
                    "certificate.key_length", "certificate.exponent", "certificate.curve", "san.uri", "san.email", "san.ip", \
                    "basic_constraints.path_len", "depth", "filename", "extracted", "extracted_cutoff", "extracted_size", "proto", "tunnel_parents", \
                    "info_code", "info_msg", "tags", "username", "password", "proxied", "orig_filenames", "orig_mime_types", "resp_filenames"]
    df = df.rename(columns={feature: f"top_{feature}" for feature in top_features})
    
    opil.timing("Converted top features")

    return df

def merge_with_ecar_bro(bro_df, host):
    ## Merge bro and ecarbro
    ecarbro_df = opil.load_preprocessed_data_full(f"ecarbro_{host}_complete", 1)
    ecarbro_df = ecarbro_df[["timestamp", "bro_uid", "actorID", "label"]]

    merged = bro_df.merge(ecarbro_df, left_on="uid", right_on="bro_uid", how="left")
    del bro_df, ecarbro_df

    logging.info(f"{len(merged[merged['label'].isna()])}/{len(merged)} zeek events without ecarbro matching")
    #rename
    merged = merged.rename(columns={"timestamp_x": "timestamp"})
    merged = merged.drop(columns=["timestamp_y"])
    merged = merged[~merged["label"].isna()]
    merged = merged.drop(columns=["uid"])

    if len(merged[merged["label"] == 1]) > 0:
        opil.timing("Check if there is a feature that directly describes label")
        for col in merged.columns:
            if col in ["label", "bro_uid", "actorID", "timestamp", "fuid", "tanh_resp_bytes", "tanh_resp_ip_bytes"]:
                continue
            mal = merged[merged["label"] == 1][col]
            ben = merged[merged["label"] == 0][col]
            if np.count_nonzero(mal.isna()) > 0 or np.count_nonzero(ben.isna()) > 0:
                continue
            mal = mal.unique()
            ben = ben.unique()
            intersection = np.intersect1d(mal, ben)
            if len(intersection) == 0:
                logging.warning(f"Warning Col: !!! {col}")
        opil.timing("Check done")

    return merged

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("ds_name", help="Name of the dataset (e.g. nids-v5_201_train) or comma-separated list", type=str)
    parser.add_argument("--remove", help="Remove old dataset files", action="store_true")
    args = parser.parse_args()

    for ds_name in args.ds_name.split(','):
        if args.remove:
            logging.info("Removing old data...")
            opil.remove_preprocessed_data(ds_name, STAGE)

        host = ds_name.split('_')[1]

        # --- Load data ---
        complete_ds = "_".join(ds_name.split("_")[:-1]).replace("-04", "") + "_complete"
        df = opil.load_preprocessed_data_full(complete_ds, STAGE - 1)
        opil.timing("Data loaded")

        # --- Split by time ---
        df = df.rename(columns={"ts": "timestamp"})
        df.loc[:, "timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
        if "_train" in ds_name:
            df = df[df["timestamp"] <= 1569243900]
        elif "_eval" in ds_name:
            df = df[df["timestamp"] > 1569243900]
        elif not "_complete" in ds_name:
            raise Exception("Invalid dataset type")

        # --- Information Extraction ---
        df = extract_information(df)
        opil.timing("Information extraction done")

        # --- Merge with ecarbro (incl. labeling) ---
        df = merge_with_ecar_bro(df, host)
        if "train" in ds_name and np.count_nonzero(df["label"] == 1) > 0:
            raise Exception(f"Error: Found malicious data in train set {ds_name}")
        opil.timing("Labeling done")

        # --- Sort by timestamp ---
        df = df.sort_values(by=["timestamp"], kind="stable")
        df = df.reset_index(drop=True)
        opil.timing("Sorting done")

        # --- Determine top feature value counts ---
        if "_train" in ds_name:
            percentage = 0.004 if "-04" in ds_name else 0.02
            top_features = opil.determine_top_feature_value_counts(df, percentage=percentage)
            opil.save_features_file(top_features, ds_name, "top_val_counts", STAGE)
            opil.timing("Top feature value counts determined")

        # --- Save ---
        opil.save_preprocessed_data(df, ds_name, STAGE)
        opil.timing("Saved")

        misc.export_df_value_counts(df, ds_name, STAGE)
        opil.timing("Value counts exported")

        print("\n".join([col for col in df.columns if not (col.startswith("top_") or col.startswith("categorical_") or col.startswith("tanh_"))]))

    opil.timing_overall()

# removed = ["sha1", "sha256", "md5", "ts", "uid", "uri", "referrer.path", "user_agent", "orig_fuids", "certificate.serial", "fuid"]
# 
# done = ["ts", "uid", "id.orig_p", "id.resp_h", "id.resp_p", "server_name", "subject", "certificate.subject", "host", "uri", "referrer.domain", "referrer.path", "duration", "sha1", "md5",
#             "user_agent", "method", "established", "resumed", "curve", "cipher", "version", 'orig_bytes', 'orig_ip_bytes', 'request_body_len', 'response_body_len', 'resp_bytes', 
#             'resp_ip_bytes', 'seen_bytes', 'total_bytes', 'resp_mime_types', 'conn_state', "certificate.serial", "orig_fuids", "fuid", "analyzers",
#             "status_msg", "trans_depth", "service", "resp_pkts", "status_code", "history", "orig_pkts", "mime_type", "tx_hosts", "referrer"]
