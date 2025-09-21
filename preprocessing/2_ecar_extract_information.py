"""
extract feature information
Author: Lars Janssen
Refactoring by: Kilian Gildemeister
"""

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parents[1]))

import argparse
import logging

import pandas as pd
import numpy as np

from cids.util import misc_funcs as misc
from cids.util import optc_util as opil
from cids.util import feature_transformation as transform

STAGE = 2
logging.basicConfig(level=logging.INFO)

# --- Removal of multicast management connections of other hosts ---
multicast_removed_total = 0
multicast_num_events_total = 0
flow_total = 0

def remove_by_mask(df, seq, text):
    df.loc[seq, "to_be_removed"] = 1
    return df

def remove_multicast_traffic(df, host):
    global multicast_removed_total, multicast_num_events_total, flow_total
    start_size = len(df)
    flow_total += len(df[df["object"] == "FLOW"])
    
    # Mark all multicast management connections
    df.loc[:, "to_be_removed"] = 0
    df = remove_by_mask(df, (df["src_port"] == "5353") | (df["dest_port"] == "5353"), "mDNS (port 5353)")
    df = remove_by_mask(df, (df["dest_port"] == "5355"), "LLMNR (port 5355)")
    df = remove_by_mask(df, df["dest_ip"] == "10.50.255.255", "NetBIOS (dst: 10.50.255.255)")
    df = remove_by_mask(df, df["dest_ip"] == "142.20.59.255", "NetBIOS (dst: 142.20.59.255)")

    # Keep all connections that involve the selected host
    ips = opil.get_host_ips(int(host))
    if len(ips) < 2:
        raise Exception(f"Error: Host {host} has less than 2 IPs")
    where_host_ip = (df["src_ip"] == "127.0.0.1") | (df["dest_ip"] == "127.0.0.1") | (df["src_ip"] == "::1") | (df["dest_ip"] == "::1")
    for ip in ips:
        where_host_ip = where_host_ip | (df["src_ip"] == ip) | (df["dest_ip"] == ip)
    df.loc[where_host_ip, "to_be_removed"] = 0
    
    # Remove
    df = df[df["to_be_removed"] == 0]
    df = df.drop(columns=["to_be_removed"])

    multicast_removed_total += start_size - len(df)
    multicast_num_events_total += start_size
    print(f"Removed {start_size - len(df)}")
    return df

def gen_is_zeek_process_feature(df, host):
    ecarbro = opil.load_preprocessed_data_full(f"ecarbro_{host}_complete", 1)
    bro_processes = ecarbro["actorID"].unique()
    if len(bro_processes) == 0:
        raise Exception(f"Error: No processes found in ecarbro dataset for host {host}")
    actors = df["actorID"].unique()
    
    df.loc[:, "is_zeek_process"] = np.where(df["actorID"].isin(bro_processes), 1, 0)
    logging.info(f"{len([actor for actor in actors if actor in bro_processes])}/{len(actors)} ecar processes have zeek data "\
          + f"(which make up {len(df[(df['is_zeek_process'] == 1) & (df['label'] == 1)])}/{len(df[df['label'] == 1])} malicious events)")
    return df

def label(df, host):
    labels = df["actorID"].copy(deep=True)
    labels[labels.isin(opil.mal_processes(host, dict=True))] = 1
    labels[labels != 1] = 0
    mal_count = labels.value_counts()[1] if 1 in labels.value_counts() else 0
    logging.info(f"Found {mal_count} malicious, {labels.value_counts()[0] if 0 in labels.value_counts() else 0} non-malicious")
    df.loc[:, "label"] = labels

    df = gen_is_zeek_process_feature(df, host)

    return df, mal_count

def extract_cmd_filename(s):
    if s is None:
        return None
    if s[0] == '"':
        s = s[1:].split('"')[0]
    else:
        s = s.split(" ")[0]
    return s

def extract_information(df, host):
    # Convert timestamp to unix timestamp
    df.loc[:, "timestamp"] = df["timestamp"].apply(opil.convert_timestamp)
    opil.timing("Timestamps converted")

    # IP address features
    df.loc[:, "src_ip"] = df["src_ip"].apply(transform.convert_ip)
    df.loc[:, "dest_ip"] = df["dest_ip"].apply(transform.convert_ip)
    df = df.rename(columns={"src_ip": "categorical_src_ip", "dest_ip": "categorical_dest_ip"})

    # Ports
    df.loc[:, "src_port"] = df["src_port"].apply(transform.convert_port)
    df.loc[:, "dest_port"] = df["dest_port"].apply(transform.convert_port)
    df = df.rename(columns={"src_port": "categorical_src_port", "dest_port": "categorical_dest_port"})
    df = df.rename(columns={"direction": "categorical_direction"})

    ## User features
    df["sid"] = df["sid"].str[:8]
    for user_feature in ["user", "principal", "sid"]:
        df.loc[:,user_feature] = df[user_feature].replace({"": None})
        df = transform.extract_user_feature(df, user_feature, host)
        df = df.rename(columns={user_feature: f"categorical_{user_feature}"})

    opil.timing("IP, port, user features done")

    ## Shell object (instead of keeping payload or context_info we just have a boolean indicating if it is a shell (command) object)
    df.loc[:,"feature_is_shell_obj"] = np.where(df["object"] == "SHELL", 1, 0)
    
    ## Path features
    df.loc[:, "command_line"] = df["command_line"].apply(extract_cmd_filename)
    df, file_features = transform.extract_path_features(df)
    opil.timing("Path features done")

    # size feature
    df.loc[:,"size"] = pd.to_numeric(df["size"])
    df.loc[:,"size"] = np.where(df["size"].isnull(), 0, df["size"])
    df = df.rename(columns={"size": "tanh_size"})

    # duration = end_time - start_time
    df.loc[:,"duration"] = pd.to_numeric(df["end_time"]) - pd.to_numeric(df["start_time"])
    df.loc[:,"duration"] = np.where(df["duration"].isnull(), 0, df["duration"])
    df = df.rename(columns={"duration": "tanh_duration"})

    # Remaining features
    df = df.rename(columns={feature: f"top_{feature}" for feature in ["l4protocol", "info_class"]})

    # Remove columns that are not needed anymore
    df = df.drop(columns=file_features + ["start_time", "end_time", "context_info", "payload", "base_address"])
    return df

def remove_events_after_process_termination(df):
    df.loc[:, "remove"] = None
    df["remove"] = df["remove"].astype(df["timestamp"].dtype)
    pt_event_indices = (df["object"] == "PROCESS") & (df["action"] == "TERMINATE")
    df.loc[pt_event_indices, "remove"] = df.loc[pt_event_indices, "timestamp"]
    df.loc[:, "remove"] = df[["actorID", "remove"]].groupby("actorID").ffill()["remove"]
    df.loc[pt_event_indices, "remove"] = 0.
    df.loc[:, "remove"] = df["remove"].fillna(0.).infer_objects(copy=False)
    df.loc[:, "diff"] = 0.
    df.loc[df["remove"] > 0, "diff"] = (df.loc[df["remove"] > 0, "timestamp"] - df.loc[df["remove"] > 0, "remove"]).astype('float64')
    logging.info(f"Removing {np.count_nonzero(df['diff'] > 1)} events after process termination")
    df = df[df["diff"] < 1]
    df = df.drop(columns=["remove", "diff"])
    return df

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("ds_name", help="Name of the dataset (e.g. hids-v5_201_train) or comma-separated list", type=str)
    parser.add_argument("--remove", help="Remove old dataset files", action="store_true")
    args = parser.parse_args()

    for ds_name in args.ds_name.split(","):

        if args.remove:
            logging.info("Removing old data...")
            opil.remove_preprocessed_data(ds_name, STAGE)
        
        host = ds_name.split('_')[1]

        # --- Load data ---
        df = opil.load_preprocessed_data_full(ds_name.replace("-tid", "").replace("-wm", ""), STAGE - 1)
        opil.timing("Data loaded")

        # --- Labeling ---
        df, mal_count = label(df, host)
        if "train" in ds_name and mal_count > 0:
            raise Exception(f"Error: Found malicious data in train set {ds_name}")
        opil.timing("Labeling done")

        # --- Remove multicast management traffic ---
        if not "-wm" in ds_name:
            df = remove_multicast_traffic(df, host)
            opil.timing("Multicast traffic removed")
        if mal_count > 0:
            if not 1 in df["label"].value_counts() or mal_count != df["label"].value_counts()[1]:
                raise Exception(f"Error: Malicious events were removed. Mismatch ({mal_count} != {df['label'].value_counts()[1]})")

        # --- Information Extraction ---
        df = extract_information(df, host)
        opil.timing("Information extraction done")
        # --- Sort by timestamp ---
        df.loc[:, "sort-last"] = 0
        df.loc[(df["object"] == "PROCESS") & (df["action"] == "TERMINATE"), "sort-last"] = 1
        if "-tid" in ds_name:
            df = df.sort_values(by=["actorID", "timestamp", "tid", "sort-last"], kind="stable")
        else:
            df = df.sort_values(by=["actorID", "timestamp", "sort-last"], kind="stable")
        df = df.reset_index(drop=True)
        df = df.drop(columns=["sort-last"])
        opil.timing("Sorting done")

        # Check if any malicious events were removed
        if mal_count > 0:
            if not 1 in df["label"].value_counts() or mal_count != df["label"].value_counts()[1]:
                raise Exception(f"Error: Malicious events were removed. Mismatch ({mal_count} != {df['label'].value_counts()[1]})")

        # --- Remove events after process termination
        df = remove_events_after_process_termination(df)
        opil.timing("Events after P-T removed")
        df = df.reset_index(drop=True)

        # --- Determine top feature value counts ---
        if "_train" in ds_name:
            top_features = opil.determine_top_feature_value_counts(df)
            opil.save_features_file(top_features, ds_name, "top_val_counts", STAGE)
            opil.timing("Top feature value counts determined")

        # --- Save ---
        opil.save_preprocessed_data(df, ds_name, STAGE)
        opil.timing("Saved")

        # --- Misc Output ---
        if multicast_num_events_total > 0:
            logging.info(f"{ds_name}: Removed {multicast_removed_total}/{multicast_num_events_total} ({(multicast_removed_total / multicast_num_events_total) * 100:.2f}%) multicast traffic (all events)")
            if flow_total > 0:
                logging.info(f"{ds_name}: Removed {multicast_removed_total}/{flow_total} ({(multicast_removed_total / flow_total) * 100:.2f}%) multicast traffic (flow objects)")
        misc.export_df_value_counts(df, ds_name, STAGE)
        opil.timing("Value counts exported")

    opil.timing_overall()
