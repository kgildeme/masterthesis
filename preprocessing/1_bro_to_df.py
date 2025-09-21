"""
Load bro data into dataframe
Author: Lars Janssen
"""
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parents[1]))

import glob 
import os 
import gc 
import argparse 
import logging 
from tqdm.auto import tqdm 
from tqdm.contrib.logging import logging_redirect_tqdm 
import gzip 

import pandas as pd 
import numpy as np 

import cids.util.optc_util as opil 
import cids.util.misc_funcs as misc 


pd.set_option('display.precision', 12) 
STAGE = 1
logging.basicConfig(level=logging.INFO)

# input to extract host and usecase of df
parser = argparse.ArgumentParser()
parser.add_argument("ds_name", help="Name of the dataset (e.g. nids-v5_201_complete) or comma-separated list", type=str)
parser.add_argument("--remove", help="Remove old dataset files", action="store_true")
args = parser.parse_args()

if not "nids" in args.ds_name or not "_complete" in args.ds_name:
    raise Exception("This script should only be used for '_complete' nids datasets")

def check_if_in_dict(d, name, value):
    if name not in d:
        d[name] = value
        return True
    return value == d[name]

def ip_to_host_num(ip):
    if ip[:7] == "142.20.":
        parts = ip[7:].split(".")
        host =  256 * (int(parts[0]) - 56) + int(parts[1]) - 1
        if host >= 0 and host <= 1000:
            return host
    return -1


class bro_attribute_extractor:
    def __init__(self, fields_line):
        self.fields = {key.strip(): i for (i,key) in enumerate(fields_line[8:].split("\t"))}

    def has_key(self, key):
        return key in self.fields

    def extract(self, string, *keys):
        parts = string.strip().split("\t")
        extracted = []
        for key in keys:
            parts_idx = self.fields[key]
            extracted.append(parts[parts_idx])
        if len(extracted) == 1:
            return extracted[0]
        return extracted

def combine_zeek_files(ds_name, category, extract_field=None, x509_compare_list=None):
    """
    Find all flows for on host in one category
    """
    # init
    host = int(ds_name.split("_")[1])
    output = {}
    fields_dict = {}
    types_dict = {}
    header = {}
    footer = {}
    line_ctr = 0
    valid_ctr = 0
    extracted_field_list = []
    out_dir = os.path.join(misc.data(), f"preprocessed/stage{STAGE}", ds_name)
    files_mask = os.path.join(misc.data_raw(), "bro", "20*", f"{category}.*.log.gz")
    files = glob.glob(files_mask)
    files.sort()
    #files = [f for f in files if f.split("/")[-1].split(".")[0] in ["files", "http", "x509", "ssl", "conn"]]
    logging.info(f"Found {len(files)} files for category '{category}'")
    if len(files) == 0:
        raise Exception("No files found")
    src_keys = ["id.orig_h", "tx_hosts"]
    dst_keys = ["id.resp_h", "rx_hosts"]

    ## Load raw bro files and combine the contents
    os.makedirs(out_dir, exist_ok=True)
    with logging_redirect_tqdm():
        for i, zeek_file in enumerate(tqdm(files, total=len(files), position=0)):
            logging.info(f"Processing ({zeek_file})")

            lines = []
            with gzip.open(zeek_file, "rt") as f:
                lines = f.readlines()
            
            # name = zeek_file.split("/")[-1].split(".")[0]
            name = category
            write_header = False
            if name not in output:
                output[name] = []
                header[name] = []
                write_header = True
            in_header = True

            key_src = None
            key_dst = None
            for line in lines:
                line_ctr += 1
                if line.startswith("#fields"):
                    # check if fields match the other files of the same type
                    if not check_if_in_dict(fields_dict, name, line):
                        logging.critical("Fields mismatch")
                        exit()
                if line.startswith("#types"):
                    if not check_if_in_dict(types_dict, name, line):        
                        logging.critical("Types mismatch")
                        exit()
                if line.startswith("#"): # Write header only once per type, as they are all the same
                    if write_header:
                        header[name].append(line)
                    continue
                if in_header and not line.startswith("#"): # Header has ended
                    write_header = False
                    in_header = False
                    ## init extractor
                    extractor = bro_attribute_extractor([line for line in header[name] if "#fields" in line][0])
                    if name != "x509":
                        for i in range(len(src_keys)):
                            if extractor.has_key(src_keys[i]) and extractor.has_key(dst_keys[i]):
                                key_src = src_keys[i]
                                key_dst = dst_keys[i]
                                break
                        if key_src == None or key_dst == None:
                            raise Exception(f"Could not find src/dst keys in {zeek_file}")

                extract_line = False
                if name == "x509":
                    if x509_compare_list is None:
                        raise Exception("x509_compare_list is None")
                    if extractor.extract(line, extract_field) in x509_compare_list:
                        extract_line = True
                else:
                    # Check for host
                    src_ip, dst_ip = extractor.extract(line, key_src, key_dst)
                    if ip_to_host_num(src_ip) == host or ip_to_host_num(dst_ip) == host:
                        extract_line = True
                if extract_line:
                    output[name].append(line)
                    valid_ctr += 1
                    if not extract_field is None:
                        value = extractor.extract(line, extract_field)
                        if len(value) > 1:
                            extracted_field_list.append(value)

                if line_ctr % int(1e7) == 0:
                    logging.info(f"Extracted {100 * valid_ctr / line_ctr:.2f}%")
            if lines[-1].startswith("#"):
                footer[name] = lines[-1]
    return output, header, footer, extracted_field_list

def zeek_log_to_df(events, header):
    columns = []
    for line in header:
        if line.startswith("#fields"):
            columns = line[len("#fields"):].strip().split("\t")
            break
    if len(columns) == 0:
        logging.error("No columns found")
        exit()
    # Load data
    data = []
    for line in events:
        if line.startswith("#"):
            raise Exception(f"Unexpected header/footer line in body: {line}")
        lst = line.strip().split("\t")
        if len(lst) != len(columns):
            raise Exception(f"Length mismatch {len(lst)} != {len(columns)}")
        data.append(lst)
    objects = pd.DataFrame(data=data, columns=columns)
    num_fuids = 1 if "fuid" in objects.columns else 0
    for fuid_syn in ["id", "cert_chain_fuids", "resp_fuids"]:
        if fuid_syn in objects.columns:
            objects = objects.rename(columns={fuid_syn: "fuid"})
            num_fuids += 1
    if num_fuids > 1:
        raise Exception(f"Multiple fuid columns: {num_fuids}")
    if "conn_uids" in objects.columns:
        objects = objects.rename(columns={"conn_uids": "uid"})
    if "client_cert_chain_fuids" in objects.columns:
        objects = objects.drop(columns=["client_cert_chain_fuids"])
    objects.replace("-", None, inplace=True)
    objects.replace("(empty)", None, inplace=True)
    # weird: uid
    # conn:  uid
    # http:  uid
    # ssl:   uid + fuid
    # x509:  fuid
    # files: fuid + uid
    return objects

def uid_intersection(df1, df2, key="uid"):
    intersection = np.intersect1d(misc.remove_none(df1[key].unique()), misc.remove_none(df2[key].unique()), assume_unique=True)
    return intersection

def load_zeek_events(bro_df_dict):
    for bro_key in bro_df_dict.keys():
        bro_df = bro_df_dict[bro_key]
        info = f"{bro_key}: {len(bro_df)}"
        if "uid" in bro_df.columns:
            info += f"({len(bro_df['uid'].unique())} unique uids" + (" incl None" if None in bro_df['uid'].unique() else "") + ")"
        if "fuid" in bro_df.columns:
            info += f"({len(bro_df['fuid'].unique())} unique fuids" + (" incl None" if None in bro_df['fuid'].unique() else "") + ")"
        logging.info(info)
    
    intersection = uid_intersection(bro_df_dict["ssl"], bro_df_dict["http"])
    if len(intersection) > 0:
        logging.error(f"SSL and HTTP intersection {intersection}")
        exit()

    # Test assumption that x509 fuids are the intersection between ssl and files
    intersection = uid_intersection(bro_df_dict["ssl"], bro_df_dict["files"], key="fuid")
    x509_fuids = bro_df_dict["x509"]["fuid"].unique()
    # check for equality
    if len(set(intersection) - set(x509_fuids)) > 0 or len(set(x509_fuids) - set(intersection)) > 0:
        logging.error("SSL and files fuid intersection is not equal to x509 fuids. Exiting...")
        exit()

    x509_dropped = bro_df_dict["x509"].drop(columns=["ts"])
    files_dropped = bro_df_dict["files"].drop(columns=["ts", "uid", "source"])
    bro_df_dict["ssl"] = bro_df_dict["ssl"].merge(x509_dropped, on="fuid", how="left")
    bro_df_dict["ssl"] = bro_df_dict["ssl"].merge(files_dropped, on="fuid", how="left")
    bro_df_dict["ssl"].replace(np.nan, None, inplace=True)

    files_dropped = bro_df_dict["files"].drop(columns=["ts", "uid", "source"])

    bro_df_dict["http"] = bro_df_dict["http"].merge(files_dropped, on="fuid", how="left")
    bro_df_dict["http"].replace(np.nan, None, inplace=True)

    conn_dropped = bro_df_dict["conn"].drop(columns=["ts", "duration", "id.orig_p", "id.orig_h", "id.resp_h", "id.resp_p", "local_orig"])
    # merge conn into httpuid_intersection
    bro_df_dict["http"] = bro_df_dict["http"].merge(conn_dropped, on="uid", how="left")
    bro_df_dict["ssl"] = bro_df_dict["ssl"].merge(conn_dropped, on="uid", how="left")

    if len(bro_df_dict["ssl"][bro_df_dict["ssl"]["established"].isna()]) > 0:
        logging.error("SSL established is na")
        exit()

    # concat    
    concat = pd.concat([bro_df_dict["ssl"], bro_df_dict["http"]], ignore_index=True)
    concat.replace(np.nan, None, inplace=True)
    logging.info(len(concat))
    return concat

if __name__ == '__main__':
    for ds_name in args.ds_name.split(','):
        # Remove old data
        if args.remove:
            logging.info("Removing old data...")
            opil.remove_preprocessed_data(ds_name, STAGE)

        bro_df_dict = {} # e.g. "conn": pd.DataFrame    # using: files, http, x509, ssl, conn

        files_field = "fuid"
        ssl_field = "cert_chain_fuids"
        x509_field = "id"

        fuids = []
        for category in ["files", "http", "ssl", "conn", "x509"]:
            to_extract = files_field if category == "files" else ssl_field if category == "ssl" else x509_field if category == "x509" else None
            x509_compare_list = fuids if category == "x509" else None
            output, header, footer, extracted = combine_zeek_files(ds_name, category, extract_field=to_extract, x509_compare_list=x509_compare_list)
            fuids += extracted
            logging.info(f"Category {category}: {len(output[category])} lines extracted, {len(extracted)} {to_extract} fields extracted")

            # Convert combined zeek logs into dataframes
            df = zeek_log_to_df(output[category], header[category])
            bro_df_dict[category] = df
            del output, header, footer
            gc.collect()

        # Merge bro files into one big table with all features as columns
        df = load_zeek_events(bro_df_dict)
        opil.timing("Bro events loaded")

        # Save
        opil.save_preprocessed_data(df, ds_name, STAGE)

opil.timing_overall()




