from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parents[1]))

import json
import argparse
import logging
import os

from cids.util import optc_util as opil, misc_funcs as misc


logging.basicConfig(level=logging.INFO)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("ds_name", help="Name of the dataset (e.g. hids-v5_201_train) or comma-separated list", type=str)
    parser.add_argument("stage", help="The preprocessing stage from which the features should be extracted", type=int)
    args = parser.parse_args()

    for ds_name in args.ds_name.split(','):

        df = opil.load_preprocessed_data_full(ds_name, args.stage)
        dtype_dict = {col: str(df[col].dtype) for col in df.columns}
        print(f"The loaded dataset {ds_name} has the following properties:\n \
              \t {len(df)} rows \n \
              \t {len(df.columns)} columns \n \
              \t {len(df[df["label"] == 0])} benign samples \n \
              \t {len(df[df["label"] == 1])} malicious samples \n ")

        with open(os.path.join(misc.data(), "tmp/dataset_features", f"{ds_name}_STAGE_{args.stage}.json"), 'w') as json_file:
            json.dump(dtype_dict, json_file, indent=2)

        logging.info(f"json for {ds_name} created")

    logging.info("DONE")