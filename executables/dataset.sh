#!/bin/bash

# Check if host_id is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <host_id> [start_stage] [dataset_type]"
  echo "dataset_type: train, eval, or both (default: both)"
  exit 1
fi

HOST_ID=$1
START_STAGE=${2:-1}  # Default to stage 1 if not provided

# Define dataset names
HIDS_TRAIN="hids-v5_${HOST_ID}_train"
HIDS_EVAL="hids-v5_${HOST_ID}_eval"
NIDS_TRAIN="nids-v5_${HOST_ID}_train"
NIDS_EVAL="nids-v5_${HOST_ID}_eval"
NIDS_COMPLETE="nids-v5_${HOST_ID}_complete"
ECARBRO_COMPLETE="ecarbro_${HOST_ID}_complete"d
CIDS_TRAIN="cids-v5_${HOST_ID}_train"
CIDS_EVAL="cids-v5_${HOST_ID}_eval"
CIDS_TRAIN_FF="cids-v5_${HOST_ID}_train-ff"
CIDS_EVAL_FF="cids-v5_${HOST_ID}_eval-ff"

# Check if any conda environment is active
if [ -z "$CONDA_PREFIX" ]; then
  echo "Please activate a conda environment before running the script."
  exit 1
fi

# Run preprocessing scripts in order based on the starting stage
if [ "$START_STAGE" -le 1 ]; then
  echo "Running Stage 1: Data Conversion"
  python3 preprocessing/1_ecar_to_df.py $HIDS_TRAIN,$HIDS_EVAL --remove
  python3 preprocessing/1_bro_to_df.py $NIDS_COMPLETE --remove
  python3 preprocessing/1_ecar-bro_to_df.py $ECARBRO_COMPLETE --remove
fi

if [ "$START_STAGE" -le 2 ]; then
  echo "Running Stage 2: Feature Extraction"
  python3 preprocessing/2_ecar_extract_information.py $HIDS_TRAIN,$HIDS_EVAL --remove
  python3 preprocessing/2_bro_extract_information.py $NIDS_TRAIN,$NIDS_EVAL --remove
fi

if [ "$START_STAGE" -le 3 ]; then
  echo "Running Stage 3: Encoding & Merging"
  python3 preprocessing/3_both_encode.py $HIDS_TRAIN,$HIDS_EVAL,$NIDS_TRAIN,$NIDS_EVAL --remove
  python3 preprocessing/3_merge.py $CIDS_TRAIN,$CIDS_EVAL,$CIDS_TRAIN_FF,$CIDS_EVAL_FF --remove
fi

if [ "$START_STAGE" -le 4 ]; then
  echo "Running Stage 4: Cleaning & Indexing"
  python3 preprocessing/4_clean_and_index.py $CIDS_TRAIN,$CIDS_EVAL,$CIDS_TRAIN_FF,$CIDS_EVAL_FF,$HIDS_TRAIN,$HIDS_EVAL,$NIDS_TRAIN,$NIDS_EVAL --remove
fi

echo "Preprocessing completed."
