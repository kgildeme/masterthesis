"""
Miscalanous helper functions
"""
from pathlib import Path
import os
import glob
import json
import io
import PIL

import matplotlib.figure
import torch
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib
from torchvision.transforms import v2 as transforms

_ROOT_DIR = Path(Path(__file__).parents[2])
_DATA_DIR = _ROOT_DIR / "data"
_RAW_DATA_DIR = Path("/projects/sloun/OpTC")
_RAW_SCIVC_CIDS_DIR = Path("/work/gildemeister/SCVIC-CIDS-2021")

def root():
    return _ROOT_DIR

def data():
    return _DATA_DIR

def data_raw(scvic=False):
    if scvic:
        return _RAW_SCIVC_CIDS_DIR
    return _RAW_DATA_DIR

def remove_none(list):
    return [x for x in list if x is not None]

def export_df_value_counts(df, name, stage):
    total = len(df)
    os.makedirs(os.path.join(data(), f"value_counts"), exist_ok=True)
    with open(os.path.join(data(), f"value_counts/value_counts_{name}_s{stage}.txt"), "w") as f:
        for feature in df.columns:
            lines = [feature + "\n\n"]
            value_counts = df[feature].value_counts()
            null_length = total - sum(value_counts)
            lines.append(str(value_counts.head(50)) + "\n")
            lines.append(f"{null_length} null/nan values\n")
            lines.append("\n\n")
            f.writelines(lines)

def trial_name(trial):
    name = f"{trial.trainable_name}_{trial.trial_id}"
    return name

def load_optimization_results(path, metric: str = None, mode: str = None):
    files = glob.glob(path + "/*/*.json")
    json_files = [(files[i], files[i+1]) for i in range(0, len(files), 2)]
    combined_data = []

    if metric is None and mode is None:
        
        for file1, file2 in json_files:
            with open(file1, 'r') as f1, open(file2, 'r') as f2:
                try:
                    json_data1 = json.load(f1)
                except json.decoder.JSONDecodeError as e:
                    print(file1)
                    raise e
                try:
                    json_data2 = json.load(f2)
                except json.decoder.JSONDecodeError as e:
                    print(file2)
                    raise e
                # Combine the two JSON objects into one dictionary
                combined_json = {**json_data1, **json_data2}
                
                # Append the combined dictionary to the list
                combined_data.append(combined_json)

    else:
        for file1, file2 in json_files:
            if not "result" in file1:
                file1, file2 = file2, file1

            with open(file1, 'r') as f1:
                lines = f1.readlines()
            try:
                best_result = json.loads(lines[0])
            except IndexError as e:
                print(file1)
                raise e 
            
            for line in lines:
                line = json.loads(line)
                if mode == "max":
                    if best_result[metric] < line[metric]:
                        best_result = line
                elif mode == "min":
                    if best_result[metric] > line[metric]:
                        best_result = line
                else:
                    raise RuntimeError("Unexpected mode")
            json_data1 = best_result
            with open(file2, 'r') as f2:
                try:
                    json_data2 = json.load(f2)
                except json.decoder.JSONDecodeError as e:
                    print(file2)
                    raise e
            # Combine the two JSON objects into one dictionary
            combined_json = {**json_data1, **json_data2}
            
            # Append the combined dictionary to the list
            combined_data.append(combined_json)

    # Create a DataFrame from the list of combined dictionaries
    combined_df = pd.DataFrame(combined_data)      
    return combined_df

def save_confusion_matrix(confusion_matrix: torch.Tensor | np.ndarray, labels: list[str], path: str | Path, index=None):
    # conver confusion matrix to DataFrame
    if isinstance(confusion_matrix, torch.Tensor):
        df = pd.DataFrame(confusion_matrix.cpu().numpy(), columns=[f"pred_{l}" for l in labels], index=index if index else [f"true_{l}" for l in labels])
    else:
        df = pd.DataFrame(confusion_matrix, columns=[f"pred_{l}" for l in labels], index=index if index else [f"true_{l}" for l in labels])
    # save DataFrame to csv
    df.to_csv(path)

def convert_fig_image(fig: matplotlib.figure.Figure):
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)

    image = PIL.Image.open(buf)
    image = transforms.ToTensor()(image).unsqueeze(0)

    return image
