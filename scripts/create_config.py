from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parents[1]))
import yaml
import os

import math
from cids.util import misc_funcs as misc

def load_yaml(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

def save_yaml(data, file_path):
    with open(file_path, 'w') as file:
        yaml.safe_dump(data, file)

def create_configs(mask_path, changes):
    mask = load_yaml(mask_path)
    base_dir = os.path.dirname(mask_path)
    
    keys = [change['key'] for change in changes]
    values_list = list(zip(*[change['values'] for change in changes]))

    for i, values in enumerate(values_list):
        new_config = mask.copy()
        for key, value in zip(keys, values):
            keys_split = key.split('.')
            d = new_config
            for k in keys_split[:-1]:
                d = d.setdefault(k, {})
            d[keys_split[-1]] = value
        new_file_name = os.path.split(mask_path)[1].split(".")[0] + f"-{i}.yaml"
        new_file_path = os.path.join(base_dir, new_file_name)
        save_yaml(new_config, new_file_path)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python create_config.py <path_to_mask_yaml>")
        sys.exit(1)
    
    mask_path = sys.argv[1]
    task_per_gpu = 3
    changes = [
        {'key': 'experiment.seed', 'values': range(1, 11)},
        {'key': 'experiment.device', 'values': [f"cuda:{(i // task_per_gpu) % 3}" for i in range(10)]},
        {'key': 'experiment.id', 'values': [f"DoSDDoS-{i}" for i in range(10)]},
    ]
    
    create_configs(mask_path, changes)