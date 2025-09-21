"""
Read and process config files in the .yaml format
"""
import os
from pathlib import Path
import yaml

from ray import tune

SUPPORTED_DISTRIBUTIONS = ["randint", "uniform", "loguniform", "choice", "grid"]

def _convert_list(config: dict):
    for key, item in config.items():
        if isinstance(item, list):
            if item[-1] in SUPPORTED_DISTRIBUTIONS:
                if item[-1] == "choice":
                    config[key] = (item[:-1], item[-1])
                elif item[-1] == "grid":
                    config[key] = (item[:-1], item[-1])
                else:
                    config[key] = tuple(item)
        elif isinstance(item, dict):
            config[key] = _convert_list(item)
    
    return config

def read(path: str | Path, convert_tuple: bool = False):
    if not os.path.exists(path):
        raise ValueError(f"Path {path} does not exit")
    
    with open(path, "r") as f:
        config = yaml.safe_load(f)

    if convert_tuple:
       config = _convert_list(config)

    return config

def transform_ray_hp_config(hp_config: dict):
    config = {}
    for key, item in hp_config.items():
        if isinstance(item, dict):
            config[key] = transform_ray_hp_config(item)
        elif isinstance(item, tuple):

            if item[-1] == "randint":
                config[key] = tune.randint(item[0], item[1])
            elif item[-1] == "uniform":
                config[key] = tune.uniform(item[0], item[1])
            elif item[-1] == "loguniform":
                config[key] = tune.loguniform(item[0], item[1])
            elif item[-1] == "choice":
                config[key] = tune.choice(item[0])
            elif item[-1] == "grid":
                config[key] = tune.grid_search(item[0])
            else:
                raise ValueError(f"Distribution {item[2]} not supported in model_config. Only {SUPPORTED_DISTRIBUTIONS} supported")
        
        else: 
            config[key] = item

    return config
