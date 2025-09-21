import subprocess
import argparse
import copy
import os
import yaml
import tempfile
import time
import multiprocessing
from itertools import product

def run_trial(exec, config, trial, device, exec_args):
    config['experiment']['device'] = device
    base_trial = config["experiment"]["seed"] - 1
    config['experiment']['trial'] = f"{config['experiment']['id']}-{trial}"
    config['experiment']['model_path'] = f"{config['experiment']['model_path']}{base_trial}"
    print(config["experiment"])
    with tempfile.NamedTemporaryFile(delete=False, suffix='.yaml', mode='w') as temp_config_file:
        yaml.dump(config, temp_config_file)
        temp_config_path = temp_config_file.name

    p = subprocess.Popen(["python", exec, "--config", temp_config_path] + exec_args)
    return p, temp_config_path

def iterconfigs(config) -> iter:
    def _is_iterable(k ,v):
        print(k)
        return isinstance(v, (list, tuple)) and k not in ["device"]

    iter_keys = []
    iter_values = []
    for section, values in config.items():
        if isinstance(values, dict):
            for k, v in values.items():
                if _is_iterable(k, v):
                    iter_keys.append(f"{section}.{k}")
                    iter_values.append(v)
    print(f"iter_keys: {iter_keys}")

    if not iter_values:
        yield config
        return

    for values in product(*iter_values):
        new_config = copy.deepcopy(config)
        for key, value in zip(iter_keys, values):
            section, param = key.split('.')
            new_config[section][param] = value
        yield new_config

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run leaf-x-out experiment with multiple configurations in parallel')
    parser.add_argument('--config', type=str, required=True, help='path to the configuration files')
    parser.add_argument('--exec', type=str, required=True, help='Path to the python script to execute')
    parser.add_argument('--exec_args', nargs=argparse.REMAINDER, help='Additional arguments for the executable')
    args = parser.parse_args()

    config = args.config
    with open(config, 'r') as file:
        config = yaml.safe_load(file)

    devices = config['experiment']['device']
    gpus_per_task = config['experiment']['gpus_per_trial']
    cpus_per_task = config['experiment']['cpus_per_trial']
    num_cpus = multiprocessing.cpu_count()
    available_devices = []
    if gpus_per_task > 0:
        for device in devices:
            available_devices.extend([device] * int(1 / gpus_per_task))
    
    gpus = len(available_devices) if gpus_per_task > 0 else float('inf')
    max_concurrent_trials = min(gpus, int(num_cpus / cpus_per_task))
    queue = []
    processes = []
    device="BLOCK"
    try:
        for i, cs in enumerate(iterconfigs(config)):
            
            if len(processes) < max_concurrent_trials:
                print(f"start {i}")
                if len(available_devices) > 0:
                    device = available_devices.pop(0)
                p, temp_config_path = run_trial(args.exec, cs, i, device, args.exec_args if args.exec_args else [])
                processes.append((p, temp_config_path, device))

            else:
                queue.append((i, cs))

        while len(processes) != 0:
            for p, temp_config_path, device in processes:
                if p.poll() is not None:
                    processes.remove((p, temp_config_path, device))
                    os.remove(temp_config_path)
                    if gpus_per_task > 0:
                        available_devices.append(device)
                    if queue and len(processes) < max_concurrent_trials:
                        i, cs = queue.pop(0)
                        print(f"start {i}")
                        p, temp_config_path = run_trial(args.exec, cs, i, device, args.exec_args if args.exec_args else [])
                        processes.append((p, temp_config_path, device))
            time.sleep(10)
    except KeyboardInterrupt:
        print("KeyboardInterrupt received, terminating all running subprocesses...")
        for p, temp_config_path, device in processes:
            p.terminate()
            os.remove(temp_config_path)
        for p, temp_config_path, device in processes:
            p.wait()
    except Exception as e:
        print("Unexpected exception received")
        for p, temp_config_path, device in processes:
            p.terminate()
            os.remove(temp_config_path)
        for p, temp_config_path, device in processes:
            p.wait()
        print(e)
