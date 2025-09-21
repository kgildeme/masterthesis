import subprocess
import argparse
import copy
import os
import yaml
import tempfile
import time
import multiprocessing

def run_trial(exec, config, seed, trial, device, exec_args):
    config['experiment']['seed'] = seed
    config['experiment']['device'] = device
    print(trial)
    config['experiment']['trial'] = f"{config['experiment']['id']}-{trial}"
    config['experiment']['model_path'] = f"{config['experiment']['model_path']}{trial}"
    print(config["experiment"])
    with tempfile.NamedTemporaryFile(delete=False, suffix='.yaml', mode='w') as temp_config_file:
        yaml.dump(config, temp_config_file)
        temp_config_path = temp_config_file.name

    p = subprocess.Popen(["python", exec, "--config", temp_config_path] + exec_args)
    return p, temp_config_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run leaf-x-out experiment with multiple configurations in parallel')
    parser.add_argument('--config', type=str, required=True, help='path to the configuration files')
    parser.add_argument('--exec', type=str, required=True, help='Path to the python script to execute')
    parser.add_argument('--exec_args', nargs=argparse.REMAINDER, help='Additional arguments for the executable')
    args = parser.parse_args()

    configs = []
    if os.path.isdir(args.config):
        for root, _, files in os.walk(args.config):
            for file in files:
                if file.endswith('.yaml'):
                    with open(os.path.join(root, file), 'r') as f:
                        configs.append(yaml.safe_load(f))
    else:
        for config in args.config.split(','):
            with open(config, 'r') as file:
                configs.append(yaml.safe_load(file))

    devices = configs[0]['experiment']['device']
    gpus_per_task = configs[0]['experiment']['gpus_per_trial']
    cpus_per_task = configs[0]['experiment']['cpus_per_trial']
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
        for c, config in enumerate(configs):
            seeds = config['experiment']['seed']
            print(f"Running for {seeds} seed(s)")
            for i, seed in enumerate(seeds):
                if len(processes) < max_concurrent_trials:
                    if len(available_devices) > 0:
                        device = available_devices.pop(0)
                    config_copy = copy.deepcopy(config)
                    p, temp_config_path = run_trial(args.exec, config_copy, seed, i, device, args.exec_args if args.exec_args else [])
                    processes.append((p, temp_config_path, device))

                else:
                    queue.append((c, i, seed))

        while len(processes) != 0:
            for p, temp_config_path, device in processes:
                if p.poll() is not None:
                    processes.remove((p, temp_config_path, device))
                    os.remove(temp_config_path)
                    if gpus_per_task > 0:
                        available_devices.append(device)
                    if queue and len(processes) < max_concurrent_trials:
                        c, i, seed = queue.pop(0)
                        config_copy = copy.deepcopy(configs[c])
                        p, temp_config_path = run_trial(args.exec, config_copy, seed, i, device, args.exec_args if args.exec_args else [])
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
