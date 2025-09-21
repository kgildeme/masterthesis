from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parents[1]))

import argparse
import os
import shutil

from cids.util import misc_funcs as misc

def backup_directories(target_path):
    # List of directories to copy
    directories_to_copy = ['results', 'config', 'models']


    # Check if the target path exists; if not, create it
    os.makedirs(target_path, exist_ok=True)
    # Remove everything in the target path first
    for item in os.listdir(target_path):
        item_path = os.path.join(target_path, item)
        if os.path.isdir(item_path):
            shutil.rmtree(item_path)
        else:
            os.remove(item_path)
    for directory in directories_to_copy:
        source_dir = os.path.join(misc.root(), directory)
        destination_dir = os.path.join(target_path, directory)

        if os.path.exists(source_dir):
            # Copy the directory and overwrite existing files
            shutil.copytree(source_dir, destination_dir, dirs_exist_ok=True)
            print(f"Copied '{source_dir}' to '{destination_dir}'")
        else:
            print(f"Directory '{source_dir}' does not exist. Skipping.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Backup specified directories to a target path.')
    parser.add_argument('target_path', type=str, help='The target path where directories will be copied.')

    args = parser.parse_args()
    
    backup_directories(args.target_path)