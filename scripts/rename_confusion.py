import os
import argparse

def rename_confusion_files(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('confussion.csv'):
                old_path = os.path.join(root, file)
                new_file_name = file.replace('confussion', 'confusion')
                new_path = os.path.join(root, new_file_name)
                os.rename(old_path, new_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Rename files ending with confussion.csv to *confusion.csv in a specified directory and its subdirectories.')
    parser.add_argument('directory', type=str, help='The path to the directory to search for files.')
    
    args = parser.parse_args()
    rename_confusion_files(args.directory)