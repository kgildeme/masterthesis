"""
Load all malicious subprocesses
Author: Lars Janssen

A little refactoring done by: Kilian Gildemeister
"""
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parents[2]))

import glob
import gzip
import os
import argparse

from cids.util import misc_funcs as misc
from cids.util import optc_util as util

parser = argparse.ArgumentParser()
parser.add_argument("host", help="Host number", default=None, type=int)
args = parser.parse_args()

host = args.host

malicious_processes_filename = os.path.join(misc.root(), f"preprocessing/labeling/malicious_processes/malicious_processes_{host}")

def read_malicious_process():
    if not os.path.exists(malicious_processes_filename):
        return []
    with open(malicious_processes_filename, "r") as f:
        return [line.strip() for line in f.readlines()]

def append_malicious_process(objectID):
    objectIDs = read_malicious_process()
    if objectID in objectIDs:
        return
    with open(malicious_processes_filename, "a") as f:
        f.write(objectID + "\n")

if __name__ ==  '__main__':
    actorID_queue = read_malicious_process()

    if len(actorID_queue) == 0:
        raise Exception(f"No malicious root processes found for host {host}. The actorIDs of root process must be added to the file {malicious_processes_filename}")

    hostnumber0 = f"{host:03d}"
    hoststring = f'SysClient0{hostnumber0}.systemia.com'
    rounded_host = host - (host % 25) + 1
    aia_host_folder = f"AIA-{rounded_host}-{rounded_host + 24}"
    files = glob.glob(os.path.join(misc.data_raw(), f"ecar/evaluation/*/{aia_host_folder}/*.json.gz"))
    files.sort()
    print(f"Found {len(files)} eval files:\n" + "\n".join(files))
    print(f"{len(actorID_queue)} root processes")

    found_subprocesses = []
    while(len(actorID_queue) > 0):
        print(f"Queue length: {len(actorID_queue)}")
        objectIDs = []
        for filename in files:
            print(f"Processing file {filename}")
            with util.open_gz(filename, "rt") as f:
                for line in f:
                    if util.extract_json_attribute(line, "action") == "CREATE" and util.extract_json_attribute(line, "hostname") == hoststring \
                            and util.extract_json_attribute(line, "object") == "PROCESS"  and util.extract_json_attribute(line, "actorID") in actorID_queue:
                        objectID = util.extract_json_attribute(line, "objectID")
                        if objectID in found_subprocesses or objectID in actorID_queue:
                            print(f"Created subprocess {objectID} at time {util.extract_json_attribute(line, 'timestamp')} (already known)")
                            continue
                        objectIDs.append(objectID)
                        print(f"Created subprocess {objectID} at time {util.extract_json_attribute(line, 'timestamp')}")
        print(f"Found {len(objectIDs)} subprocesses")
        actorID_queue = objectIDs
        found_subprocesses += objectIDs

    print(f"Found {len(found_subprocesses)} new subprocesses in total")
    for objectID in found_subprocesses:
        append_malicious_process(objectID)
