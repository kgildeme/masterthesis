from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parents[1]))

import argparse
import glob
import os
import json

from cids.util import misc_funcs as misc
from cids.util import optc_util as util

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", help="Host number", default=None, type=int)
    parser.add_argument("--pid", help="Process ID which should be filtered", default=None, type=int)
    args = parser.parse_args()

    host = args.host
    pid = args.pid

    hostnumber0 = f"{host:03d}"
    hoststring = f'SysClient0{hostnumber0}.systemia.com'
    rounded_host = host - (host % 25) + 1
    aia_host_folder = f"AIA-{rounded_host}-{rounded_host + 24}"
    files = glob.glob(os.path.join(misc.data_raw(), f"ecar/evaluation/*/{aia_host_folder}/*.json.gz"))
    files.sort()
    print(f"Found {len(files)} eval files:\n" + "\n".join(files))
    actorIDS = {}
    for filename in files:
        print(f"Processing file {filename}")
        with util.open_gz(filename, "rt") as f:
            for line in f:
                if util.extract_json_attribute(line, "hostname") == hoststring and int(util.extract_json_attribute(line, "pid", nostring=True)) == pid:
                    actor = util.extract_json_attribute(line, "actorID")
                    if actor not in actorIDS:
                        print(f"Add actor {actor}")
                        actorIDS[actor] = [(util.extract_json_attribute(line, "object"), util.extract_json_attribute(line, "objectID"), util.extract_json_attribute(line, "action"), util.extract_json_attribute(line, "pid", nostring=True), util.extract_json_attribute(line, "timestamp"))]
                    else:
                        actorIDS[actor].append((util.extract_json_attribute(line, "object"), util.extract_json_attribute(line, "objectID"), util.extract_json_attribute(line, "action"), util.extract_json_attribute(line, "pid", nostring=True), util.extract_json_attribute(line, "timestamp")))

    
    with open(os.path.join(misc.data(), f"tmp/pid_to_actorid/host{host}_pid{pid}.json"), 'w') as f:
        json.dump(actorIDS, f, indent=2)
    print("Search done")

