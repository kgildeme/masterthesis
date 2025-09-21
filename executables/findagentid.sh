#!/bin/bash

# Define an array of host-pid pairs
declare -a host_pid_pairs=(
    "660 880"
    "104 3160"
    "205 5012"
    "321 2980"
    "255 3472"
    "355 1884"
    "503 1472"
    "462 2536"
    "559 1400"
    "419 1700"
    "609 3460"
    "771 4244"
    "955 4760"
    "874 5224"
    "811 3780"
    "10 3584"
    "69 4152"
    "358 2984"
    "618 4060"
    "851 4652"
    # Add more host-pid pairs as needed
)

# Iterate over each host-pid pair and execute the search_pid.py script
for pair in "${host_pid_pairs[@]}"; do
    set -- $pair
    host=$1
    pid=$2
    echo "Running search_pid.py for host $host and pid $pid"
    python3 /opt/gildemeister/gildemeister-implementation/scripts/search_pid.py --host $host --pid $pid
done

echo "All searches completed."