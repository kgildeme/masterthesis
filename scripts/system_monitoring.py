import psutil
import os
import signal
import time
import argparse
from datetime import datetime

from matplotlib import pyplot as plt

# Set the memory usage thresholds
MEMORY_THRESHOLD_TERM = 90  # in percentage
MEMORY_THRESHOLD_KILL = 95  # in percentage

def get_process_with_highest_memory():
    """Returns the process object with the highest memory consumption from the current user."""
    current_user = os.getuid()  # Get current user's UID
    processes = [(proc, proc.memory_percent()) for proc in psutil.process_iter(['pid', 'name', 'memory_percent']) if proc.uids().real == current_user]
    
    if not processes:
        return None  # Return None if no processes are found

    highest_memory_process = max(processes, key=lambda p: p[1])
    return highest_memory_process[0]  # Return the process object


def check_and_handle_memory():
    """Check system memory usage and handle processes if thresholds are exceeded."""
    # Get system-wide memory usage
    memory = psutil.virtual_memory()
    memory_percent_used = memory.percent

    # print(f"Memory usage: {memory_percent_used}%")

    if memory_percent_used >= MEMORY_THRESHOLD_TERM:
        # Find the process with the highest memory consumption
        process = get_process_with_highest_memory()
        print(f"Process using most memory: {process.name()} (PID: {process.pid}), Memory: {process.memory_percent()}%")
        
        if memory_percent_used >= MEMORY_THRESHOLD_KILL:
            # Kill the process if memory usage exceeds 98%
            print(f"Memory usage exceeds {MEMORY_THRESHOLD_KILL}%. Sending SIGKILL to PID: {process.pid}")
            os.kill(process.pid, signal.SIGKILL)
        elif memory_percent_used >= MEMORY_THRESHOLD_TERM:
            # Send SIGTERM to the process if memory usage exceeds 95%
            print(f"Memory usage exceeds {MEMORY_THRESHOLD_TERM}%. Sending SIGTERM to PID: {process.pid}")
            os.kill(process.pid, signal.SIGTERM)

def save_memory_plot(memory_usage, interval, plot_dir):
    """Save the memory usage plot to the dedicated directory."""
    plt.figure(figsize=(10, 6))
    plt.plot(memory_usage, label="Memory Usage (%)", color='b', marker='o')
    plt.xlabel(f"Time (every {interval} seconds)")
    plt.ylabel("Memory Usage (%)")
    plt.title("System Memory Usage Over Time")
    plt.legend()
    plt.grid(True)

    # Create filename with timestamp and measurement count
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plot_filename = os.path.join(plot_dir, f"memory_usage_{timestamp}.png")
    
    # Save the plot as a PNG file
    plt.savefig(plot_filename)
    plt.close()  # Close the plot to free memory
    print(f"Plot saved to: {plot_filename}")

if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser(description="Monitor system memory and terminate processes if thresholds are exceeded.")
    parser.add_argument("--interval", type=int, default=5, help="The interval in seconds between memory checks (default is 5 seconds).")
    parser.add_argument("--plot_every", type=int, default=10, help="Plot memory usage after every X measurements.")
    parser.add_argument("--plot_dir", type=str, default="memory_plots", help="Directory to save memory usage plots.")
    
    args = parser.parse_args()
    check_interval = args.interval
    plot_every = args.plot_every
    plot_dir = args.plot_dir

    print(f"Start Monitoring every {check_interval}s and plotting every {plot_every} measurements")
    # Create the plot directory if it does not exist
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
        print(f"Created directory: {plot_dir}")

    memory_usage = []  # To store memory usage over time
    measurement_count = 0  # To count measurements

    # Continuous memory monitoring
    try:
        while True:
            check_and_handle_memory()

            # Store current memory usage
            memory = psutil.virtual_memory()
            memory_usage.append(memory.percent)
            measurement_count += 1

            # Save memory usage plot every x measurements
            if measurement_count % plot_every == 0:
                print(f"Saving memory usage plot after {measurement_count} measurements...")
                print(f"Current usage is: {memory_usage[-1]}")
                save_memory_plot(memory_usage, check_interval, plot_dir)
                memory_usage = []  # To store memory usage over time
                measurement_count = 0  # To count measurements

            time.sleep(check_interval)  # Check every user-defined interval
    except KeyboardInterrupt:
        print("Monitoring stopped.")