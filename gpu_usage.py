import time
import GPUtil
import pandas as pd
from datetime import datetime
import subprocess

def monitor_gpu_usage(process, data_filename="gpu_usage_data.csv", stats_filename="gpu_usage_stats.txt"):
    # Initialize an empty list to collect data
    data = []

    # Record the start time of the process
    start_time = datetime.now()

    try:
        while process.poll() is None:  # Check if the process is still running
            current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            # GPU usage
            gpus = GPUtil.getGPUs()
            for gpu in gpus:
                gpu_id = gpu.id
                gpu_name = gpu.name
                memory_utilization = gpu.memoryUtil * gpu.memoryTotal  # Absolute memory utilization in MB

                # Append data to the list
                data.append({
                    'Timestamp': current_time,
                    'GPU_ID': gpu_id,
                    'GPU_Name': gpu_name,
                    'Memory_Utilization_MB': memory_utilization
                })

            # Sleep for a while before next log
            time.sleep(1)
    
    except KeyboardInterrupt:
        pass
    
    finally:
        # Record the end time of the process
        end_time = datetime.now()
        elapsed_time = end_time - start_time

        # Convert collected data to DataFrame
        df = pd.DataFrame(data)

        # Save the raw data to a CSV file
        df.to_csv(data_filename, index=False)
        print(f"\nDetailed GPU usage data saved to {data_filename}")

        # Prepare the statistics summary
        stats = []
        stats.append(f"Monitoring stopped. GPU Memory Utilization Statistics:\n")
        if not df.empty:
            for gpu_id in df['GPU_ID'].unique():
                gpu_df = df[df['GPU_ID'] == gpu_id]
                stats.append(f"\nGPU {gpu_id} - {gpu_df['GPU_Name'].iloc[0]}:\n")
                stats.append(gpu_df['Memory_Utilization_MB'].describe().to_string())
        else:
            stats.append("No data collected.\n")
        
        stats.append(f"\nProcess ran for: {elapsed_time}\n")

        # Save the statistics to a text file
        with open(stats_filename, 'w') as f:
            f.write("\n".join(stats))
        print(f"Summary statistics saved to {stats_filename}")

if __name__ == "__main__":
    # Command to start the other process
    command = "python evaluation_scripts/test_stihl_d435i.py --datapath /home/user/Desktop/2024-04-11/tum/d435i --outputpath output/gpu_test_mono --disable_vis"

    # Start the other process
    process = subprocess.Popen(command.split())

    print(f"Started process '{' '.join(command)}' with PID {process.pid}")

    # Monitor GPU usage and save results
    monitor_gpu_usage(process=process, data_filename="droid-slam-mono_gpu_usage.csv", stats_filename="droid-slam-mono_gpu_usage.txt")