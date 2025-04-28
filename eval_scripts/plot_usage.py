# Script to visualize GPU and CPU/Memory logs

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import re
from datetime import datetime, timedelta

# --- Configuration ---
gpu_log_file = './logs/gpu_usage_64119179.csv'
sar_log_file = './logs/cpu_mem_usage_64119179.log'
output_plot_file = './logs/resource_usage_plot.png'

# --- GPU Log Parsing ---
print(f"Reading GPU log: {gpu_log_file}")
try:
    gpu_df = pd.read_csv(gpu_log_file, skipinitialspace=True)
    gpu_df.columns = [
        'timestamp', 'utilization.gpu [%]', 'utilization.memory [%]',
        'memory.used [MiB]', 'memory.free [MiB]', 'memory.total [MiB]'
    ]
    gpu_df['timestamp'] = pd.to_datetime(gpu_df['timestamp'], format='%Y/%m/%d %H:%M:%S.%f')
    mem_cols = ['memory.used [MiB]', 'memory.free [MiB]', 'memory.total [MiB]']
    for col in mem_cols:
        if col in gpu_df.columns and gpu_df[col].dtype == 'object':
             gpu_df[col] = pd.to_numeric(gpu_df[col].str.replace(r'\s*MiB', '', regex=True), errors='coerce')
        elif col in gpu_df.columns:
             gpu_df[col] = pd.to_numeric(gpu_df[col], errors='coerce')
    gpu_df.rename(columns={
        'memory.used [MiB]': 'memory.used_mib',
        'memory.free [MiB]': 'memory.free_mib',
        'memory.total [MiB]': 'memory.total_mib'
    }, inplace=True)
    start_time_gpu = gpu_df['timestamp'].iloc[0]
    gpu_df['elapsed_seconds'] = (gpu_df['timestamp'] - start_time_gpu).dt.total_seconds()
    print(f"GPU log read successfully. Start time: {start_time_gpu}")
except FileNotFoundError:
    print(f"Error: GPU log file not found at {gpu_log_file}")
    gpu_df = None
except Exception as e:
    print(f"Error reading GPU log: {e}")
    gpu_df = None

# --- SAR Log Parsing (Handles -P ALL) ---
print(f"Reading SAR log: {sar_log_file}")
per_cpu_data = {} # Dictionary to hold lists of records for each CPU ID ('all', '0', '1', ...)
mem_data = []
sar_start_date_str = None
first_timestamp_str = None

try:
    with open(sar_log_file, 'r') as f:
        lines = f.readlines()

    # Extract date from header
    match = re.search(r'(\d{2}/\d{2}/\d{4})', lines[0])
    if match:
        sar_start_date_str = match.group(1)
        print(f"SAR log date extracted: {sar_start_date_str}")
    elif gpu_df is not None and not gpu_df.empty:
         sar_start_date_str = start_time_gpu.strftime('%m/%d/%Y')
         print(f"Using date from GPU log: {sar_start_date_str}")
    else:
         print("Warning: Could not determine date for SAR log.")

    current_time_dt = None
    for line in lines:
        line = line.strip()
        if not line or 'Average:' in line or 'Linux' in line or '(int)' in line:
            continue

        parts = re.split(r'\s+', line)

        # Attempt to parse timestamp
        try:
            time_str_full = f"{parts[0]} {parts[1]}"
            if sar_start_date_str:
                current_time_dt = datetime.strptime(f"{sar_start_date_str} {time_str_full}", '%m/%d/%Y %I:%M:%S %p')
                if first_timestamp_str is None:
                    first_timestamp_str = current_time_dt
            else: current_time_dt = None # Cannot proceed without date
        except (ValueError, IndexError):
            continue # Not a line with a parsable timestamp

        if current_time_dt is None: continue

        # Check for per-CPU line (output from `sar -P ALL -u`)
        if len(parts) >= 10 and (parts[2].isdigit() or parts[2] == 'all'):
            cpu_id_str = parts[2]
            try:
                user = float(parts[3])
                system = float(parts[5]) # Note: %system is often the 6th field (index 5)
                iowait = float(parts[6])
                idle = float(parts[9]) # Note: %idle is often the 10th field (index 9)
                total_util = user + system # A common way to calculate active %

                cpu_record = {
                    'timestamp': current_time_dt, '%user': user, '%system': system,
                    '%idle': idle, '%iowait': iowait, 'total_util': total_util
                }
                if cpu_id_str not in per_cpu_data:
                    per_cpu_data[cpu_id_str] = []
                per_cpu_data[cpu_id_str].append(cpu_record)
            except (ValueError, IndexError, TypeError) as e:
                 print(f"Debug SAR Parse Error (CPU line '{cpu_id_str}'): {e}")
                 print(f"  Problematic line parts: {parts}")
                 pass
        # Check for memory line (output from `sar -r`) - keep logic even if not plotting
        elif len(parts) >= 10 and parts[1].replace('.', '', 1).isdigit():
             try:
                 memused_gb = float(parts[2]) / (1024**2) # kbmemused is usually 3rd field (index 2)
                 mem_data.append({'timestamp': current_time_dt, 'memused_gb': memused_gb})
             except (ValueError, IndexError, TypeError) as e:
                 print(f"Debug SAR Parse Error (Memory line): {e}")
                 print(f"  Problematic line parts: {parts}")
                 pass

    # Convert collected data to DataFrames
    cpu_dfs = {}
    for cpu_id, data_list in per_cpu_data.items():
        if data_list:
             df = pd.DataFrame(data_list)
             if not df.empty:
                  start_time = df['timestamp'].iloc[0]
                  df['elapsed_seconds'] = (df['timestamp'] - start_time).dt.total_seconds()
                  cpu_dfs[cpu_id] = df
                  print(f"CPU SAR data for CPU '{cpu_id}' parsed successfully.")
             else: print(f"Warning: No valid data parsed for CPU '{cpu_id}'.")
        else: print(f"Warning: No data found for CPU '{cpu_id}'.")

    # Assign 'all' dataframe for overall plot
    cpu_df = cpu_dfs.get('all', pd.DataFrame())
    if cpu_df.empty: print("Warning: Could not find or parse 'all' CPU data from SAR log.")

    # Create memory dataframe (unused in plot but parsed)
    mem_df = pd.DataFrame(mem_data)
    if not mem_df.empty: print(f"Memory SAR data parsed successfully.")

except FileNotFoundError:
    print(f"Error: SAR log file not found at {sar_log_file}")
    cpu_df, cpu_dfs, mem_df = pd.DataFrame(), {}, pd.DataFrame()
except Exception as e:
    print(f"Error reading SAR log: {e}")
    cpu_df, cpu_dfs, mem_df = pd.DataFrame(), {}, pd.DataFrame()


# --- Plotting (2 Plots: GPU, CPU) ---
if gpu_df is not None or not cpu_df.empty: # Check if we have GPU or at least 'all' CPU data
    print("Generating plots...")
    fig, axes = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
    ax_gpu_util = axes[0]
    ax_gpu_mem = ax_gpu_util.twinx()
    ax_cpu = axes[1]
    colors = plt.cm.tab10.colors
    cpu_colors = plt.cm.viridis # Colormap for individual cores

    # Plot GPU
    if gpu_df is not None and not gpu_df.empty:
        ln1 = ax_gpu_util.plot(gpu_df['timestamp'], gpu_df['utilization.gpu [%]'], label='GPU Utilization (%)', color=colors[0])
        ln2 = ax_gpu_mem.plot(gpu_df['timestamp'], gpu_df['memory.used_mib'], label='GPU Memory Used (MiB)', color=colors[1], linestyle='--')
        ax_gpu_util.set_ylabel('GPU Utilization (%)', color=colors[0])
        ax_gpu_mem.set_ylabel('GPU Memory Used (MiB)', color=colors[1])
        ax_gpu_util.tick_params(axis='y', labelcolor=colors[0])
        ax_gpu_mem.tick_params(axis='y', labelcolor=colors[1])
        ax_gpu_util.set_ylim(0, 105)
        ax_gpu_util.set_yticks([0, 25, 50, 75, 100])
        gpu_mem_total = gpu_df['memory.total_mib'].dropna().iloc[0] if not gpu_df['memory.total_mib'].dropna().empty else 46068
        ax_gpu_mem.set_ylim(0, gpu_mem_total * 1.05)
        ax_gpu_util.set_title('GPU Usage Over Time')
        ax_gpu_util.grid(True, axis='y', linestyle=':')
        lns = ln1 + ln2
        labs = [l.get_label() for l in lns]
        ax_gpu_util.legend(lns, labs, loc='upper left')
    else:
         ax_gpu_util.set_title('GPU Usage Over Time (No Data)')

    # Plot CPU (Overall 'all' + first 8 cores)
    if not cpu_df.empty: # Plot overall if 'all' data exists
        ax_cpu.plot(cpu_df['timestamp'], cpu_df['total_util'], label='Total CPU Util (All Cores)', color=colors[2], linewidth=2)

        # Plot individual cores (0-7)
        num_cores_to_plot = 8
        for i in range(num_cores_to_plot):
             cpu_id_str = str(i)
             if cpu_id_str in cpu_dfs and not cpu_dfs[cpu_id_str].empty:
                  core_df = cpu_dfs[cpu_id_str]
                  ax_cpu.plot(core_df['timestamp'], core_df['total_util'],
                               label=f'CPU {cpu_id_str} % Util',
                               color=cpu_colors(i / num_cores_to_plot),
                               linestyle='-', linewidth=0.8, alpha=0.8)

        ax_cpu.set_ylabel('CPU Utilization (%)')
        ax_cpu.set_ylim(0, 105)
        ax_cpu.set_title(f'Overall and Individual CPU Core (0-{num_cores_to_plot-1}) Utilization')
        ax_cpu.grid(True, linestyle=':')
        ax_cpu.legend(loc='upper left', fontsize='small') # Show legend for all lines plotted on ax_cpu
        ax_cpu.set_xlabel('Time')
        ax_cpu.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        plt.xticks(rotation=45)
    else:
         ax_cpu.set_title('CPU Usage Over Time (No Data or Failed to Parse \'all\')')
         ax_cpu.set_xlabel('Time')

    fig.suptitle(f'Resource Usage Logs\nGPU: {gpu_log_file}\nSAR: {sar_log_file}', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if output_plot_file:
        print(f"Saving plot to {output_plot_file}")
        plt.savefig(output_plot_file, dpi=150)

    print("Displaying plot...")
    plt.show()

else:
    print("Could not read GPU or 'all' CPU log files. No plot generated.")
