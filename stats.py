import os
import csv

# Define a function to parse the logs and extract relevant information
def parse_log(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Check the first line for correctness
    status = "correct" if "The sparse multiplication result matches the control matrix." in lines[0] else "incorrect"

    # Initialize a dictionary to store stats
    stats = {
        "task_clock": None,
        "context_switches": None,
        "cpu_migrations": None,
        "page_faults": None,
        "cycles": None,
        "stalled_cycles_frontend": None,
        "stalled_cycles_backend": None,
        "instructions": None,
        "branches": None,
        "branch_misses": None,
        "elapsed_time": None,
        "user_time": None,
        "sys_time": None
    }

    # Extract stats from the perf output using pattern matching
    for line in lines:
        if "task-clock" in line:
            stats["task_clock"] = line.split()[0].strip()
        elif "context-switches" in line:
            stats["context_switches"] = line.split()[0].strip()
        elif "cpu-migrations" in line:
            stats["cpu_migrations"] = line.split()[0].strip()
        elif "page-faults" in line:
            stats["page_faults"] = line.split()[0].strip()
        elif "cycles" in line:
            stats["cycles"] = line.split()[0].strip()
        elif "stalled-cycles-frontend" in line:
            stats["stalled_cycles_frontend"] = line.split()[0].strip()
        elif "stalled-cycles-backend" in line:
            stats["stalled_cycles_backend"] = line.split()[0].strip()
        elif "instructions" in line:
            stats["instructions"] = line.split()[0].strip()
        elif "branches" in line:
            stats["branches"] = line.split()[0].strip()
        elif "branch-misses" in line:
            stats["branch_misses"] = line.split()[0].strip()
        elif "seconds time elapsed" in line:
            stats["elapsed_time"] = line.split()[0].strip()
        elif "seconds user" in line:
            stats["user_time"] = line.split()[0].strip()
        elif "seconds sys" in line:
            stats["sys_time"] = line.split()[0].strip()

    # Extract run name (based on the filename), removing the '.log' extension
    run_name = os.path.splitext(os.path.basename(file_path))[0]

    # Return the extracted information as a dictionary
    return {
        "run_name": run_name,
        "status": status,
        **stats
    }

# Path to the directory containing log files
log_dir = 'logs'

# Get a list of all log files in the directory
log_files = [os.path.join(log_dir, f) for f in os.listdir(log_dir) if os.path.isfile(os.path.join(log_dir, f))]

# Prepare a list to store the parsed data
parsed_data = []

# Parse each log file and append the data
for log_file in log_files:
    parsed_data.append(parse_log(log_file))

# Define the CSV file name
csv_file = 'output.csv'

# Define the field names (headers) for the CSV
fieldnames = [
    "run_name", "status", "task_clock", "context_switches", "cpu_migrations", "page_faults",
    "cycles", "stalled_cycles_frontend", "stalled_cycles_backend", "instructions", "branches", 
    "branch_misses", "elapsed_time", "user_time", "sys_time"
]

# Write the parsed data to a CSV file with ';' as the delimiter
with open(csv_file, 'w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=';')
    writer.writeheader()  # Write the header row
    writer.writerows(parsed_data)

print(f"CSV file '{csv_file}' created successfully.")
