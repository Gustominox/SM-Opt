import os
import csv
import re  

# List of directories to process
logs_root_dirs = [
    # './logs-64',  
    # './logs-1024',
    # './logs-2048',
    './logs-4096'
]

# Function to process a single file and extract the timeInMsec
def extract_time_from_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        print(lines)
        
        if len(lines) > 1:
            try:
                # The time in msec is assumed to be in the first field of the second row
                time_in_msec = lines[1].split(',')[0]
                return time_in_msec
            except IndexError:
                print(f"Error processing file: {file_path}, line 2 does not have enough data.")
                return None
        else:
            print(f"Error: {file_path} does not have enough lines.")
            return None

# Function to extract the number from the directory name
def extract_size_from_dir(directory):
    match = re.search(r'(\d+)', directory)  # Look for one or more digits
    if match:
        return match.group(1)  # Return the number as a string
    else:
        return 'Unknown'  # If no number is found, return 'Unknown'

# Create a new CSV to store the sorted output
output_csv = 'output_sorted.csv'
with open(output_csv, mode='w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Filename', 'Size', 'timeInSec', 'timeInMsec'])  # Write header

    # Iterate over each specified root directory
    for root_dir in logs_root_dirs:
        for dirpath, dirnames, filenames in os.walk(root_dir):
            # List to collect rows for each directory
            rows = []

            for filename in filenames:
                if filename.endswith('.csv'):
                    # Get the full path of the file
                    file_path = os.path.join(dirpath, filename)

                    # Extract the directory and size
                    directory = os.path.basename(dirpath)
                    size = extract_size_from_dir(os.path.basename(dirpath))

                    time_in_msec = extract_time_from_file(file_path)
                    try:
                        time_in_sec = float(time_in_msec) / 1000.0
                    except:
                        time_in_sec = None

                    # Add the row to the list for this directory
                    rows.append([filename, size, time_in_sec, time_in_msec])

            # Sort the rows by time_in_sec (ignoring rows with None in time_in_sec)
            rows.sort(key=lambda row: (row[2] if row[2] is not None else float('inf')))

            # Write the sorted rows for this directory to the output CSV
            writer.writerows(rows)
            
print(f"CSV with extracted data has been saved as {output_csv}")
