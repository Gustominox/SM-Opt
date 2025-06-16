import pandas as pd

# Load the CSV file
file_path = 'output_sorted_4096.csv'  # Change this to your actual CSV file path
data = pd.read_csv(file_path)

# Extract the implementation name by removing the run number (the part after the last '-')
data['Implementation'] = data['Filename'].str.rsplit('-', n=1).str[:-1].str.join('-')

# Group by implementation and calculate the required statistics
statistics_per_implementation = data.groupby('Implementation')['timeInSec'].describe()

# Add the number of runs (which is always 5 in this case)
statistics_per_implementation['number_of_runs'] = 5

# Reorder the columns to match the requested format: mean, std, max, min, number_of_runs
statistics_per_implementation = statistics_per_implementation[['mean', 'std', 'max', 'min', 'number_of_runs']]

# Print the results
print(statistics_per_implementation)
