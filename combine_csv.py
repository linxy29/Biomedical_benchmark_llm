import pandas as pd
import os

# Define the path to the results folder
results_folder = '/home/lighthouse/data/benchmark/MedMCQA/results'

# Get a list of all CSV files in the results folder
csv_files = [os.path.join(results_folder, f) for f in os.listdir(results_folder) if f.endswith('.csv')]

# Initialize an empty list to store DataFrames
dfs = []

# Loop through each CSV file and read it into a DataFrame
for csv_file in csv_files:
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Extract the file name (e.g., 'results_123.csv')
    filename = os.path.basename(csv_file)
    
    # Extract the id from the filename.
    # This assumes the filename format is "results_{id}.csv"
    id_value = filename[len("results_"):-len(".csv")]
    
    # Add the id as a new column to the DataFrame
    df['id'] = id_value
    
    # Append the DataFrame to the list
    dfs.append(df)

# Combine all DataFrames into one
combined_df = pd.concat(dfs, ignore_index=True)

# Save the combined DataFrame to a new CSV file
combined_df.to_csv('/home/lighthouse/data/benchmark/MedMCQA/combined_results.csv', index=False)

print("All CSV files have been combined into 'combined_results.csv'")

