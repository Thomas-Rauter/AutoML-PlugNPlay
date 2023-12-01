import pandas as pd
import sys
import os

# Check if a filename argument is provided
if len(sys.argv) != 2:
    print("Usage: python shuffle_csv.py <filename.csv>")
    sys.exit(1)

# Get the filename from the command-line argument
filename = sys.argv[1]

# Check if the file exists
if not os.path.isfile(filename):
    print(f"The file '{filename}' does not exist.")
    sys.exit(1)

# Read the CSV file into a pandas DataFrame
df = pd.read_csv(filename)

# Shuffle the rows of the DataFrame
shuffled_df = df.sample(frac=1, random_state=42)  # Shuffle with a fixed random seed for reproducibility

# Overwrite the original file with the shuffled data
shuffled_df.to_csv(filename, index=False)

print(f"The file '{filename}' has been shuffled and overwritten.")
