import pandas as pd
import random
import sys
import argparse
from scipy import stats
from datetime import datetime
import os


def create_timestamp():
    now = datetime.now()                                    # Get the current time
    precise_time = now.strftime("%d-%m-%Y_%H-%M-%S")        # Format the time as "day-month-year_hour-minute-second"
    return precise_time


timestamp = create_timestamp()

# Set seed for random number generator
random.seed(999)

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("dataset_file", type=str, help="Path to the dataset file")
parser.add_argument("split_ratio", type=float, help="Fraction of data to split as dev and test set (e.g., 0.2 for 20%)")
args = parser.parse_args()

# Load the original dataset
original_df = pd.read_csv(args.dataset_file)

# Drop rows with any NaN values
cleaned_df = original_df.dropna().copy()

# Calculate the total size of dev and test sets
split_size = round(len(cleaned_df) * args.split_ratio)

# Randomly select rows for the dev set
dev_set = cleaned_df.sample(n=split_size, random_state=999)

# Remove the selected dev set rows from the original dataset
cleaned_df.drop(dev_set.index, inplace=True)

# Randomly select rows for the test set from the remaining data
test_set = cleaned_df.sample(n=split_size, random_state=999)

# Randomly select rows for the train set from the remaining data
train_set = cleaned_df.drop(test_set.index)

# Create a directory for output files
base_filename = args.dataset_file.rsplit('.', 1)[0]
output_dir = os.path.join(os.getcwd(), f"{base_filename}__{timestamp}")
os.makedirs(output_dir, exist_ok=True)

# Set paths for dev, test, and train set files in the output directory
dev_set_file = os.path.join(output_dir, f"{base_filename}_dev_set.csv")
test_set_file = os.path.join(output_dir, f"{base_filename}_test_set.csv")
train_set_file = os.path.join(output_dir, f"{base_filename}_train_set.csv")

# Save the dev, test, and train sets to separate files
dev_set.to_csv(dev_set_file, index=False)
test_set.to_csv(test_set_file, index=False)
train_set.to_csv(train_set_file, index=False)

# Calculate mean and standard deviation from the train set
train_set_without_last_column = train_set.iloc[:, :-1]          # The labels usually do not get normalized.
mean_train = train_set_without_last_column.mean()
std_train = train_set_without_last_column.std()

# Z-score normalize train, dev, and test sets
"""
Normalize the features via Z-score normalization (subtract mean and divide by the standard deviation).

According to Andrew Ng, you can basically always normalize your features, it rarely does any harm.
Normalization can make features, that are entirely positive and where negative values make no sense
(such as square meters of a house), to contain some negative values. This however does not matter,
it will work anyway. This is because only the ratios within one feature matter, not the absolute values per se.
"""
# Normalize the feature columns (all but the last) in each set
train_set_features_normalized = (train_set.iloc[:, :-1] - mean_train) / std_train
dev_set_features_normalized = (dev_set.iloc[:, :-1] - mean_train) / std_train
test_set_features_normalized = (test_set.iloc[:, :-1] - mean_train) / std_train

# Concatenate the normalized features with the label column for each set
train_set_normalized = pd.concat([train_set_features_normalized, train_set.iloc[:, -1]], axis=1)
dev_set_normalized = pd.concat([dev_set_features_normalized, dev_set.iloc[:, -1]], axis=1)
test_set_normalized = pd.concat([test_set_features_normalized, test_set.iloc[:, -1]], axis=1)

# Set paths for normalized set files in the output directory
train_set_normalized_file = os.path.join(output_dir, f"{base_filename}_train_set_normalized.csv")
dev_set_normalized_file = os.path.join(output_dir, f"{base_filename}_dev_set_normalized.csv")
test_set_normalized_file = os.path.join(output_dir, f"{base_filename}_test_set_normalized.csv")

# Save the files in the specified output directory
train_set_normalized.to_csv(train_set_normalized_file, index=False)
dev_set_normalized.to_csv(dev_set_normalized_file, index=False)
test_set_normalized.to_csv(test_set_normalized_file, index=False)

# Perform a statistical test for similarity between dev and test sets
p_value = stats.ks_2samp(dev_set.values.flatten(), test_set.values.flatten()).pvalue
rounded_p_value = round(p_value, 2)
print("Null hypothesis: Dev and test set are the same. Statistical Test P-Value (Kolmogorov-Smirnov):", rounded_p_value)

# Create the text content
content = f"mean_train:\n{mean_train}\n\nstdv_train:\n{std_train}"

# Set path for the text file in the output directory
text_file = os.path.join(output_dir, "mean_stdv_train_for_normalizing.txt")

# Write the content to the file in the output directory
with open(text_file, "w") as file:
    file.write(content)
