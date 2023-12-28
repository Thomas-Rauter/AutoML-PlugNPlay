#!/bin/bash

# Find all CSV and CBM files in the current directory
mapfile -t csv_files < <(find . -maxdepth 1 -name "*.csv" -print0 | xargs -0 -r -n1 basename)
mapfile -t cbm_files < <(find . -maxdepth 1 -name "*.cbm" -print0 | xargs -0 -r -n1 basename)

# Check the number of CSV and CBM files
num_csv_files=${#csv_files[@]}
num_cbm_files=${#cbm_files[@]}

# Define the Docker image name
docker_image_name="thomasrauter/autocatboost"

# Function to run Docker container for training
run_training() {
    local_csv_file=$1
    echo "Finding optimal model for $local_csv_file"
    docker run --rm -v "$(pwd)":/data $docker_image_name "tune" "/data/$local_csv_file"
}

# Function to run Docker container for prediction
run_prediction() {
    local_csv_file=$1
    local_cbm_file=$2
    echo "Running prediction with $local_csv_file and model $local_cbm_file"
    docker run -v "$(pwd)":/data $docker_image_name "predict" "/data/$local_csv_file" "/data/$local_cbm_file"
}

# Decision logic for running training or prediction
if [[ $num_csv_files -eq 1 && $num_cbm_files -eq 0 ]]; then
    run_training "${csv_files[0]}"
elif [[ $num_csv_files -eq 1 && $num_cbm_files -eq 1 ]]; then
    run_prediction "${csv_files[0]}" "${cbm_files[0]}"
else
    echo "Error: The directory must contain exactly one CSV file and optionally one CBM file."
    exit 1
fi
