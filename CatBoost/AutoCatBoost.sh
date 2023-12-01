#!/bin/bash

# Find all CSV and CBM files in the current directory
mapfile -t csv_files < <(find . -maxdepth 1 -name "*.csv" -print0 | xargs -0 -r -n1 basename)
mapfile -t cbm_files < <(find . -maxdepth 1 -name "*.cbm" -print0 | xargs -0 -r -n1 basename)

# Check the number of CSV and CBM files
num_csv_files=${#csv_files[@]}
num_cbm_files=${#cbm_files[@]}


# Define the Docker image name and tag
image_name="thomasrauter/autocatboost"
image_tag="1.0"

# Check if the image exists locally
if docker image inspect "${image_name}:${image_tag}" &> /dev/null; then
    echo "Docker image '${image_name}:${image_tag}' is already available locally."
else
    echo "Docker image '${image_name}:${image_tag}' not found locally. Pulling from Docker Hub..."
    docker pull "${image_name}:${image_tag}"
fi


# Function to run Docker container for training
run_training() {
    local_csv_file=$1
    echo "Finding optimal model for $local_csv_file"

    docker run --name autocatboost \
      -v "$(pwd)/$local_csv_file":/app/input/train_data.csv \
      thomasrauter/autocatboost:1.0 "tune"

    docker cp autocatboost:/app/output/best_catboost_model.cbm ./best_catboost_model.cbm
    docker cp autocatboost:/app/output/best_catboost_model_report.pdf ./best_catboost_model_report.pdf
    docker rm autocatboost

}


# Function to run Docker container for prediction
run_prediction() {
    local_csv_file=$1
    local_cbm_file=$2
    echo "Running prediction with $local_csv_file and model $local_cbm_file"

    docker run --name autocatboost \
      -v "$(pwd)/$local_csv_file":/app/input/unseen_data.csv \
      -v "$(pwd)/$local_cbm_file":/app/input/catboost_model.cbm \
      thomasrauter/autocatboost:1.0 "predict"

    docker cp autocatboost:/app/output/predictions_unseen_data.csv ./predictions_unseen_data.csv
    docker rm autocatboost

  # Check if the variable local_csv_file is not empty
  if [ -z "$local_csv_file" ]; then
      echo "Error: local_csv_file variable is not set."
      exit 1
  fi

  # Delete the file with the name equal to $local_csv_file
  rm -f "./$local_csv_file"

  # Rename predictions_unseen_data.csv to $local_csv_file
  mv -f "predictions_unseen_data.csv" "./$local_csv_file"

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
