#!/bin/bash

# Assigning command line arguments to variables
local_train_dataset=$1
local_dev_dataset=$2

# Check if the nn_hyperpara_screener image is available locally
if [[ "$(docker images -q nn_hyperpara_screener 2> /dev/null)" == "" ]]; then
  echo "Image not found locally. Pulling from Docker Hub..."
  docker pull thomasrauter/auto_ml-hub:nn_hyperpara_screener
fi

# Running the Docker container with the provided arguments
docker run -it --rm --name nn_hyperpara_container \
  --volume "$(pwd)":/data \
  nn_hyperpara_screener $local_train_dataset $local_dev_dataset
