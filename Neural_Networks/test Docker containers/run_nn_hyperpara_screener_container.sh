#!/bin/bash

# Check if the nn_hyperpara_screener image is available locally
if [[ "$(docker images -q nn_hyperpara_screener 2> /dev/null)" == "" ]]; then
  echo "Image not found locally. Pulling from Docker Hub..."
  docker pull thomasrauter/auto_ml-hub:nn_hyperpara_screener:version_1
fi

# Running the Docker container with the provided arguments
docker run --name nn_hyperpara_container \
  -u "$(id -u):$(id -g)" \
  --volume "$(pwd)":/main/input \
  nn_hyperpara_screener "$@"

# Copy the output directory from the container to the current working directory
docker cp nn_hyperpara_container:/main/output .

# Removing the container after the operation
docker rm nn_hyperpara_container
