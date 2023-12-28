# Check if the nn_hyperpara_screener image is available locally
if (-not (docker images -q nn_hyperpara_screener 2>$null)) {
    Write-Host "Image not found locally. Pulling from Docker Hub..."
    docker pull thomasrauter/auto_ml-hub:version_1
}

# Define the volume
$volume = "$(Get-Location):/data"

# Running the Docker container with the provided arguments
docker run --rm --name nn_hyperpara_container `
  --volume $volume `
  thomasrauter/auto_ml-hub:version_1 $args

