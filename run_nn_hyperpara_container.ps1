# PowerShell script to run a Docker container

# Assigning command line arguments to variables
$trainDataset = $args[0]
$devDataset = $args[1]
$imageName = "nn_hyperpara_screener"

# Check if Docker is running
try {
    docker info > $null
}
catch {
    Write-Error "Docker is not running. Please start Docker and try again."
    exit 1
}

# Check if the image is available locally
$imageExists = docker images -q $imageName
if (-not $imageExists) {
    Write-Host "Image not found locally. Pulling from Docker Hub..."
    docker pull $imageName
}

# Run the Docker container
docker run -it --rm `
    --name "nn_hyperpara_container" `
    --volume "${PWD}:/data" `
    $imageName $trainDataset $devDataset
