#!/bin/bash
set -e

# Load environment variables
if [ -f .env ]; then
  export $(grep -v '^#' .env | xargs)
fi

# Default values
ENV=${1:-production}
VERSION=${2:-latest}
DOCKER_REGISTRY=${DOCKER_REGISTRY:-""}
SERVICE_NAME="moe-inference-system"

# Login to Docker registry if credentials are provided
if [ ! -z "$DOCKER_USERNAME" ] && [ ! -z "$DOCKER_PASSWORD" ]; then
  echo "Logging in to Docker registry..."
  echo "$DOCKER_PASSWORD" | docker login -u "$DOCKER_USERNAME" --password-stdin "$DOCKER_REGISTRY"
fi

# Build and tag the Docker image
IMAGE_TAG="${SERVICE_NAME}:${VERSION}"
if [ ! -z "$DOCKER_REGISTRY" ]; then
  IMAGE_TAG="${DOCKER_REGISTRY}/${SERVICE_NAME}:${VERSION}"
fi

echo "Building Docker image: $IMAGE_TAG"
docker build -t "$IMAGE_TAG" .

# Push to registry if configured
if [ ! -z "$DOCKER_REGISTRY" ]; then
  echo "Pushing image to registry..."
  docker push "$IMAGE_TAG"
fi

# Deploy to Kubernetes (example - customize as needed)
if [ "$ENV" = "production" ]; then
  echo "Deploying to production..."
  # Example: kubectl apply -f kubernetes/production/
elif [ "$ENV" = "staging" ]; then
  echo "Deploying to staging..."
  # Example: kubectl apply -f kubernetes/staging/
else
  echo "Environment '$ENV' not configured for deployment."
fi

echo "Deployment completed successfully!"
