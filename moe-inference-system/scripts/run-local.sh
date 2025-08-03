#!/bin/bash
set -e

echo "Building Docker image..."
docker build -t moe-inference-system:local .

echo "\nStarting container..."
docker run -d \
  --name moe-inference-system \
  -p 3000:3000 \
  -e NODE_ENV=production \
  -e PORT=3000 \
  moe-inference-system:local

echo "\nContainer is running! Access the application at http://localhost:3000"
echo "To view logs, run: docker logs -f moe-inference-system"
echo "To stop the container, run: docker stop moe-inference-system && docker rm moe-inference-system"
