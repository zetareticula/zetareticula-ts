# Mixture of Experts Inference System

A high-performance, scalable Mixture of Experts (MoE) inference system with dynamic bit-depth optimization.

## Features

- Dynamic expert routing with softmax gating
- Adaptive bit-depth optimization based on hardware class
- Production-ready API with input validation and error handling
- Comprehensive logging and monitoring
- Containerized deployment with Docker
- Kubernetes-ready configuration

## Prerequisites

- Node.js 18+
- npm 9+
- Docker 20.10+
- Kubernetes (for production deployment)

## Getting Started

### Local Development

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/zetareticula-ts.git
   cd zetareticula-ts/moe-inference-system
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```

4. Start the development server:
   ```bash
   npm run dev
   ```

### Using Docker

```bash
# Build the Docker image
docker-compose build

# Start the services
docker-compose up -d

# View logs
docker-compose logs -f
```

## API Endpoints

- `POST /infer` - Perform inference using the MoE model
- `GET /health` - Health check endpoint

### Example Request

```bash
curl -X POST http://localhost:3000/infer \
  -H "Content-Type: application/json" \
  -d '{
    "features": [0.1, 0.2, 0.3, 0.4],
    "hardwareClass": "gpu"
  }'
```

## Configuration

Environment variables can be configured in the `.env` file:

```env
PORT=3000
NODE_ENV=development
LOG_LEVEL=debug
```

## Testing

```bash
# Run tests
npm test

# Run tests with coverage
npm run test:coverage
```

## Deployment

### Kubernetes

Deploy to Kubernetes using the provided manifests:

```bash
kubectl apply -f kubernetes/
```

## Monitoring

The application exposes Prometheus metrics at `/metrics` when the `ENABLE_METRICS` environment variable is set to `true`.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
