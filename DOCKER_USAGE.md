# Docker Containerization Guide

This guide explains how to run the Solar Prediction MLOps pipeline using Docker containers.

## Architecture Overview

We have **6 containers**:

1. **Prefect Server Container** (`solar-prefect-server`)
   - Runs Prefect server and UI
   - Exposes Prefect UI on port 4200
   - Persistent data storage within container
   - Accessible via browser at http://localhost:4200

2. **ML Pipeline Container** (`solar-ml-pipeline`)
   - Runs Prefect worker only
   - Connects to Prefect server container
   - Executes ML pipeline training
   - Registers models to MLflow

3. **API Container** (`solar-api-service`)
   - Runs FastAPI service for predictions
   - Fetches models from MLflow
   - Logs predictions to Supabase
   - Exposes API on port 8000

4. **Monitoring Container** (`solar-monitoring`)
   - Runs Evidently AI drift monitoring
   - Exposes metrics on port 8080
   - Monitors data drift and model performance
   - Accessible via http://localhost:8080/metrics

5. **Prometheus Container** (`solar-prometheus`)
   - Collects metrics from monitoring container
   - Exposes monitoring UI on port 9090
   - Persistent metrics storage
   - Accessible via browser at http://localhost:9090

6. **Grafana Container** (`solar-grafana`)
   - Visualizes metrics and creates dashboards
   - Exposes dashboard UI on port 3000
   - Persistent dashboard storage
   - Accessible via browser at http://localhost:3000
   - Default credentials: admin/admin

## Prerequisites

- Docker and Docker Compose installed
- Access to MLflow server (EC2)
- Supabase credentials
- AWS credentials for S3 access

## Quick Start

### 1. Set up Environment Variables

```bash
# Copy the example environment file
cp docker/env.example .env

# Edit .env with your actual credentials
nano .env
```

Fill in your actual values:
- `MLFLOW_TRACKING_URI`: Your MLflow server URL
- `SUPABASE_URL` and `SUPABASE_KEY`: Your Supabase credentials
- `AWS_*`: Your AWS credentials
- `S3_BUCKET_NAME`: Your S3 bucket name

### 2. Build and Run Containers

```bash
# Build and start both containers
docker-compose up --build

# Or run in background
docker-compose up --build -d
```

### 3. Monitor the Startup Process

The containers will start in this order:

1. **Prefect Server Container** starts:
   - Prefect server starts on port 4200
   - Prefect UI becomes accessible

2. **ML Pipeline Container** starts:
   - Connects to Prefect server container
   - Worker "ml-pipeline-worker" starts
   - ML pipeline runs automatically
   - Model gets registered to MLflow

3. **API Container** starts:
   - Waits for model to be available in MLflow
   - FastAPI service starts on port 8000

### 4. Access Services

- **Prefect UI**: http://localhost:4200
- **API Documentation**: http://localhost:8000/docs
- **API Health Check**: http://localhost:8000/
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (admin/admin)

## Container Management

### View Logs

```bash
# View all container logs
docker-compose logs

# View specific container logs
docker-compose logs ml-pipeline
docker-compose logs api-service

# Follow logs in real-time
docker-compose logs -f
```

### Stop Containers

```bash
# Stop containers
docker-compose down

# Stop and remove volumes
docker-compose down -v
```

### Restart Containers

```bash
# Restart all containers
docker-compose restart

# Restart specific container
docker-compose restart ml-pipeline
```

## Development Workflow

### Making Code Changes

Since we mount the current directory as a volume, you can make changes to your code and they'll be reflected immediately:

1. Edit your Python files
2. The containers will automatically reload (if using `--reload` flag)
3. No need to rebuild containers for code changes

### Rebuilding Containers

If you change dependencies or Dockerfiles:

```bash
# Rebuild and restart
docker-compose up --build
```

## Troubleshooting

### Container Won't Start

1. **Check environment variables**:
   ```bash
   docker-compose config
   ```

2. **Check container logs**:
   ```bash
   docker-compose logs ml-pipeline
   ```

3. **Verify external services**:
   - MLflow server is accessible
   - Supabase credentials are correct
   - AWS credentials are valid

### Model Not Available

If the API container fails to find the model:

1. **Check ML container logs**:
   ```bash
   docker-compose logs ml-pipeline
   ```

2. **Verify MLflow connection**:
   - Check if MLflow server is running
   - Verify tracking URI is correct

3. **Check model registration**:
   - Access MLflow UI to see if model is registered
   - Verify model has "production" status

### Port Conflicts

If ports 4200 or 8000 are already in use:

```bash
# Edit docker-compose.yml to use different ports
ports:
  - "4201:4200"  # Use port 4201 on host
  - "8001:8000"  # Use port 8001 on host
```

## Production Deployment

For production, consider:

1. **Remove `--reload` flag** from startup scripts
2. **Use proper secrets management** instead of .env files
3. **Set up proper logging** and monitoring
4. **Configure resource limits** for containers
5. **Use Docker volumes** for persistent data

## File Structure

```
.
├── Dockerfile.ml              # ML Pipeline container
├── Dockerfile.api             # API container
├── docker-compose.yml         # Container orchestration
├── .dockerignore              # Files to exclude from build
├── docker/
│   ├── start_ml_container.sh  # ML container startup script
│   ├── start_api_container.sh # API container startup script
│   └── env.example            # Environment variables template
└── DOCKER_USAGE.md           # This file
```

## Next Steps

1. **Test the containers** with your actual data
2. **Configure monitoring** and alerting
3. **Set up CI/CD** pipeline for automated deployments
4. **Implement retraining triggers** based on monitoring 