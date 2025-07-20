# Docker Containerization Plan for Solar Prediction MLOps

## Overview
This document outlines the plan to containerize the solar prediction MLOps pipeline for reproducibility and deployment.

## Current Architecture
- **pipeline.py**: Main ML pipeline that trains models and registers to MLflow
- **retrain.py**: Retraining script that merges data and runs pipeline  
- **api/serve_model.py**: FastAPI service for predictions
- **External Services**: MLflow (EC2), Supabase
- **Prefect**: Local server with UI for deployment management

## Container Architecture

### 1. ML Pipeline Container (`Dockerfile.ml`)
**Purpose**: Run Prefect server + worker + ML scripts

**Components**:
- Prefect server (accessible via UI)
- Prefect worker named "ml-pipeline-worker"
- ML pipeline scripts (`pipeline.py`, `retrain.py`)
- All ML dependencies

**Startup Sequence**:
1. Start Prefect server
2. Wait for server to be ready
3. Start Prefect worker ("ml-pipeline-worker")
4. Automatically trigger ML pipeline deployment
5. Wait for pipeline completion (model training + registration)
6. Container ready - model available in MLflow

**Port Exposure**: 4200 (Prefect UI)

### 2. API Container (`Dockerfile.api`)
**Purpose**: Run FastAPI service for predictions

**Components**:
- FastAPI service (`serve_model.py`)
- Model loading from MLflow
- Prediction logging to Supabase

**Dependencies**: 
- Requires ML container to complete first (model must be registered)

## Deployment Order
1. Start ML Pipeline Container
2. Wait for ML container to be ready (model trained and registered)
3. Start API Container

## Key Benefits
- **Reproducibility**: Anyone can run the same environment
- **Separation of Concerns**: ML pipeline separate from API
- **Scalability**: Can scale API independently
- **Fault Isolation**: One service failing doesn't affect the other

## External Dependencies
- **MLflow**: Running on EC2 (model registry)
- **Supabase**: External database (prediction logging)
- **S3**: Data storage (AWS)

## User Experience
- User runs containers
- Accesses Prefect UI at `http://localhost:4200`
- Sees "ml-pipeline-worker" in Prefect UI
- Can trigger deployments through familiar UI
- API available for predictions once ML container is ready

## Files to Create
1. `Dockerfile.ml` - ML Pipeline container ✅
2. `docker/start_ml_container.sh` - ML container startup script ✅
3. `Dockerfile.api` - API container ✅
4. `docker/start_api_container.sh` - API container startup script ✅
5. `Dockerfile.prefect` - Prefect server container ✅
6. `docker/start_prefect_server.sh` - Prefect server startup script ✅
7. `docker-compose.yml` - Orchestrate all 3 containers ✅
8. `.dockerignore` - Exclude unnecessary files ✅
9. `docker/env.example` - Environment variables template ✅
10. `DOCKER_USAGE.md` - Usage instructions ✅

## Environment Variables Needed
- MLflow tracking URI
- Supabase credentials
- S3 credentials
- Model configuration

## Next Steps
1. Create startup scripts
2. Create API container Dockerfile
3. Create docker-compose.yml
4. Test container startup sequence
5. Document usage instructions 