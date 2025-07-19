# Solar Prediction MLOps - Docker Architecture Plan

## 🎯 Project Overview
Portfolio MLOps project with peer review requirements. Need reproducible, professional Docker setup.

## 🏗️ Current Architecture
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Local         │    │   EC2           │    │   Local         │
│   Development   │    │   (Remote)      │    │   Monitoring    │
│                 │    │                 │    │                 │
│ • pipeline.py   │───►│ • MLflow        │◄───│ • monitor_drift │
│ • Prefect flow  │    │ • SQLite DB     │    │ • serve_model   │
│ • Training      │    │ • S3 artifacts  │    │ • FastAPI       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🚀 Target Docker Architecture
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Prefect       │    │   Model         │    │   Monitoring    │
│   Container     │    │   Container     │    │   Container     │
│                 │    │                 │    │                 │
│ • pipeline.py   │    │ • FastAPI       │    │ • Prometheus    │
│ • monitor_drift │    │ • Model serve   │    │ • Grafana       │
│ • retrain.py    │    │ • Predictions   │    │                 │
│ • Orchestration │    │ • Model cache   │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 📋 Container Breakdown

### Container 1: Prefect (Orchestration)
**Purpose**: Workflow orchestration and management
**Services**:
- Prefect server
- Training flows (pipeline.py)
- Monitoring flows (monitor_drift.py) - **TO BE WRAPPED**
- Retraining flows (retrain.py) - **TO BE WRAPPED**
- Model update orchestration

**Responsibilities**:
- Schedule and run ML workflows
- Manage model training pipeline
- Orchestrate drift detection
- Handle retraining triggers
- Coordinate model updates

### Container 2: Model (Serving)
**Purpose**: Model serving and predictions
**Services**:
- FastAPI with uvicorn
- Model loading and caching
- Prediction serving
- Health check endpoints
- Model version tracking

**Responsibilities**:
- Serve model predictions via API
- Load and cache models from MLflow
- Handle prediction requests
- Provide health checks
- Track model versions

### Container 3: Monitoring (Prometheus + Grafana)
**Purpose**: Metrics collection and visualization
**Services**:
- Prometheus (official image)
- Grafana (official image)
- Metrics scraping
- Dashboard visualization

**Responsibilities**:
- Collect metrics from model container
- Store time-series data
- Visualize drift detection results
- Display performance metrics
- Send alerts

## 🔄 Model Update Strategy

### Blue-Green Deployment
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Step 1        │    │   Step 2        │    │   Step 3        │
│   Blue          │    │   Both Running  │    │   Green         │
│   (Old Model)   │    │                 │    │   (New Model)   │
│                 │    │                 │    │                 │
│ • Model v1      │    │ • Model v1      │    │ • Model v2      │
│ • FastAPI       │    │ • Model v2      │    │ • FastAPI       │
│ • Active        │    │ • Both Active   │    │ • Active        │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Update Process:
1. **Retraining**: New model trained and registered in MLflow
2. **New Container**: New model container starts with new model
3. **Health Check**: Verify new container is ready
4. **Traffic Switch**: Route requests to new container
5. **Old Container**: Stop old container (keep for rollback)
6. **Cleanup**: Eventually remove old images (keep last 2-3)

## 🎯 Key Decisions Made

### ✅ Architecture Decisions:
- **Separate model container** (not shared with Prefect)
- **Blue-green deployment** for zero downtime updates
- **Keep last 2-3 model versions** for rollback capability
- **Official images** for Prometheus and Grafana
- **Docker Compose** for orchestration

### ✅ Implementation Decisions:
- **Wrap monitor_drift.py** as Prefect flow
- **Container restart** approach for model updates
- **Health checks** for model version detection
- **Environment variables** for configuration
- **Volume mounts** for data persistence

### ✅ Portfolio Considerations:
- **Reproducible setup** for peer review
- **Professional architecture** showing microservices
- **Industry best practices** for MLOps
- **Clear documentation** and setup instructions

## 📝 Implementation Checklist

### Phase 1: Prefect Flow Wrapping
- [ ] Wrap monitor_drift.py as Prefect flow
- [ ] Wrap retrain.py as Prefect flow (when triggers defined)
- [ ] Test flows locally
- [ ] Add scheduling and error handling

### Phase 2: Container Creation
- [ ] Create Prefect container Dockerfile
- [ ] Create Model container Dockerfile
- [ ] Set up Prometheus container (official image)
- [ ] Set up Grafana container (official image)
- [ ] Test individual containers

### Phase 3: Docker Compose
- [ ] Create docker-compose.yml
- [ ] Configure container networking
- [ ] Set up environment variables
- [ ] Configure volume mounts
- [ ] Test complete setup

### Phase 4: Model Update Logic
- [ ] Implement health check endpoints
- [ ] Add model version tracking
- [ ] Create update orchestration
- [ ] Test blue-green deployment
- [ ] Add rollback capability

### Phase 5: Production Ready
- [ ] Add logging and monitoring
- [ ] Configure alerts
- [ ] Create documentation
- [ ] Test peer review setup
- [ ] Performance optimization

## 🔧 Technical Specifications

### Prefect Container:
```dockerfile
# Base image: Python 3.9+
# Dependencies: prefect, mlflow, evidently, supabase, boto3
# Ports: 4200 (Prefect UI)
# Volumes: ./flows:/app/flows
```

### Model Container:
```dockerfile
# Base image: Python 3.9+
# Dependencies: fastapi, uvicorn, mlflow, supabase, boto3
# Ports: 8000 (FastAPI)
# Volumes: ./models:/app/models (if needed)
```

### Monitoring Containers:
```yaml
# Prometheus: Official image prom/prometheus
# Grafana: Official image grafana/grafana
# Ports: 9090 (Prometheus), 3000 (Grafana)
```

## 🌐 Communication Flow

### Prefect → Model Container:
- Health checks
- Model update signals
- Performance monitoring

### Model Container → Monitoring:
- Prediction metrics
- Model performance
- Error rates

### External → Model Container:
- Prediction requests
- API calls

### External → Prefect:
- Flow triggers
- Manual retraining

## 📊 Benefits for Portfolio

### ✅ Technical Skills Demonstrated:
- **Containerization**: Docker and microservices
- **Orchestration**: Prefect workflow management
- **Monitoring**: Prometheus + Grafana integration
- **CI/CD**: Automated model updates
- **Infrastructure**: Multi-container architecture

### ✅ MLOps Best Practices:
- **Model versioning**: MLflow integration
- **Drift detection**: Automated monitoring
- **Zero-downtime deployment**: Blue-green strategy
- **Observability**: Comprehensive monitoring
- **Reproducibility**: Docker-based setup

### ✅ Professional Presentation:
- **Industry-standard architecture**
- **Easy peer review setup**
- **Clear documentation**
- **Production-ready practices**

## 🚨 Important Notes

### For Peer Review:
- **One command setup**: `docker-compose up -d`
- **Clear access points**: Documented URLs for each service
- **Reproducible environment**: Same setup for everyone
- **Professional documentation**: README with architecture diagrams

### For Production:
- **Security**: Environment variables for secrets
- **Scaling**: Container resource limits
- **Monitoring**: Health checks and alerts
- **Backup**: Data persistence strategies

## 📚 Next Steps

1. **Start with Prefect flow wrapping** (monitor_drift.py)
2. **Create individual containers** and test
3. **Set up Docker Compose** for orchestration
4. **Implement model update logic**
5. **Add monitoring and documentation**
6. **Test peer review setup**

---

**Last Updated**: [Current Date]
**Status**: Planning Phase
**Next Action**: Wrap monitor_drift.py as Prefect flow 