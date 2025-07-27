# Solar Prediction ML Pipeline - Sequence Diagram

```mermaid
sequenceDiagram
    participant K as Kaggle
    participant S3 as S3 Bucket
    participant PP as Data Pipeline
    participant ML as MLflow
    participant API as FastAPI
    participant DB as Supabase DB
    participant EV as Evidently AI
    participant PM as Prometheus
    participant GF as Grafana
    participant RT as Retrain Flow
    participant AM as Archive Manager

    Note over K,AM: Initial Data Pipeline Setup
    
    K->>S3: Upload raw solar data
    S3->>PP: Load raw data from S3
    PP->>PP: Feature engineering & preprocessing
    PP->>S3: Save processed data back to S3
    PP->>S3: Load processed data for training
    PP->>ML: Train multiple models (XGBoost, Random Forest, etc.)
    ML->>ML: Log experiments & metrics
    PP->>ML: Load top 3 models from MLflow
    PP->>PP: Evaluate models on test data
    PP->>ML: Register best model as production
    ML->>API: FastAPI loads production model from MLflow

    Note over K,AM: Production Inference Phase
    
    API->>API: Receive prediction requests
    API->>API: Make predictions using production model
    API->>DB: Log input data & predictions to PostgreSQL
    API->>API: Return predictions to clients

    Note over K,AM: Monitoring & Drift Detection Phase
    
    EV->>S3: Fetch current baseline data
    EV->>DB: Fetch logged input data from Supabase
    EV->>EV: Calculate drift metrics (statistical, distance, etc.)
    PM->>EV: Scrape drift metrics
    PM->>GF: Send metrics to Grafana
    GF->>GF: Visualize drift metrics & trends
    
    alt Drift Detected
        EV->>EV: Generate drift alerts
        EV->>GF: Display drift alerts in Grafana
        Note over EV,RT: Drift threshold exceeded
        RT->>S3: Load baseline data
        RT->>S3: Load new drift-causing data
        RT->>RT: Merge baseline + new data
        RT->>S3: Save merged data (overwrite baseline)
        RT->>PP: Trigger retraining pipeline
        PP->>S3: Load merged data
        PP->>PP: Feature engineering on merged data
        PP->>S3: Save processed merged data
        PP->>ML: Train new models on merged data
        ML->>ML: Log new experiments
        PP->>ML: Load top 3 new models
        PP->>PP: Evaluate new models
        PP->>ML: Register new best model as production
        ML->>AM: Archive old production model
        AM->>ML: Store archived model version
        ML->>API: Update FastAPI with new production model
    else No Drift
        EV->>EV: Continue monitoring
        Note over EV,RT: Normal operation continues
    end

    Note over K,AM: Continuous Monitoring Loop
    
    loop Every monitoring interval
        EV->>S3: Fetch latest baseline
        EV->>DB: Fetch recent predictions
        EV->>EV: Recalculate drift metrics
        PM->>EV: Update Prometheus metrics
        GF->>PM: Refresh Grafana dashboards
    end
```

## Key Components Explained

### Data Flow
1. **Kaggle → S3**: Initial data ingestion from Kaggle datasets
2. **S3 → Pipeline**: Data loading and processing
3. **Pipeline → MLflow**: Model training and experiment tracking
4. **MLflow → FastAPI**: Production model serving

### Monitoring Stack
- **Evidently AI**: Drift detection and analysis
- **Prometheus**: Metrics collection and storage
- **Grafana**: Visualization and alerting
- **Supabase**: Prediction logging and data storage

### Retraining Flow
- **Drift Detection**: Automated monitoring triggers retraining
- **Data Merging**: New data merged with baseline
- **Model Retraining**: Complete pipeline re-execution
- **Model Registration**: New best model becomes production
- **Model Archiving**: Old models stored for rollback capability

### Key Features
- **Automated Drift Detection**: Statistical, distance-based, and enhanced drift metrics
- **Continuous Monitoring**: Real-time metrics and visualization
- **Automated Retraining**: Triggered by drift detection
- **Model Versioning**: MLflow model registry with production/archive management
- **Prediction Logging**: Complete audit trail of inputs and outputs 