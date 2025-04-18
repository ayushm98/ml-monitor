# ML-Monitor

A production-grade ML monitoring and deployment platform for end-to-end MLOps workflows.

## Overview

ML-Monitor demonstrates a complete ML lifecycle implementation with automated training, deployment, monitoring, and drift detection capabilities.

**Use Case**: Bike-sharing demand forecasting with real-world data

## Features

- **Experiment Tracking**: MLflow for model versioning and metrics
- **Automated Pipelines**: Airflow DAGs for training and retraining
- **Production API**: FastAPI service with health checks and OpenAPI docs
- **Real-time Monitoring**: Prometheus + Grafana dashboards
- **Drift Detection**: PSI and KS tests for model degradation alerts
- **CI/CD**: GitHub Actions for automated testing and deployment
- **Containerization**: Docker Compose orchestration

## Tech Stack

- **ML Framework**: scikit-learn (RandomForest/XGBoost)
- **API**: FastAPI
- **Experiment Tracking**: MLflow
- **Orchestration**: Apache Airflow
- **Monitoring**: Prometheus + Grafana
- **Containerization**: Docker + Docker Compose
- **CI/CD**: GitHub Actions
- **Data Validation**: Great Expectations
- **Testing**: pytest
- **Database**: PostgreSQL

## Project Structure

```
ml-monitor/
├── data/
│   ├── raw/              # Raw bike-sharing data
│   └── processed/        # Cleaned, feature-engineered data
├── src/
│   ├── api/              # FastAPI application
│   ├── models/           # ML training & inference
│   ├── data/             # Data pipeline
│   ├── monitoring/       # Drift detection
│   └── utils/            # Logging, exceptions
├── tests/
│   ├── unit/             # Unit tests
│   └── integration/      # Integration tests
├── airflow/
│   └── dags/             # Training & retraining DAGs
├── monitoring/
│   ├── prometheus/       # Prometheus config
│   └── grafana/          # Dashboard definitions
├── docker/               # Dockerfiles
└── .github/workflows/    # CI/CD pipelines
```

## Quick Start

```bash
# Clone repository
git clone https://github.com/ayushm98/ml-monitor.git
cd ml-monitor

# Set up virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run tests
pytest tests/

# Start services (Docker required)
docker-compose up -d

# Access services
# API: http://localhost:8000/docs
# MLflow: http://localhost:5000
# Grafana: http://localhost:3000
```

## Development Roadmap

### Phase 1: Foundation (Days 1-3)
- [x] Project structure
- [ ] Data pipeline with validation
- [ ] MLflow setup and baseline model
- [ ] FastAPI prediction service

### Phase 2: Infrastructure (Days 4-6)
- [ ] Docker Compose stack
- [ ] Prometheus + Grafana monitoring
- [ ] Drift detection service

### Phase 3: Automation (Days 7-9)
- [ ] Airflow training pipelines
- [ ] CI/CD with GitHub Actions
- [ ] Production logging and error handling

### Phase 4: Polish (Days 10-12)
- [ ] SHAP explainability
- [ ] Comprehensive testing (80%+ coverage)
- [ ] Documentation

### Phase 5: Deployment (Days 13-14)
- [ ] Cloud deployment
- [ ] Demo video and blog post

## Monitoring Metrics

### Model Performance
- RMSE, MAE, R² score
- Prediction latency (p50, p95, p99)
- Feature drift scores (PSI, KS test)

### System Health
- Request rate, error rate, latency
- CPU, memory, disk usage
- Model loading time

### Business Metrics
- Predictions per hour
- Model version distribution
- Retraining frequency

## Contributing

This is a portfolio project. Feedback and suggestions are welcome via issues!

## License

MIT License

## Contact

Ayush Kumar Malik
- GitHub: [@ayushm98](https://github.com/ayushm98)
- LinkedIn: [ayush67](https://linkedin.com/in/ayush67)
- Email: ayushkumarmalik10@gmail.com
