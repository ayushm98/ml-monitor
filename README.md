# ML-Monitor: Production MLOps Platform

End-to-end MLOps platform for credit card fraud detection with automated monitoring, drift detection, and model retraining.

## Features

- **Real-time Fraud Detection API** - FastAPI service with <100ms latency
- **Automated Model Retraining** - Airflow pipelines for continuous improvement  
- **Drift Detection** - PSI-based monitoring with auto-retraining triggers
- **Observability** - Prometheus metrics + Grafana dashboards
- **CI/CD Pipeline** - Automated testing and deployment
- **Model Explainability** - SHAP integration for interpretability

## Quick Start

```bash
# Start all services
docker-compose up -d

# Check API health
curl http://localhost:8000/health

# Make prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d @sample_transaction.json
```

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   Client    │────▶│  FastAPI     │────▶│   MLflow    │
│ Application │     │   Service    │     │  Registry   │
└─────────────┘     └──────────────┘     └─────────────┘
                           │                      
                           ▼                      
                    ┌──────────────┐     ┌─────────────┐
                    │  Prometheus  │────▶│   Grafana   │
                    │   Metrics    │     │ Dashboards  │
                    └──────────────┘     └─────────────┘
                           │                      
                           ▼                      
                    ┌──────────────┐              
                    │   Airflow    │              
                    │  Pipelines   │              
                    └──────────────┘              
```

## Performance

- **Latency**: p95 < 100ms
- **Throughput**: 1000+ req/sec
- **Model Accuracy**: ROC-AUC 0.9863
- **Uptime**: 99.9%

## Documentation

- [Deployment Guide](docs/DEPLOYMENT.md)
- [Performance Tuning](docs/PERFORMANCE.md)
- [Model Explainability](docs/EXPLAINABILITY.md)
- [CI/CD Setup](.github/README.md)

## Tech Stack

- Python 3.11
- FastAPI + Uvicorn
- scikit-learn + XGBoost
- MLflow 2.9
- Apache Airflow 2.8
- Prometheus + Grafana
- Docker + Docker Compose

## License

MIT
