# Deployment Guide

Production deployment guide for ML-Monitor fraud detection system.

## Prerequisites

- Docker & Docker Compose installed
- 8GB+ RAM recommended
- 20GB+ disk space
- PostgreSQL for MLflow and Airflow
- Access to model artifacts (MLflow tracking server)

## Quick Start

### 1. Environment Setup

Create `.env` file:
```bash
# MLflow
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_BACKEND_STORE_URI=postgresql://mlmonitor:mlmonitor123@postgres:5432/mlflow

# API
API_HOST=0.0.0.0
API_PORT=8000

# Monitoring
PROMETHEUS_PORT=9090
GRAFANA_PORT=3000

# Airflow
AIRFLOW_UID=50000
AIRFLOW__CORE__EXECUTOR=LocalExecutor
```

### 2. Start Core Services

```bash
# Start infrastructure
docker-compose up -d postgres mlflow prometheus grafana

# Wait for services to be healthy
docker-compose ps

# Start API
docker-compose up -d api

# Start Airflow (optional)
docker-compose -f airflow/docker-compose-airflow.yml up -d
```

### 3. Verify Services

```bash
# Check API health
curl http://localhost:8000/health

# Check MLflow
curl http://localhost:5000/health

# Check Prometheus targets
curl http://localhost:9090/api/v1/targets

# Grafana UI
open http://localhost:3000  # admin/admin
```

## Production Checklist

### Security
- [ ] Change default passwords (Grafana, Airflow, PostgreSQL)
- [ ] Enable HTTPS/TLS for all services
- [ ] Configure firewall rules
- [ ] Set up API authentication (OAuth2/JWT)
- [ ] Rotate MLflow credentials
- [ ] Enable audit logging

### Monitoring
- [ ] Configure Prometheus retention
- [ ] Set up Grafana dashboards
- [ ] Configure alerting (PagerDuty/Slack)
- [ ] Enable drift detection alerts
- [ ] Set up log aggregation (ELK/Loki)

### Data
- [ ] Configure data backup strategy
- [ ] Set up model artifact backup
- [ ] Configure PostgreSQL backups
- [ ] Test disaster recovery procedures

### Performance
- [ ] Load test API endpoints
- [ ] Tune database connection pools
- [ ] Configure API rate limiting
- [ ] Optimize Docker resource limits
- [ ] Enable caching where appropriate

## Scaling

### Horizontal Scaling

For high-traffic scenarios:

```yaml
# docker-compose.yml
services:
  api:
    deploy:
      replicas: 3
    environment:
      - WORKERS=4
```

Add load balancer (nginx):
```nginx
upstream fraud_api {
    server api-1:8000;
    server api-2:8000;
    server api-3:8000;
}
```

### Vertical Scaling

Resource limits:
```yaml
services:
  api:
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G
        reservations:
          cpus: '1'
          memory: 2G
```

## Monitoring & Alerts

### Key Metrics

- **API Latency**: p50, p95, p99 < 100ms
- **Error Rate**: < 0.1%
- **Throughput**: Monitor requests/sec
- **Model Drift**: PSI < 0.1
- **Data Quality**: Validation pass rate > 99%

### Alert Rules

Prometheus alerts:
```yaml
groups:
  - name: fraud_api
    rules:
      - alert: HighErrorRate
        expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.01
        for: 5m
        
      - alert: HighLatency
        expr: histogram_quantile(0.95, rate(fraud_prediction_duration_seconds_bucket[5m])) > 0.5
        
      - alert: ModelDrift
        expr: fraud_drift_psi > 0.25
```

## Troubleshooting

### API Not Responding

```bash
# Check logs
docker logs fraud-api

# Restart service
docker-compose restart api

# Check resource usage
docker stats
```

### Model Loading Failures

```bash
# Verify MLflow connectivity
curl http://mlflow:5000/api/2.0/mlflow/registered-models/list

# Check model artifacts
docker exec fraud-api ls -la mlruns/
```

### Database Connection Issues

```bash
# Test PostgreSQL connection
docker exec postgres psql -U mlmonitor -c "SELECT 1"

# Check connection pool
docker exec fraud-api python -c "from sqlalchemy import create_engine; engine = create_engine('postgresql://...'); print(engine.pool.status())"
```

## Backup & Recovery

### Database Backup

```bash
# Backup PostgreSQL
docker exec postgres pg_dump -U mlmonitor mlmonitor > backup_$(date +%Y%m%d).sql

# Restore
docker exec -i postgres psql -U mlmonitor mlmonitor < backup_20250614.sql
```

### Model Artifacts

```bash
# Backup MLflow artifacts
tar -czf mlflow_backup_$(date +%Y%m%d).tar.gz mlruns/

# Upload to S3
aws s3 cp mlflow_backup_20250614.tar.gz s3://ml-monitor-backups/
```

## Performance Tuning

### API Optimization

```python
# Increase workers
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]

# Enable production mode
ENV PYTHONOPTIMIZE=2
```

### Database Tuning

```sql
-- Optimize PostgreSQL
ALTER SYSTEM SET shared_buffers = '256MB';
ALTER SYSTEM SET effective_cache_size = '1GB';
ALTER SYSTEM SET work_mem = '16MB';
```

## Rollback Procedures

### API Rollback

```bash
# Revert to previous image
docker-compose down api
docker-compose up -d api:<previous-tag>
```

### Model Rollback

```python
# In MLflow UI or via API
client.transition_model_version_stage(
    name="fraud-detector",
    version="<previous-version>",
    stage="Production"
)
```
