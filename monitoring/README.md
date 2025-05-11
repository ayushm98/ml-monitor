# Monitoring Stack

Complete monitoring solution for the ML-Monitor fraud detection system.

## Components

### Prometheus
- **Port**: 9090
- **Purpose**: Metrics collection and time-series database
- **Scrape Targets**:
  - Fraud API (10s interval)
  - MLflow Server (30s interval)
  - Prometheus itself (15s interval)

### Grafana
- **Port**: 3000
- **Credentials**: admin/admin
- **Purpose**: Visualization and alerting
- **Pre-configured**:
  - Prometheus datasource
  - Fraud Detection dashboard

## Metrics Collected

### API Metrics
- `fraud_predictions_total`: Counter of total predictions (labeled by prediction type)
- `fraud_prediction_duration_seconds`: Histogram of prediction latency
- `fraud_probability`: Gauge of last fraud probability

### System Metrics
- Python runtime metrics (GC, memory, threads)
- HTTP request metrics (latency, status codes)

## Dashboards

### Fraud Detection Dashboard
- Prediction rate (per minute)
- Prediction latency (p50, p95, p99)
- Total predictions counter
- Current fraud probability gauge
- Fraud vs legitimate distribution

## Access

After running `docker-compose up`:
- Prometheus UI: http://localhost:9090
- Grafana UI: http://localhost:3000
- API Metrics: http://localhost:8000/metrics

## Configuration

- `prometheus/prometheus.yml`: Scrape configuration
- `grafana/datasources/`: Datasource provisioning
- `grafana/dashboards/`: Dashboard provisioning
