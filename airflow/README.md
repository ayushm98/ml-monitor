# Airflow Orchestration

Automated ML pipeline orchestration using Apache Airflow.

## DAGs

### 1. Model Retraining (`model_retraining_dag.py`)

**Schedule**: Weekly  
**Purpose**: Automated model retraining and deployment

**Workflow**:
```
Check Data → Validate Quality → Train Model → Evaluate → Promote → Restart API
```

**Tasks**:
1. **check_data_freshness** - Verify dataset availability
2. **validate_data_quality** - Run data quality checks
3. **train_model** - Train new RandomForest model
4. **evaluate_model** - Check performance metrics (ROC-AUC > 0.95)
5. **promote_to_production** - Move model to Production stage in MLflow
6. **restart_api_service** - Reload API with new model

**Triggers**:
- Scheduled weekly
- Manual trigger via UI
- Triggered by drift monitoring

### 2. Drift Monitoring (`drift_monitoring_dag.py`)

**Schedule**: Hourly  
**Purpose**: Monitor data drift and trigger alerts

**Workflow**:
```
Check Drift → Decide Action → [No Drift | Minor | Alert | Retrain]
```

**Decision Logic**:
- **PSI < 0.1**: No drift → Log and continue
- **PSI 0.1-0.15**: Minor drift → Log warning
- **PSI 0.15-0.25**: Moderate drift → Send alert
- **PSI > 0.25**: Critical drift → Trigger retraining

**Tasks**:
1. **check_drift** - Query API /drift/status endpoint
2. **decide_action** - Branch based on PSI severity
3. **log_no_drift** - Log healthy status
4. **log_minor_drift** - Log minor shift
5. **send_alert** - Alert team of moderate drift
6. **trigger_retraining** - Auto-trigger retraining DAG

## Setup

### Start Airflow

```bash
# Initialize Airflow database
docker-compose -f airflow/docker-compose-airflow.yml up airflow-init

# Start Airflow services
docker-compose -f airflow/docker-compose-airflow.yml up -d
```

### Access Airflow UI

- URL: http://localhost:8080
- Username: `admin`
- Password: `admin`

### Trigger DAG Manually

Via UI:
1. Navigate to http://localhost:8080
2. Find DAG in list
3. Click play button

Via CLI:
```bash
docker exec airflow-scheduler airflow dags trigger model_retraining
docker exec airflow-scheduler airflow dags trigger drift_monitoring
```

## Configuration

### Environment Variables

Set in `docker-compose-airflow.yml`:
- `AIRFLOW__CORE__EXECUTOR`: LocalExecutor for single-node
- `AIRFLOW__DATABASE__SQL_ALCHEMY_CONN`: PostgreSQL connection
- `AIRFLOW__CORE__LOAD_EXAMPLES`: false (no example DAGs)

### DAG Configuration

Edit DAG files to customize:
- `schedule_interval`: Cron expression or preset
- `default_args`: Retries, email settings
- `tags`: DAG categorization

### Email Alerts

Configure SMTP in `docker-compose-airflow.yml`:
```yaml
AIRFLOW__SMTP__SMTP_HOST: smtp.gmail.com
AIRFLOW__SMTP__SMTP_PORT: 587
AIRFLOW__SMTP__SMTP_USER: your-email@gmail.com
AIRFLOW__SMTP__SMTP_PASSWORD: your-app-password
AIRFLOW__SMTP__SMTP_MAIL_FROM: airflow@ml-monitor.com
```

## Monitoring

### DAG Run Status

Check DAG run history:
```bash
docker exec airflow-scheduler airflow dags list-runs -d model_retraining
```

### Task Logs

View in UI or via CLI:
```bash
docker exec airflow-scheduler airflow tasks logs model_retraining train_model 2024-01-15
```

### Scheduler Health

```bash
docker exec airflow-scheduler airflow jobs check --job-type SchedulerJob
```

## Best Practices

1. **Idempotency**: All tasks should be idempotent (safe to retry)
2. **XCom**: Use for small data sharing between tasks
3. **Task Dependencies**: Use `>>` or `set_upstream()`/`set_downstream()`
4. **Retries**: Configure appropriate retry logic
5. **Alerts**: Set up email/Slack for failures
6. **Resources**: Monitor Airflow resource usage

## Troubleshooting

### DAG Not Appearing

- Check DAG file syntax: `python airflow/dags/my_dag.py`
- Verify DAG directory is mounted
- Check scheduler logs

### Task Failures

- View task logs in UI
- Check task instance details
- Verify external service connectivity (MLflow, API)

### Database Issues

```bash
# Reset database
docker-compose -f airflow/docker-compose-airflow.yml down -v
docker-compose -f airflow/docker-compose-airflow.yml up airflow-init
```

## Future Enhancements

- Add data quality monitoring DAG
- Implement A/B testing workflow
- Add model performance tracking
- Create automated rollback on degradation
- Integrate with Slack for alerts
- Add model explainability reports
