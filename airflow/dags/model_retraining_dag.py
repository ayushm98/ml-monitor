"""Airflow DAG for automated model retraining."""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils.dates import days_ago

default_args = {
    'owner': 'ml-monitor',
    'depends_on_past': False,
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}


def check_data_freshness(**context):
    """Check if new data is available for retraining."""
    import pandas as pd
    from pathlib import Path
    
    data_path = Path('/app/data/raw/creditcard.csv')
    if not data_path.exists():
        raise FileNotFoundError("Dataset not found")
    
    df = pd.read_csv(data_path)
    context['ti'].xcom_push(key='data_size', value=len(df))
    
    print(f"Dataset contains {len(df)} transactions")
    return True


def validate_data_quality(**context):
    """Validate data quality before training."""
    from src.data.validation import FraudDataValidator
    from src.data.ingestion import FraudDataLoader
    
    loader = FraudDataLoader()
    train_df, test_df = loader.load_and_split()
    
    validator = FraudDataValidator()
    validation_result = validator.validate_fraud_data(train_df, test_df)
    
    if not validation_result['passed']:
        raise ValueError(f"Data validation failed: {validation_result['message']}")
    
    context['ti'].xcom_push(key='validation_passed', value=True)
    print("Data quality validation passed")
    return True


def train_model(**context):
    """Train new fraud detection model."""
    import subprocess
    
    result = subprocess.run(
        ['python', 'src/models/train.py', '--n_estimators', '100', '--max_depth', '10'],
        capture_output=True,
        text=True,
        cwd='/app'
    )
    
    if result.returncode != 0:
        raise RuntimeError(f"Training failed: {result.stderr}")
    
    print("Model training completed successfully")
    context['ti'].xcom_push(key='training_completed', value=True)
    return True


def evaluate_model(**context):
    """Evaluate new model and compare with production."""
    import mlflow
    from mlflow.tracking import MlflowClient
    
    client = MlflowClient(tracking_uri='http://mlflow:5000')
    
    # Get latest run
    experiment = client.get_experiment_by_name('fraud-detection')
    if experiment:
        runs = client.search_runs(experiment_ids=[experiment.experiment_id], max_results=1)
        if runs:
            latest_run = runs[0]
            metrics = latest_run.data.metrics
            
            roc_auc = metrics.get('roc_auc', 0)
            
            # Threshold for production deployment
            if roc_auc < 0.95:
                raise ValueError(f"Model performance below threshold: {roc_auc:.4f}")
            
            context['ti'].xcom_push(key='roc_auc', value=roc_auc)
            print(f"Model evaluation passed with ROC-AUC: {roc_auc:.4f}")
            return True
    
    return False


def promote_to_production(**context):
    """Promote model to production stage in MLflow."""
    import mlflow
    from mlflow.tracking import MlflowClient
    
    client = MlflowClient(tracking_uri='http://mlflow:5000')
    
    # Get latest version
    model_name = "fraud-detector"
    latest_versions = client.get_latest_versions(model_name, stages=["None"])
    
    if latest_versions:
        latest_version = latest_versions[0]
        client.transition_model_version_stage(
            name=model_name,
            version=latest_version.version,
            stage="Production",
            archive_existing_versions=True
        )
        print(f"Promoted model version {latest_version.version} to Production")
        return True
    
    return False


with DAG(
    'model_retraining',
    default_args=default_args,
    description='Automated model retraining pipeline',
    schedule_interval='@weekly',  # Run weekly
    start_date=days_ago(1),
    catchup=False,
    tags=['ml', 'retraining', 'fraud-detection'],
) as dag:
    
    check_data = PythonOperator(
        task_id='check_data_freshness',
        python_callable=check_data_freshness,
        provide_context=True,
    )
    
    validate_data = PythonOperator(
        task_id='validate_data_quality',
        python_callable=validate_data_quality,
        provide_context=True,
    )
    
    train = PythonOperator(
        task_id='train_model',
        python_callable=train_model,
        provide_context=True,
    )
    
    evaluate = PythonOperator(
        task_id='evaluate_model',
        python_callable=evaluate_model,
        provide_context=True,
    )
    
    promote = PythonOperator(
        task_id='promote_to_production',
        python_callable=promote_to_production,
        provide_context=True,
    )
    
    restart_api = BashOperator(
        task_id='restart_api_service',
        bash_command='docker-compose restart api',
    )
    
    # Define task dependencies
    check_data >> validate_data >> train >> evaluate >> promote >> restart_api
