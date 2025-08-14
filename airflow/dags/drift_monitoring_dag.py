"""Airflow DAG for drift monitoring and alerting."""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.email import EmailOperator
from airflow.utils.dates import days_ago

default_args = {
    'owner': 'ml-monitor',
    'depends_on_past': False,
    'email_on_failure': True,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}


def check_drift_status(**context):
    """Check current drift status from API."""
    import requests
    import json
    
    try:
        response = requests.get('http://api:8000/drift/status', timeout=10)
        response.raise_for_status()
        
        drift_data = response.json()
        
        context['ti'].xcom_push(key='drift_detected', value=drift_data.get('drift_detected', False))
        context['ti'].xcom_push(key='max_psi', value=drift_data.get('max_psi', 0))
        context['ti'].xcom_push(key='drift_data', value=json.dumps(drift_data))
        
        print(f"Drift check complete. Detected: {drift_data.get('drift_detected')}")
        print(f"Max PSI: {drift_data.get('max_psi', 0):.4f}")
        
        return drift_data.get('drift_detected', False)
        
    except Exception as e:
        print(f"Error checking drift: {e}")
        return False


def decide_action(**context):
    """Decide on action based on drift severity."""
    drift_detected = context['ti'].xcom_pull(key='drift_detected', task_ids='check_drift')
    max_psi = context['ti'].xcom_pull(key='max_psi', task_ids='check_drift')
    
    if not drift_detected:
        return 'log_no_drift'
    
    if max_psi > 0.25:
        # Critical drift - trigger retraining
        return 'trigger_retraining'
    elif max_psi > 0.15:
        # Moderate drift - send alert
        return 'send_alert'
    else:
        # Minor drift - just log
        return 'log_minor_drift'


def log_no_drift(**context):
    """Log when no drift is detected."""
    print("No significant drift detected. System operating normally.")
    return True


def log_minor_drift(**context):
    """Log minor drift."""
    max_psi = context['ti'].xcom_pull(key='max_psi', task_ids='check_drift')
    print(f"Minor drift detected (PSI: {max_psi:.4f}). Monitoring continues.")
    return True


def send_drift_alert(**context):
    """Send alert for moderate drift."""
    import json
    
    drift_data = json.loads(context['ti'].xcom_pull(key='drift_data', task_ids='check_drift'))
    max_psi = drift_data.get('max_psi', 0)
    features_with_drift = drift_data.get('features_with_drift', [])
    
    alert_message = f"""
    Moderate Data Drift Detected!
    
    Max PSI: {max_psi:.4f}
    Features with drift: {len(features_with_drift)}
    
    Drifted features:
    {json.dumps(features_with_drift, indent=2)}
    
    Action: Monitor closely. Consider retraining if PSI exceeds 0.25.
    """
    
    print(alert_message)
    context['ti'].xcom_push(key='alert_sent', value=True)
    return True


def trigger_model_retraining(**context):
    """Trigger retraining DAG for critical drift."""
    from airflow.api.common.trigger_dag import trigger_dag
    
    max_psi = context['ti'].xcom_pull(key='max_psi', task_ids='check_drift')
    
    print(f"CRITICAL DRIFT DETECTED (PSI: {max_psi:.4f})")
    print("Triggering automatic model retraining...")
    
    # Trigger retraining DAG
    trigger_dag(
        dag_id='model_retraining',
        run_id=f'auto_retrain_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
        conf={'triggered_by': 'drift_detection', 'max_psi': max_psi}
    )
    
    return True


with DAG(
    'drift_monitoring',
    default_args=default_args,
    description='Monitor production data drift and trigger alerts',
    schedule_interval='@hourly',  # Check every hour
    start_date=days_ago(1),
    catchup=False,
    tags=['monitoring', 'drift', 'alerts'],
) as dag:
    
    check_drift = PythonOperator(
        task_id='check_drift',
        python_callable=check_drift_status,
        provide_context=True,
    )
    
    decide = BranchPythonOperator(
        task_id='decide_action',
        python_callable=decide_action,
        provide_context=True,
    )
    
    no_drift = PythonOperator(
        task_id='log_no_drift',
        python_callable=log_no_drift,
        provide_context=True,
    )
    
    minor_drift = PythonOperator(
        task_id='log_minor_drift',
        python_callable=log_minor_drift,
        provide_context=True,
    )
    
    alert = PythonOperator(
        task_id='send_alert',
        python_callable=send_drift_alert,
        provide_context=True,
    )
    
    retrain = PythonOperator(
        task_id='trigger_retraining',
        python_callable=trigger_model_retraining,
        provide_context=True,
    )
    
    # Task dependencies
    check_drift >> decide
    decide >> [no_drift, minor_drift, alert, retrain]
