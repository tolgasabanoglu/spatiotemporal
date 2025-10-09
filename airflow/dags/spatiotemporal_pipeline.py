from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago
from datetime import timedelta

default_args = {
    "owner": "tolga",
    "depends_on_past": False,
    "email": ["youremail@example.com"],
    "email_on_failure": True,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="spatiotemporal_pipeline",
    default_args=default_args,
    description="Fetch Garmin data and upload to BigQuery",
    schedule_interval="0 22 * * 0",  # Every Sunday at 22:00
    start_date=days_ago(1),
    catchup=False,
    max_active_runs=1,
    tags=["garmin", "bigquery", "spatiotemporal"],
) as dag:

    fetch_data = BashOperator(
        task_id="fetch_garmin_data",
        bash_command="python /Users/tolgasabanoglu/Desktop/github/spatiotemporal/scripts/parse_garmin.py",
    )

    upload_bq = BashOperator(
        task_id="upload_to_bigquery",
        bash_command="python /Users/tolgasabanoglu/Desktop/github/spatiotemporal/scripts/load_to_bigquery.py",
    )

    fetch_data >> upload_bq
