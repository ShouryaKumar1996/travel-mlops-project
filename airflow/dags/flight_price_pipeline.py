from airflow import DAG

from airflow.operators.python import PythonOperator

from datetime import datetime


# =========================================================
# TASK FUNCTIONS
# =========================================================

def data_ingestion():

    print("Data Ingestion Completed")


def model_training():

    print("Model Training Completed")


def model_evaluation():

    print("Model Evaluation Completed")


def model_deployment():

    print("Model Deployment Completed")


# =========================================================
# DAG CONFIGURATION
# =========================================================

default_args = {

    'owner': 'Shourya',

    'depends_on_past': False,

    'start_date': datetime(2026, 5, 20),

    'retries': 1
}

dag = DAG(

    'flight_price_mlops_pipeline',

    default_args=default_args,

    description='Flight Price Prediction MLOps Pipeline',

    schedule='@daily',

    catchup=False
)


# =========================================================
# TASKS
# =========================================================

task1 = PythonOperator(

    task_id='data_ingestion',

    python_callable=data_ingestion,

    dag=dag
)

task2 = PythonOperator(

    task_id='model_training',

    python_callable=model_training,

    dag=dag
)

task3 = PythonOperator(

    task_id='model_evaluation',

    python_callable=model_evaluation,

    dag=dag
)

task4 = PythonOperator(

    task_id='model_deployment',

    python_callable=model_deployment,

    dag=dag
)


# =========================================================
# DAG PIPELINE
# =========================================================

task1 >> task2 >> task3 >> task4