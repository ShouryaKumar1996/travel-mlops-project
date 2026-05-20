````markdown
# Flight Price Prediction MLOps Project

## Project Overview

This project is an end-to-end MLOps implementation for Flight Price Prediction using Machine Learning, Docker, Kubernetes, Jenkins, MLflow, Airflow, Flask, and Streamlit.

The system predicts airline ticket prices based on flight details and demonstrates the complete ML lifecycle from model training to deployment and orchestration.

---

# Project Architecture

```text
Data Ingestion
      ↓
Data Preprocessing
      ↓
Model Training
      ↓
MLflow Experiment Tracking
      ↓
Flask API Deployment
      ↓
Docker Containerization
      ↓
Kubernetes Deployment
      ↓
Jenkins CI/CD Pipeline
      ↓
Airflow DAG Orchestration
````

---

# Tech Stack

| Category            | Tools Used            |
| ------------------- | --------------------- |
| Programming         | Python                |
| Machine Learning    | Scikit-Learn, XGBoost |
| Frontend            | Streamlit             |
| Backend API         | Flask                 |
| Experiment Tracking | MLflow                |
| Containerization    | Docker                |
| Orchestration       | Kubernetes            |
| CI/CD               | Jenkins               |
| Workflow Scheduling | Airflow               |

---

# Dataset

The dataset contains flight information such as:

* Source City
* Destination City
* Flight Type
* Airline Agency
* Distance
* Flight Duration
* Date Features

### Target Variable

* Flight Ticket Price

---

# Machine Learning Models

Models Implemented:

* Random Forest Regressor
* XGBoost Regressor

### Final Production Model

* XGBoost Regressor

---

# Model Performance

| Metric   | Value     |
| -------- | --------- |
| R² Score | ADD_VALUE |
| RMSE     | ADD_VALUE |
| MAE      | ADD_VALUE |

---

# Project Structure

```text
travel-mlops-project/

├── airflow/
├── airflow_setup/
├── data/
├── flask_api/
├── kubernetes/
├── models/
├── notebooks/
├── screenshots/
├── streamlit_app/
├── mlruns/
├── Jenkinsfile
├── Dockerfile
├── requirements.txt
├── README.md
├── .gitignore
```

---

# MLflow Experiment Tracking

MLflow is used for:

* Model tracking
* Metric logging
* Experiment comparison
* Artifact storage

### MLflow Dashboard

![MLflow](screenshots/mlflow_dashboard.png)

---

# Flask API

The trained model is deployed using Flask API.

### API Endpoint

```python
POST /predict
```

### Flask API Screenshot

![Flask API](screenshots/flask_api.png)

---

# Streamlit Web Application

A user-friendly Streamlit frontend was developed for real-time flight price prediction.

### Streamlit Screenshot

![Streamlit](screenshots/streamlit_app.png)

---

# Docker Containerization

Docker was used to containerize the Flask application for reproducible deployment.

### Docker Screenshot

![Docker](screenshots/docker_container.png)

---

# Kubernetes Deployment

The Docker container was deployed on Kubernetes using:

* Deployment YAML
* Service YAML

### Kubernetes Screenshot

![Kubernetes](screenshots/kubernetes_pods.png)

---

# Jenkins CI/CD Pipeline

Jenkins was implemented to automate:

* Dependency installation
* Flask testing
* Docker image building
* Kubernetes deployment

### Jenkins Screenshot

![Jenkins](screenshots/jenkins_pipeline.png)

---

# Airflow DAG Orchestration

Apache Airflow was used to orchestrate the ML pipeline workflow.

### DAG Tasks

* Data Ingestion
* Model Training
* Model Evaluation
* Model Deployment

### Airflow Screenshot

![Airflow](screenshots/airflow_dag.png)

---

# How to Run the Project

## 1. Clone Repository

```bash
git clone YOUR_GITHUB_REPO_LINK
```

---

## 2. Create Conda Environment

```bash
conda create -n travel_mlops python=3.10
```

```bash
conda activate travel_mlops
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Run Flask API

```bash
cd flask_api
python app.py
```

---

## 5. Run Streamlit Application

```bash
cd streamlit_app
streamlit run app.py
```

---

## 6. Run MLflow

```bash
mlflow ui
```

---

## 7. Build Docker Container

```bash
docker build -t flight-price-api .
```

```bash
docker run -p 5001:5001 flight-price-api
```

---

## 8. Run Kubernetes Deployment

```bash
kubectl apply -f kubernetes/deployment.yaml
```

```bash
kubectl apply -f kubernetes/service.yaml
```

---

## 9. Run Airflow

```bash
docker compose up
```

---

# Future Improvements

* Real-time data integration
* Cloud deployment
* Automated retraining
* Monitoring dashboard
* Model drift detection

---

# Author

Shourya Kumar


```
```
