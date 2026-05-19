# Travel MLOps Project

## Project Overview

This project is an end-to-end MLOps pipeline for travel flight price prediction.  
The system includes model training, experiment tracking, API deployment, Docker containerization, and future orchestration using Kubernetes, Jenkins, and Airflow.

---

## Features

- Flight price prediction using Machine Learning
- MLflow experiment tracking
- Flask REST API deployment
- Docker containerization
- GitHub version control
- Streamlit frontend (Upcoming)
- Kubernetes deployment (Upcoming)
- Jenkins CI/CD pipeline (Upcoming)
- Airflow orchestration (Upcoming)

---

## Project Structure

```bash
travel-mlops-project/
│
├── data/                  # Dataset files
├── notebooks/             # Model training notebooks
├── models/                # Saved ML models
├── flask_api/             # Flask deployment API
│   ├── app.py
│   ├── Dockerfile
│   ├── requirements.txt
│   └── models/
│
├── screenshots/           # Project screenshots
├── kubernetes/            # Kubernetes YAML files
├── airflow/               # Airflow DAGs
├── streamlit_app/         # Streamlit frontend
├── tests/                 # Testing files
└── README.md
```

---

## Tech Stack

- Python
- Scikit-learn
- XGBoost
- Flask
- MLflow
- Docker
- Git & GitHub
- Kubernetes
- Jenkins
- Airflow
- Streamlit

---

## Machine Learning Models

The following models were trained and evaluated:

1. Linear Regression
2. Random Forest Regressor
3. XGBoost Regressor

---

## MLflow Experiment Tracking

MLflow was used for:

- Experiment tracking
- Model comparison
- Metric logging
- Artifact management

---

## Flask API Endpoints

### Home Endpoint

```http
GET /
```

Response:

```json
{
    "message": "Flight Price Prediction API Running"
}
```

---

### Prediction Endpoint

```http
POST /predict
```

Sample Request:

```json
{
    "from": "Recife (PE)",
    "to": "Florianopolis (SC)",
    "flightType": "firstClass",
    "time": 1.76,
    "distance": 676.53,
    "agency": "FlyingDrops",
    "year": 2019,
    "month": 9,
    "day": 26,
    "weekday": 3
}
```

Sample Response:

```json
{
    "predicted_price": XXXX 
}
```

---

## Docker Setup

Build Docker Image:

```bash
docker build -t flight-price-api .
```

Run Docker Container:

```bash
docker run -p 5001:5001 flight-price-api
```

---



Shourya Kumar
