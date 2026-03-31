# ⚙️ Secure MLOps Pipeline

> A production ready ML pipeline demonstrating best practices for model deployment, versioning, security, and monitoring.

---

## Overview

This project implements a **full MLOps lifecycle** for a machine learning model from data versioning and experiment tracking to containerized deployment and CI/CD automation. Every component is built with a security first mindset, making it a practical reference for teams running ML in production.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        ML Pipeline                          │
│                                                             │
│  Data (DVC) ──► Train ──► Evaluate ──► Register (MLflow)    │
│                                              │              │
│                                         Deploy (Docker)     │
│                                              │              │
│                                         Monitor + Alert     │
└─────────────────────────────────────────────────────────────┘
```

---

## Features

- **Data Versioning** — DVC tracks datasets and pipeline stages with full reproducibility
- **Experiment Tracking** — MLflow logs all runs, parameters, metrics, and model artifacts
- **Model Deployment** — FastAPI inference server containerized with Docker
- **CI/CD Automation** — GitHub Actions pipeline runs training, evaluation, and deployment on push
- **Monitoring** — Drift detection and alerting integrated into the serving layer
- **Security** — Secrets management, dependency pinning, and scan-ready Dockerfile

---

## Project Structure

```
mlops-pipeline/
├── .github/workflows/     # CI/CD pipeline definitions
├── .dvc/                  # DVC config and cache pointers
├── data/                  # DVC-tracked datasets
├── deployment/            # Docker and serving configs
├── models/                # Trained model artifacts
├── monitoring/            # Drift and performance monitoring
├── outputs/               # Evaluation results and reports
├── scripts/               # Utility scripts
├── train.py               # Model training entrypoint
├── evaluate.py            # Evaluation script
├── deploy.py              # Deployment script
├── dvc.yaml               # DVC pipeline definition
└── requirements.txt
```

---

## Tech Stack

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-Experiment_Tracking-0194E2?style=flat&logo=mlflow&logoColor=white)
![DVC](https://img.shields.io/badge/DVC-Data_Versioning-945DD6?style=flat)
![Docker](https://img.shields.io/badge/Docker-Containerization-2496ED?style=flat&logo=docker&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-Serving-009688?style=flat&logo=fastapi&logoColor=white)
![GitHub Actions](https://img.shields.io/badge/GitHub_Actions-CI/CD-2088FF?style=flat&logo=githubactions&logoColor=white)

---

## Getting Started

```bash
git clone https://github.com/shoumik27/mlops-pipeline.git
cd mlops-pipeline
pip install -r requirements.txt
```

**Run the pipeline:**
```bash
dvc repro          # Reproduce full pipeline (data → train → evaluate)
mlflow ui          # View experiment tracking at localhost:5000
```

**Deploy the model:**
```bash
python deploy.py
```

**Run with Docker:**
```bash
docker build -t mlops-pipeline .
docker run -p 8000:8000 mlops-pipeline
```

---

## Author

**Shoumik Chandra** — AI Security Researcher   
[GitHub](https://github.com/shoumik27)
