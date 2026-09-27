# ⚡ AI Energy Auditor: TOU Simulator

An end-to-end Machine Learning Operations (MLOps) pipeline and web application designed to simulate Time-of-Use (TOU) energy billing and predict peak load consumption. Built with a focus on enterprise-grade architecture for energy domain applications.

## 🏗️ System Architecture

This project implements a 4-tier architecture:
1. **Model Training & Registry**: Databricks Unity Catalog & PySpark.
2. **Experiment Tracking**: MLflow (v3.16.0).
3. **Backend API**: FastAPI with strict Pydantic payload validation to ensure data integrity and prevent system crashes (HTTP 422 handling).
4. **Frontend UI**: Streamlit application integrated via RESTful API.

## 🚀 Tech Stack
* **Language**: Python 3.14.4
* **Engine**: OpenJDK 25, Hadoop (winutils.exe), PySpark 4.2.0
* **Serving**: FastAPI, Uvicorn, Requests
* **Validation**: Pydantic

## ⚙️ How to Run Locally

### 1. Start the Backend API (FastAPI)
Activate your virtual environment and run the Uvicorn server:
```bash
uvicorn main:app --reload