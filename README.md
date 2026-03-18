# 💼 AI Salary Prediction System

An end-to-end Machine Learning system that predicts salaries for AI-related job roles based on job attributes, built with XGBoost, deployed via FastAPI, and containerized using Docker.

---

## 📌 Project Overview

This project demonstrates a complete ML lifecycle:

- Data preprocessing & feature engineering
- Model training using XGBoost
- Model evaluation
- Inference pipeline
- REST API using FastAPI
- Docker containerization for deployment

---

## 🧠 Problem Statement

Predict the expected salary of AI-related job roles using features such as:

- Job category
- Experience level
- Education
- Location
- Industry
- Market demand indicators

---

## 🏗️ Project Structure
```
END-TO-END/
│
├── api/
│   └── app.py
│
├── Model/
│   ├── ai_jobs_market_2025_2026.csv
│   ├── train.ipynb
│   ├── eval.py
│   ├── test.csv
│   ├── xgb_salary_pipeline_best.pkl
│   ├── old_xgb_salary_pipeline_best.pkl
│   └── OneHotEncoder_categories.pkl
│
├── Dockerfile
├── requirements.txt
├── .dockerignore
└── .gitignore
```
---

## ⚙️ Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn
- XGBoost
- FastAPI
- Uvicorn
- Docker

---

## 📊 Model Performance

- R² Score: ~0.88

---

## 🚀 Running Locally

pip install -r requirements.txt  
uvicorn api.app:app --reload  

Open: http://127.0.0.1:8000/docs

---

## 🐳 Run Using Docker

docker pull shaggybagels/ai_salary_prediction:latest  
docker run -p 8000:8000 shaggybagels/ai_salary_prediction  

Open: http://localhost:8000/docs

---

## 📥 Example Input
```
{
  "job_title": "AI Engineer",
  "job_category": "Machine Learning",
  "experience_level": "Senior",
  "years_of_experience": 5,
  "education_required": "Masters",
  "country": "USA",
  "remote_work": "Remote",
  "company_size": "Enterprise",
  "industry": "Technology",
  "ai_salary_premium_pct": 20,
  "demand_score": 0.85,
  "demand_growth_yoy_pct": 12.5,
  "benefits_score_10": 8,
  "posting_year": 2025,
  "posting_month": 6,
  "is_senior": 1,
  "is_remote_friendly": 1,
  "is_llm_role": 1
}
```
---

## 📤 Example Output
```
{
  "predicted_salary": 182345.23
}
```
---

## 📦 Docker Image

https://hub.docker.com/r/shaggybagels/ai_salary_prediction

---

## 💡 Key Features

✔ End-to-end ML pipeline  
✔ Feature engineering & encoding  
✔ FastAPI-based inference  
✔ Dockerized deployment  

---
