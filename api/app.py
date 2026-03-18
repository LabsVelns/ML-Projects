from fastapi import FastAPI
from pydantic import BaseModel
from Model.eval import load_model, preprocess, predict
import pandas as pd

app = FastAPI()
model = load_model()

class JobInput(BaseModel):
    job_title: str
    job_category: str
    experience_level: str
    years_of_experience: float
    education_required: str
    country: str
    remote_work: str
    company_size: str
    industry: str
    ai_salary_premium_pct: float
    demand_score: int
    demand_growth_yoy_pct: float
    benefits_score_10: int
    posting_year: int
    posting_month: int
    is_senior: int
    is_remote_friendly: int
    is_llm_role: int


@app.post("/predict_salary")
def predict_salary(job_input: JobInput):
    input_data = job_input.dict()
    df = pd.DataFrame([input_data])
    salary_prediction = predict(model, df)
    return {"predicted_salary": float(salary_prediction)}
