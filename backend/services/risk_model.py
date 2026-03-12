import joblib
import pandas as pd
import numpy as np
from utils.forecast import forecast_risk

MODEL_PATH = "models/risk_model_xgb.pkl"
risk_model = joblib.load(MODEL_PATH)

def predict_risk(features: dict):
    df = pd.DataFrame([features])
    risk_score = float(risk_model.predict(df)[0])
    history, preds = forecast_risk(risk_score, days=[30, 90])
    return {
        "current_risk": risk_score,
        "risk_30_day": preds.get(30, None),
        "risk_90_day": preds.get(90, None)
    }