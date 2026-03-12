import pandas as pd
import xgboost as xgb
import joblib
import os

DATA_PATH = "data/processed/risk_features.csv"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "risk_model_xgb.pkl")
os.makedirs(MODEL_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)

risk_score = (
    0.4 * df["debris_count"] +
    0.3 * df["velocity_kms"] +
    0.2 * df["avg_area"] - 
    0.0003 * df["orbital_altitude_km"]
   )

risk_score = (risk_score - risk_score.min()) / (risk_score.max() - risk_score.min())
X = df[[
    "debris_count",
    "avg_area",
    "max_area",
    "orbital_altitude_km",
    "velocity_kms"
]]
y = risk_score
model = xgb.XGBRegressor(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    objective="reg:squarederror",
    random_state=42,
    n_jobs=-1
)

model.fit(X, y)

joblib.dump(model, MODEL_PATH)
print(f"\nXGBoost risk model saved to: {MODEL_PATH}")