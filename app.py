import streamlit as st
import numpy as np
import pandas as pd
import joblib
import requests
from utils.forecast import forecast_risk, plot_forecast
import matplotlib.pyplot as plt

# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="ORBITGUARD",
    layout="wide"
)

# 👇 IMPORTANT: Turn OFF backend for deployment
USE_BACKEND = False
API_URL = "http://127.0.0.1:8000/api/detect"

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_risk_model():
    return joblib.load("models/risk_model_xgb.pkl")

risk_model = load_risk_model()

# ---------------- UI ----------------
st.title("🛰️ ORBITGUARD – Space Debris Collision Risk System")

st.sidebar.header("System Info")
st.sidebar.write("YOLOv8 (Backend)")
st.sidebar.write("XGBoost Risk Model")
st.sidebar.write("FastAPI + Streamlit")

if not USE_BACKEND:
    st.info("🚀 Running in demo mode (backend disabled for cloud deployment)")

uploaded_file = st.file_uploader(
    "Upload Satellite Image",
    type=["jpg", "jpeg", "png"]
)

# ---------------- MAIN LOGIC ----------------
if uploaded_file is not None:

    st.subheader("🧠 Debris Detection Results")

    if USE_BACKEND:
        # ---------- CALL BACKEND ----------
        files = {
            "file": (
                uploaded_file.name,
                uploaded_file.getvalue(),
                uploaded_file.type
            )
        }

        with st.spinner("Running debris detection via backend..."):
            response = requests.post(API_URL, files=files)

        if response.status_code != 200:
            st.error("Backend detection failed")
            st.stop()

        data = response.json()

        debris_count = data["debris_count"]
        avg_area = data["avg_area"]
        max_area = data["max_area"]

    else:
        # ---------- MOCK DATA (FOR DEPLOYMENT) ----------
        debris_count = np.random.randint(5, 50)
        avg_area = np.random.uniform(10, 100)
        max_area = np.random.uniform(100, 500)

    # ---------- DISPLAY METRICS ----------
    col1, col2, col3 = st.columns(3)
    col1.metric("Debris Count", debris_count)
    col2.metric("Avg Area", f"{avg_area:.2f}")
    col3.metric("Max Area", f"{max_area:.2f}")

    # ---------- RISK MODEL ----------
    st.subheader("⚠️ Collision Risk Assessment")

    orbital_altitude_km = np.random.uniform(300, 1500)
    velocity_kms = np.random.uniform(2, 14)

    features = pd.DataFrame([{
        "debris_count": debris_count,
        "avg_area": avg_area,
        "max_area": max_area,
        "orbital_altitude_km": orbital_altitude_km,
        "velocity_kms": velocity_kms
    }])

    risk_score = float(risk_model.predict(features)[0])

    st.metric("Current Collision Risk", f"{risk_score:.2f}")

    # ---------- FORECAST ----------
    st.subheader("📈 Risk Forecast")

    history, preds = forecast_risk(
        current_risk=risk_score,
        days=[30, 90]
    )

    col1, col2 = st.columns(2)
    col1.metric("30-Day Risk", f"{preds[30]:.2f}")
    col2.metric("90-Day Risk", f"{preds[90]:.2f}")

    fig = plot_forecast(history, preds, None)
    st.pyplot(fig)

    # ---------- RECOMMENDATIONS ----------
    st.subheader("🛡️ Safety Recommendation")

    if risk_score < 0.3:
        st.success("Low risk. No immediate action required.")
    elif risk_score < 0.6:
        st.warning("Moderate risk. Monitor trajectory closely.")
    else:
        st.error("High risk! Collision avoidance maneuver recommended.")