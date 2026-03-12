import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import random

def forecast_risk(current_risk, days=[30, 90]):
    np.random.seed(42)
    time = np.arange(0, 60)
    noise = np.random.normal(0, 0.02, size=len(time))

    drift = np.linspace(-0.05, 0.05, len(time))
    risk_history = np.clip(current_risk + drift + noise, 0, 1)

    model = LinearRegression()
    model.fit(time.reshape(-1, 1), risk_history)

    forecasts = {}
    for d in days:
        future_time = np.array([[60 + d]])
        pred = model.predict(future_time)[0]
        forecasts[d] = float(np.clip(pred, 0, 1))
    return risk_history, forecasts

def plot_forecast(risk_history, forecasts, ax):
    days_past = np.arange(len(risk_history))

    # forecast points
    days_future = [len(risk_history) + d for d in forecasts.keys()]
    future_values = list(forecasts.values())

    # past trend
    ax.plot(days_past, risk_history, label="Past Risk", color="blue")

    # forecast points
    ax.scatter(days_future, future_values,
               color="red",
               label="Forecast",
               s=100)

    ax.set_xlabel("Days")
    ax.set_ylabel("Collision Risk")
    ax.set_title("Collision Risk Forecast")
    ax.legend()
    ax.grid(True)
    
if __name__ == "__main__":
    history, preds = forecast_risk(current_risk=0.42)
    plot_forecast(history, preds)