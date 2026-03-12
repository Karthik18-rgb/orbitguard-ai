# 🛰️ OrbitGuard AI
### Space Debris Collision Risk Detection System

OrbitGuard is an AI-powered system designed to detect space debris from satellite imagery and estimate potential satellite collision risks using machine learning.

The system integrates **computer vision, machine learning, forecasting models, and API services** to simulate an automated orbital debris monitoring system.

---

# 🚀 Features

• Detects space debris using **YOLOv8 object detection**  
• Extracts debris features from satellite images  
• Predicts collision risk using **XGBoost regression model**  
• Forecasts **future collision risks**  
• Backend API built with **FastAPI**  
• Interactive frontend using **Streamlit**

---

# 🧠 System Pipeline

```
Satellite Image
      ↓
YOLOv8 Debris Detection
      ↓
Feature Extraction
      ↓
Risk Prediction (XGBoost)
      ↓
Risk Forecasting
      ↓
FastAPI Backend
      ↓
Streamlit Dashboard
```

---

# 🛠 Tech Stack

Python  
YOLOv8 (Ultralytics)  
FastAPI  
Streamlit  
XGBoost  
Scikit-learn  
NumPy / Pandas  
Matplotlib  

---

# 📂 Project Structure

```
orbitguard-ai/
│
├── backend/
│   ├── routes/
│   │   ├── detect.py
│   │   └── risk.py
│   │
│   ├── services/
│   │   ├── detector.py
│   │   └── risk_model.py
│   │
│   ├── main.py
│   └── schemas.py
├── data/
|   ├── processed 
│   └── raw
├── utils/
│   ├── extract_features.py
│   ├── forecast.py
│   └── test_detector.py
├── models/
│   └── risk_model_xgb.pkl
├── runs/
|   └── detect/
|        ├── models
|        ├── predict
|        └── train
├── utils/
|   ├── extract_features.py
|   ├── forecast.py
|   └── test_detector.py
├── app.py
├── train_detector.py
├── train_risk_model.py
└── requirements.txt
```
---

# ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/yourusername/orbitguard-ai.git
cd orbitguard-ai
```

Create virtual environment

```bash
python -m venv venv
```

Activate environment

Windows

```
venv\Scripts\activate
```

Install dependencies

```
pip install -r requirements.txt
```

---

# ▶️ Run Backend

```
python -m uvicorn backend.main:app --reload
```

API will start at

```
http://127.0.0.1:8000
```

Swagger docs

```
http://127.0.0.1:8000/docs
```

---

# ▶️ Run Frontend

```
streamlit run app.py
```

Upload satellite images to detect debris and evaluate collision risk.

---

# 📊 Example Output

The system provides:

• Detected debris objects  
• Collision risk score  
• 30-day risk forecast  
• 90-day risk forecast  
• Risk trend visualization  

---

# 🔮 Future Improvements

• Real orbital satellite trajectory data  
• Satellite maneuver simulation  
• Real-time debris tracking  
• Transformer-based risk prediction  
• Cloud deployment (AWS / GCP)

---

# 👨‍💻 Author

Karthik Singh

AI / Machine Learning Projects