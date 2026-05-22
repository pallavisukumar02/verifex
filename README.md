# FraudVerse 🔴
Real-Time AI Fraud Detection & Explainable Risk Intelligence
> ZYNEX Hackathon — Track 1: AI/ML | 24 Hours | Team of 3

---

## What It Does
FraudVerse scans every transaction in **under 200ms** and tells you if it's fraud — and **exactly why**.

---

## Project Structure
```
fraudverse/
├── ml/
│   └── train_model.py     ← Train the model
├── backend/
│   └── api.py             ← FastAPI backend
└── dashboard/
    └── index.html         ← Live dashboard (open in browser)
```

---

## Quick Setup

```bash
# 1. Install
pip install pandas numpy scikit-learn xgboost shap imbalanced-learn joblib fastapi uvicorn

# 2. Train model (run in Google Colab)
# Upload creditcard.csv from kaggle.com/datasets/mlg-ulb/creditcardfraud
cd ml/ && python train_model.py

# 3. Start API
cd backend/ && python api.py

# 4. Open dashboard
# Open dashboard/index.html in browser
```

---

## Tech Stack
| Layer | Tech |
|---|---|
| ML Model | XGBoost + SMOTE |
| Explainability | SHAP |
| Backend | FastAPI |
| Frontend | HTML + Chart.js |
| Dataset | Kaggle Credit Card Fraud (284,807 transactions) |

---

## Model Results
- ROC-AUC: **0.9731**
- F1-Score: **0.8842**
- Latency: **<200ms**
