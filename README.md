# 🧠 Employee Attrition Prediction & Workforce Analytics Platform

> An end-to-end ML-powered HR analytics platform that predicts which employees are at risk of leaving, explains *why*, and quantifies the business cost — helping HR teams act before it's too late.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green?logo=fastapi)
![React](https://img.shields.io/badge/React-18+-61DAFB?logo=react)
![MongoDB](https://img.shields.io/badge/MongoDB-Atlas-47A248?logo=mongodb)
![Power BI](https://img.shields.io/badge/Power%20BI-Dashboard-F2C811?logo=powerbi)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 📌 Problem Statement

Employee turnover costs companies an estimated **50–200% of an employee's annual salary** in recruiting, onboarding, and lost productivity. Most organizations only react *after* someone resigns — this platform shifts that to proactive retention by identifying at-risk employees early, explaining the key drivers, and ranking them by financial impact so HR knows exactly who to talk to first.

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 🎯 **Risk Prediction** | Classifies employees as Low / Medium / High attrition risk |
| 💡 **Explainability** | SHAP-based factor breakdown per employee ("why is this person flagged?") |
| 💰 **ROI Prioritization** | Ranks at-risk employees by expected cost of attrition vs. intervention |
| 🔮 **What-If Simulator** | Live sliders (salary hike, overtime, role change) that update risk score in real time |
| 📤 **Bulk Scoring** | CSV upload → scored results table, exportable to Excel |
| 📊 **Power BI Dashboard** | Leadership-level attrition trends, cost heatmaps, and top drivers |
| 📋 **Intervention Logging** | HR can log retention actions and track outcomes per employee |
| 🔐 **Role-Based Access** | Admin (company-wide) vs. Manager (own team only) via JWT auth |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     React Frontend                       │
│   Dashboard · Employee Table · What-If Simulator        │
└───────────────────────┬─────────────────────────────────┘
                        │ REST API
┌───────────────────────▼─────────────────────────────────┐
│                   FastAPI Backend                        │
│   /predict · /predict/bulk · /employees · /auth         │
└──────────┬────────────────────────────┬─────────────────┘
           │                            │
┌──────────▼──────────┐    ┌───────────▼─────────────────┐
│      MongoDB        │    │        ML Module             │
│  Employees · Preds  │    │  XGBoost · Random Forest    │
│  Interventions      │    │  SHAP · Calibration         │
└─────────────────────┘    └─────────────────────────────┘
                                        │
                           ┌────────────▼────────────────┐
                           │   Power BI Dashboard         │
                           │   Excel Monthly Reports      │
                           └─────────────────────────────┘
```

---

## 📂 Project Structure

```
employee-attrition-platform/
├── data/
│   └── README.md               # ← Dataset download instructions
│
├── notebooks/
│   ├── 01_eda.ipynb            # Exploratory Data Analysis
│   ├── 02_preprocessing.ipynb  # Feature engineering
│   └── 03_modeling.ipynb       # Model training & comparison
│
├── ml/
│   ├── preprocess.py           # Preprocessing pipeline
│   ├── train.py                # Model training script
│   ├── evaluate.py             # Evaluation & metrics report
│   ├── predict.py              # Inference helper
│   └── saved_models/           # Trained model artifacts (gitignored)
│
├── backend/
│   └── app/
│       ├── main.py             # FastAPI app entry point
│       ├── routes/             # API route handlers
│       ├── schemas/            # Pydantic request/response models
│       └── db.py               # MongoDB connection
│
├── frontend/                   # React app
│   └── src/
│       ├── pages/              # Dashboard, Employee Detail, Bulk Upload
│       └── components/         # RiskBadge, SHAPChart, WhatIfSimulator
│
├── powerbi/
│   └── dashboard.pbix          # Power BI report file
│
├── reports/
│   └── monthly_report.py       # Automated Excel report (openpyxl)
│
├── .gitignore
├── requirements.txt
├── docker-compose.yml
└── README.md
```

---

## 🤖 ML Pipeline

### Dataset
**IBM HR Analytics Employee Attrition & Performance** — 1,470 employees, 35 features.
Download from [Kaggle](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset) and place in `data/raw/`.

### Models Trained

| Model | Type | Purpose |
|---|---|---|
| Logistic Regression | Baseline | Classification benchmark |
| Random Forest | Bagging | Attrition classification |
| XGBoost | Boosting | Attrition classification (best) |
| Random Forest Regressor | Regression | Expected remaining tenure |

### Evaluation Metrics (XGBoost — best model)

| Metric | Score |
|---|---|
| Accuracy | 88.46% |
| Precision | — |
| Recall | — |
| F1 Score | — |
| ROC-AUC | — |
| PR-AUC | — |

> ⚠️ *Fill in Precision / Recall / F1 / ROC-AUC / PR-AUC from your `evaluate.py` output. These matter more than accuracy for imbalanced data.*

### Key Techniques
- Class imbalance handling via `class_weight='balanced'`
- Hyperparameter tuning with `GridSearchCV` (5-fold Stratified CV)
- SHAP values for per-prediction explainability
- Probability calibration for reliable ROI score calculation
- Optimal decision threshold tuned for maximum F1 (not default 0.5)

### ROI Prioritization Formula
```
priority_score = (attrition_probability × estimated_replacement_cost)
               − estimated_intervention_cost
```
Employees are ranked by `priority_score` so HR acts on the highest-impact cases first.

---

## 🔑 Key EDA Findings

- Employees working **overtime churn at 3× the rate** of non-overtime employees
- Attrition is highest in the **first 2 years** of tenure, then stabilizes significantly
- **Sales department** has the highest attrition rate across all departments
- Low **job satisfaction** and **environment satisfaction** are strong leading indicators
- Monthly income below a certain threshold correlates strongly with flight risk

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- Node.js 18+
- MongoDB (local or Atlas)

### 1. Clone the repo
```bash
git clone https://github.com/Abdul-Hamid-k/Employee-Attrition-Prediction-Workforce-Analytics-Platform.git
cd Employee-Attrition-Prediction-Workforce-Analytics-Platform
```

### 2. Set up Python environment
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Download the dataset
See `data/README.md` for instructions.

### 4. Train the model
```bash
python ml/train.py
```

### 5. Run evaluation report
```bash
python ml/evaluate.py
```

### 6. Start the backend
```bash
cd backend
uvicorn app.main:app --reload
# API docs at http://localhost:8000/docs
```

### 7. Start the frontend
```bash
cd frontend
npm install
npm start
# App at http://localhost:3000
```

### Or run everything with Docker
```bash
docker-compose up --build
```

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/auth/login` | Login and receive JWT token |
| `GET` | `/employees` | List employees with risk scores |
| `GET` | `/employees/{id}` | Employee detail + risk history |
| `POST` | `/predict` | Single employee prediction |
| `POST` | `/predict/bulk` | Bulk CSV prediction |
| `POST` | `/intervention` | Log a retention action |

> Full interactive docs available at `/docs` after starting the backend.

---

## 📊 Power BI Dashboard

The `powerbi/dashboard.pbix` file includes:
- Attrition trend over time by department
- Revenue at risk heatmap
- Top attrition drivers (aggregated SHAP values)
- Risk tier distribution across teams

---

## 🛠️ Tech Stack

**ML & Data:** Python, Pandas, NumPy, Scikit-learn, XGBoost, SHAP, Matplotlib, Seaborn

**Backend:** FastAPI, PyMongo, Pydantic, JWT Auth

**Frontend:** React 18, Tailwind CSS, Recharts

**Database:** MongoDB

**BI & Reporting:** Power BI, Excel (openpyxl)

**DevOps:** Docker, GitHub Actions, Render (backend), Vercel (frontend)

---

## 🗺️ Roadmap

- [x] EDA & data preprocessing
- [x] Model training (Logistic Regression, Random Forest, XGBoost)
- [x] Hyperparameter tuning & evaluation
- [x] SHAP explainability
- [x] ROI prioritization engine
- [ ] FastAPI backend
- [ ] React frontend
- [ ] Power BI dashboard
- [ ] Docker + deployment
- [ ] Survival analysis (Kaplan-Meier)
- [ ] Automated drift monitoring
- [ ] Email/Slack alerts on threshold breach

---

## 👤 Author

**Abdul Hamid Khatri**
Senior Software Engineer | ML & Data Science

[![GitHub](https://img.shields.io/badge/GitHub-Abdul--Hamid--k-181717?logo=github)](https://github.com/Abdul-Hamid-k)

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
