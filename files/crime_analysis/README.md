# 🚔 Crime Analysis and Safety Prediction System

A complete end-to-end data science and web application project for analyzing crime data, predicting safety levels, and providing actionable insights through an interactive dashboard.

---

## 📁 Project Structure

```
crime_analysis/
├── data/
│   └── crime_safety_dataset.csv       # Raw dataset
├── src/
│   ├── preprocessing.py               # Data cleaning & feature engineering
│   ├── models.py                      # ML model training & evaluation
│   └── predictor.py                   # Inference helper
├── app/
│   └── dashboard.py                   # Streamlit interactive dashboard
├── models/
│   ├── best_model.pkl                 # Trained best ML model
│   ├── encoders.pkl                   # Label encoders for all features
│   ├── scaler.pkl                     # StandardScaler instance
│   └── feature_cols.pkl               # Ordered feature column list
├── reports/
│   ├── eda_dashboard.png              # EDA visualization export
│   ├── heatmap_crime_time.png         # Crime × time-of-day heatmap
│   ├── model_comparison.json          # All model metrics
│   └── best_model_report.txt          # Classification report
├── notebooks/
│   └── (place Jupyter EDA notebooks here)
├── train.py                           # One-click training script
├── requirements.txt                   # Python dependencies
└── README.md
```

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the models

```bash
python train.py
```

This will:
- Load and preprocess the dataset
- Engineer time, date, and severity features
- Train Logistic Regression, Random Forest, Gradient Boosting, and XGBoost
- Save the best model and all artifacts to `/models`
- Export evaluation reports to `/reports`

### 3. Launch the dashboard

```bash
streamlit run app/dashboard.py
```

Open your browser to `http://localhost:8501`

---

## 🧠 Machine Learning Pipeline

### Features Used
| Feature | Description |
|---------|-------------|
| `city_enc` | Encoded city identifier |
| `state_enc` | Encoded state identifier |
| `location_description_enc` | Encoded location type |
| `crime_type_enc` | Encoded crime category |
| `victim_gender_enc` | Encoded gender |
| `victim_race_enc` | Encoded race |
| `victim_age` | Victim age in years |
| `year / month / day` | Date components |
| `day_of_week` | 0 (Mon) – 6 (Sun) |
| `hour` | Hour of incident (0–23) |
| `is_weekend` | Binary flag |
| `severity_score` | Crime severity (1–5) |

### Target Variable
`safety_label`: **Safe** / **Moderate** / **Risky**

### Models Trained
| Model | Notes |
|-------|-------|
| Logistic Regression | Baseline linear model |
| Random Forest | Ensemble with GridSearchCV tuning |
| Gradient Boosting | Boosted ensemble |
| XGBoost | Gradient boosting with regularization |

### Evaluation Metrics
- Accuracy, Precision, Recall, F1-Score (weighted)
- Confusion matrix per model

---

## 📊 Dashboard Pages

| Page | Description |
|------|-------------|
| 📊 Dashboard | KPIs, crime distribution charts, gender/age breakdowns |
| 🗺️ Crime Map | Interactive Plotly scatter-map with safety color coding |
| 📈 Trends | Monthly trends, day-hour heatmap, weekday vs weekend |
| 🔮 Safety Prediction | Real-time form → ML prediction with confidence chart |
| 📋 Data Explorer | Filterable raw data, stats, model comparison table |
| 📚 Knowledge Center | Crime info, safety tips, emergency contacts, reporting guide |

---

## ☁️ Deployment

### Streamlit Cloud (recommended)
1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) → New App
3. Set main file: `app/dashboard.py`
4. Add `requirements.txt` at repo root

### Render.com
```yaml
# render.yaml
services:
  - type: web
    name: crime-analysis
    env: python
    buildCommand: pip install -r requirements.txt && python train.py
    startCommand: streamlit run app/dashboard.py --server.port $PORT
```

### Docker
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt && python train.py
EXPOSE 8501
CMD ["streamlit", "run", "app/dashboard.py", "--server.address", "0.0.0.0"]
```

---

## 📦 Dataset Description

| Column | Type | Description |
|--------|------|-------------|
| `id` | int | Unique incident identifier |
| `date` | str | Incident date (YYYY-MM-DD) |
| `time` | str | Incident time (HH:MM:SS) |
| `crime_type` | str | Category of crime |
| `city` | str | City of incident |
| `state` | str | State abbreviation |
| `location_description` | str | Location type |
| `victim_age` | int | Age of victim |
| `victim_gender` | str | Gender of victim |
| `victim_race` | str | Race of victim |

---

## 📞 Emergency Contacts (US)

| Service | Contact |
|---------|---------|
| Police / Fire / Medical | **911** |
| Non-emergency Police | **311** |
| Domestic Violence Hotline | **1-800-799-7233** |
| Crisis / Mental Health | **988** |
| FBI Tips | **1-800-CALL-FBI** |
| FTC Fraud | **1-877-382-4357** |

---

## 🛠️ Tech Stack

- **Language**: Python 3.10+
- **ML**: scikit-learn, XGBoost
- **Visualization**: Plotly, Matplotlib, Seaborn
- **Dashboard**: Streamlit
- **Data**: Pandas, NumPy
- **Serialization**: Joblib

---

## 📄 License

For educational and research purposes only.
