"""
Prediction helper – takes raw user inputs and returns safety label.
"""
import numpy as np
import pandas as pd


SEVERITY_MAP = {
    "Homicide": 5, "Robbery": 4, "Assault": 4, "Arson": 4,
    "Domestic Violence": 3, "Burglary": 3, "Drug Offense": 2,
    "Theft": 2, "Vandalism": 1, "Fraud": 1,
}

TIME_OF_DAY_MAP = {
    (0, 5): "Night", (6, 11): "Morning", (12, 16): "Afternoon",
    (17, 20): "Evening", (21, 23): "Late Night",
}

DAY_NAMES = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
MONTH_NAMES = ["January","February","March","April","May","June",
               "July","August","September","October","November","December"]


def _get_time_of_day(hour: int) -> str:
    for (lo, hi), label in TIME_OF_DAY_MAP.items():
        if lo <= hour <= hi:
            return label
    return "Night"


def _safe_encode(le, value, default=0):
    """Encode a value, returning default if unseen."""
    try:
        return int(le.transform([value])[0])
    except Exception:
        return default


def predict_safety(
    model, encoders, scaler, feature_cols,
    city: str, state: str, location: str,
    crime_type: str, victim_gender: str, victim_race: str,
    victim_age: int, date_str: str, time_str: str,
) -> dict:
    dt = pd.to_datetime(date_str, errors="coerce")
    tm = pd.to_datetime(time_str, format="%H:%M", errors="coerce")
    hour = tm.hour if not pd.isnull(tm) else 12
    day_of_week = dt.dayofweek if not pd.isnull(dt) else 0
    is_weekend = int(day_of_week in [5, 6])
    severity = SEVERITY_MAP.get(crime_type, 2)
    day_name = dt.day_name() if not pd.isnull(dt) else "Monday"
    month_name = dt.month_name() if not pd.isnull(dt) else "January"

    row = {
        "city_enc":                   _safe_encode(encoders["city"], city),
        "state_enc":                  _safe_encode(encoders["state"], state),
        "location_description_enc":   _safe_encode(encoders["location_description"], location),
        "crime_type_enc":             _safe_encode(encoders["crime_type"], crime_type),
        "victim_gender_enc":          _safe_encode(encoders["victim_gender"], victim_gender),
        "victim_race_enc":            _safe_encode(encoders["victim_race"], victim_race),
        "victim_age":                 victim_age,
        "year":                       dt.year if not pd.isnull(dt) else 2024,
        "month":                      dt.month if not pd.isnull(dt) else 1,
        "day":                        dt.day if not pd.isnull(dt) else 1,
        "day_of_week":                day_of_week,
        "hour":                       hour,
        "is_weekend":                 is_weekend,
        "severity_score":             severity,
    }

    X = np.array([[row[c] for c in feature_cols]])
    X_scaled = scaler.transform(X)
    pred_enc = model.predict(X_scaled)[0]
    label_names = list(encoders["safety_label"].classes_)
    label = label_names[pred_enc]

    # Probability if model supports it
    try:
        proba = model.predict_proba(X_scaled)[0]
        proba_dict = {label_names[i]: round(float(p)*100, 1) for i, p in enumerate(proba)}
    except Exception:
        proba_dict = {label: 100.0}

    icon_map = {"Safe": "🟢", "Moderate": "🟡", "Risky": "🔴"}
    return {
        "label": label,
        "icon": icon_map.get(label, "⚪"),
        "probabilities": proba_dict,
        "severity_score": severity,
        "time_of_day": _get_time_of_day(hour),
        "is_weekend": bool(is_weekend),
    }
