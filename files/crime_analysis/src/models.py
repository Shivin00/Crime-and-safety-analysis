"""
Crime Safety – Model Training & Evaluation
"""
import numpy as np
import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix,
)

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False


def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size,
                            random_state=random_state, stratify=y)


def evaluate(model, X_test, y_test, label_names=None):
    y_pred = model.predict(X_test)
    return {
        "accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, average="weighted", zero_division=0), 4),
        "recall":    round(recall_score(y_test, y_pred, average="weighted", zero_division=0), 4),
        "f1":        round(f1_score(y_test, y_pred, average="weighted", zero_division=0), 4),
        "report":    classification_report(y_test, y_pred, target_names=label_names, zero_division=0),
        "confusion": confusion_matrix(y_test, y_pred),
        "y_pred":    y_pred,
    }


def train_all_models(X_train, y_train, X_test, y_test, label_names=None):
    results = {}

    # 1. Logistic Regression
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    results["Logistic Regression"] = {"model": lr, **evaluate(lr, X_test, y_test, label_names)}

    # 2. Random Forest (with simple tuning)
    rf_params = {"n_estimators": [100, 200], "max_depth": [None, 10]}
    rf_base = RandomForestClassifier(random_state=42)
    rf_gs = GridSearchCV(rf_base, rf_params, cv=3, scoring="f1_weighted", n_jobs=-1)
    rf_gs.fit(X_train, y_train)
    best_rf = rf_gs.best_estimator_
    results["Random Forest"] = {"model": best_rf, **evaluate(best_rf, X_test, y_test, label_names)}

    # 3. Gradient Boosting
    gb = GradientBoostingClassifier(n_estimators=150, learning_rate=0.1,
                                    max_depth=4, random_state=42)
    gb.fit(X_train, y_train)
    results["Gradient Boosting"] = {"model": gb, **evaluate(gb, X_test, y_test, label_names)}

    # 4. XGBoost (if available)
    if HAS_XGB:
        xgb = XGBClassifier(n_estimators=150, learning_rate=0.1,
                             max_depth=4, random_state=42,
                             eval_metric="mlogloss", verbosity=0)
        xgb.fit(X_train, y_train)
        results["XGBoost"] = {"model": xgb, **evaluate(xgb, X_test, y_test, label_names)}

    return results


def get_best_model(results):
    best_name = max(results, key=lambda k: results[k]["f1"])
    return best_name, results[best_name]["model"]


def save_artifacts(models_dir, best_model, encoders, scaler, feature_cols):
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(best_model, os.path.join(models_dir, "best_model.pkl"))
    joblib.dump(encoders,   os.path.join(models_dir, "encoders.pkl"))
    joblib.dump(scaler,     os.path.join(models_dir, "scaler.pkl"))
    joblib.dump(feature_cols, os.path.join(models_dir, "feature_cols.pkl"))
    print(f"Artifacts saved to {models_dir}")


def load_artifacts(models_dir):
    model      = joblib.load(os.path.join(models_dir, "best_model.pkl"))
    encoders   = joblib.load(os.path.join(models_dir, "encoders.pkl"))
    scaler     = joblib.load(os.path.join(models_dir, "scaler.pkl"))
    feat_cols  = joblib.load(os.path.join(models_dir, "feature_cols.pkl"))
    return model, encoders, scaler, feat_cols
