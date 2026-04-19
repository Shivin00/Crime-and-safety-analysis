#!/usr/bin/env python3
"""
train.py – Run this script once to preprocess data and train all models.
Saves trained artifacts to the /models directory.

Usage:
    python train.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from preprocessing import full_pipeline
from models import split_data, train_all_models, get_best_model, save_artifacts
import json

DATA_PATH   = os.path.join(os.path.dirname(__file__), "data", "crime_safety_dataset.csv")
MODELS_DIR  = os.path.join(os.path.dirname(__file__), "models")
REPORTS_DIR = os.path.join(os.path.dirname(__file__), "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

print("=" * 55)
print("  Crime Analysis – Model Training Pipeline")
print("=" * 55)

print("\n[1/4] Loading & preprocessing data...")
df, X, y, encoders, feature_cols, scaler = full_pipeline(DATA_PATH)
label_names = list(encoders["safety_label"].classes_)
print(f"      Dataset shape: {df.shape} | Labels: {label_names}")

print("\n[2/4] Splitting into train/test sets...")
X_train, X_test, y_train, y_test = split_data(X, y)
print(f"      Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

print("\n[3/4] Training models (this may take ~1 min)...")
results = train_all_models(X_train, y_train, X_test, y_test, label_names)

print("\n=== Model Results ===")
for name, res in results.items():
    print(f"  {name:25s} Acc={res['accuracy']:.4f}  F1={res['f1']:.4f}")

best_name, best_model = get_best_model(results)
print(f"\n  ✅ Best model: {best_name}")

print("\n[4/4] Saving artifacts...")
save_artifacts(MODELS_DIR, best_model, encoders, scaler, feature_cols)

# Save comparison report
summary = {}
for name, res in results.items():
    summary[name] = {k: float(v) if hasattr(v, '__float__') else v
                     for k, v in res.items()
                     if k not in ("model","report","confusion","y_pred")}
with open(os.path.join(REPORTS_DIR, "model_comparison.json"), "w") as f:
    json.dump(summary, f, indent=2)

with open(os.path.join(REPORTS_DIR, "best_model_report.txt"), "w") as f:
    f.write(f"Best Model: {best_name}\n\n")
    f.write(results[best_name]["report"])

print("\n✅ Training complete! Artifacts saved to /models")
print("   Run the dashboard with: streamlit run app/dashboard.py")
