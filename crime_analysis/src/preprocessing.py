"""
Crime Safety Dataset - Data Preprocessing Module
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler


def load_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Drop duplicates
    df.drop_duplicates(inplace=True)
    # Strip whitespace in string columns
    str_cols = df.select_dtypes(include="object").columns
    for col in str_cols:
        df[col] = df[col].str.strip()
    # Fill missing values
    df["victim_age"].fillna(df["victim_age"].median(), inplace=True)
    df.fillna("Unknown", inplace=True)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Parse date
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["day_of_week"] = df["date"].dt.dayofweek          # 0=Mon, 6=Sun
    df["day_name"] = df["date"].dt.day_name()
    df["month_name"] = df["date"].dt.month_name()
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    # Parse time
    df["time"] = pd.to_datetime(df["time"], format="%H:%M:%S", errors="coerce")
    df["hour"] = df["time"].dt.hour
    df["time_of_day"] = pd.cut(
        df["hour"],
        bins=[-1, 5, 11, 16, 20, 23],
        labels=["Night", "Morning", "Afternoon", "Evening", "Late Night"],
    )

    # Severity score
    severity_map = {
        "Homicide": 5,
        "Robbery": 4,
        "Assault": 4,
        "Arson": 4,
        "Domestic Violence": 3,
        "Burglary": 3,
        "Drug Offense": 2,
        "Theft": 2,
        "Vandalism": 1,
        "Fraud": 1,
    }
    df["severity_score"] = df["crime_type"].map(severity_map).fillna(2)

    # Safety label  (target)
    df["safety_label"] = pd.cut(
        df["severity_score"],
        bins=[0, 1.5, 3.5, 5],
        labels=["Safe", "Moderate", "Risky"],
    )
    return df


def encode_features(df: pd.DataFrame):
    df = df.copy()
    cat_cols = ["crime_type", "city", "state", "location_description",
                 "victim_gender", "victim_race", "time_of_day", "day_name",
                 "month_name"]
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col + "_enc"] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    target_le = LabelEncoder()
    df["safety_label_enc"] = target_le.fit_transform(df["safety_label"].astype(str))
    encoders["safety_label"] = target_le
    return df, encoders


def get_feature_matrix(df: pd.DataFrame):
    feature_cols = [
        "city_enc", "state_enc", "location_description_enc",
        "crime_type_enc", "victim_gender_enc", "victim_race_enc",
        "victim_age", "year", "month", "day", "day_of_week",
        "hour", "is_weekend", "severity_score",
    ]
    X = df[feature_cols]
    y = df["safety_label_enc"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y, feature_cols, scaler


def full_pipeline(filepath: str):
    df = load_data(filepath)
    df = clean_data(df)
    df = engineer_features(df)
    df, encoders = encode_features(df)
    X, y, feature_cols, scaler = get_feature_matrix(df)
    return df, X, y, encoders, feature_cols, scaler
