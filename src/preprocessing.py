"""
Feature engineering and preprocessing for drug response prediction.
Handles encoding, scaling, and feature construction.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
import joblib
import os

CLINICAL_FEATURES = [
    "age", "weight_kg", "bmi", "creatinine", "egfr",
    "alt_liver_enzyme", "albumin", "hemoglobin"
]
GENETIC_FEATURES = ["BRCA1_mut", "CYP2D6_pm", "EGFR_mut", "KRAS_mut", "TP53_mut"]
CATEGORICAL_FEATURES = ["gender", "drug_assigned"]


def load_data(path: str = "data/patient_data.csv") -> pd.DataFrame:
    return pd.read_csv(path)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Interaction: genetic burden score
    df["genetic_burden"] = df[GENETIC_FEATURES].sum(axis=1)

    # Renal function category
    df["renal_function"] = pd.cut(
        df["egfr"],
        bins=[0, 30, 60, 90, float("inf")],
        labels=["Severe", "Moderate", "Mild", "Normal"]
    ).astype(str)

    # Age group
    df["age_group"] = pd.cut(
        df["age"],
        bins=[0, 40, 60, 75, float("inf")],
        labels=["Young", "Middle", "Senior", "Elderly"]
    ).astype(str)

    # BMI category
    df["bmi_category"] = pd.cut(
        df["bmi"],
        bins=[0, 18.5, 25, 30, float("inf")],
        labels=["Underweight", "Normal", "Overweight", "Obese"]
    ).astype(str)

    return df


def preprocess(df: pd.DataFrame, fit: bool = True, scaler=None, encoders=None):
    """
    Preprocess data for ML.
    Returns X (features), y_class (label), y_score (regression target).
    """
    df = engineer_features(df)

    # Encode categoricals
    cat_cols = CATEGORICAL_FEATURES + ["renal_function", "age_group", "bmi_category"]
    if encoders is None:
        encoders = {}

    for col in cat_cols:
        le = LabelEncoder()
        if fit:
            df[col + "_enc"] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
        else:
            df[col + "_enc"] = encoders[col].transform(df[col].astype(str))

    encoded_cats = [c + "_enc" for c in cat_cols]
    feature_cols = CLINICAL_FEATURES + GENETIC_FEATURES + ["genetic_burden"] + encoded_cats

    X = df[feature_cols].values
    y_class = (df["response_label"] == "Responder").astype(int).values
    y_score = df["response_score"].values

    if scaler is None:
        scaler = StandardScaler()

    if fit:
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)

    return X_scaled, y_class, y_score, scaler, encoders, feature_cols


def save_artifacts(scaler, encoders, path="models/"):
    os.makedirs(path, exist_ok=True)
    joblib.dump(scaler, os.path.join(path, "scaler.pkl"))
    joblib.dump(encoders, os.path.join(path, "encoders.pkl"))
    print("Preprocessing artifacts saved.")


def load_artifacts(path="models/"):
    scaler = joblib.load(os.path.join(path, "scaler.pkl"))
    encoders = joblib.load(os.path.join(path, "encoders.pkl"))
    return scaler, encoders
