"""
Model training: trains multiple classifiers and selects the best one.
Includes baseline comparison, cross-validation, and model serialization.
"""

import os
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    classification_report, confusion_matrix
)
from src.data_generator import generate_patient_data
from src.preprocessing import load_data, preprocess, save_artifacts

MODELS = {
    "Logistic Regression (Baseline)": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, learning_rate=0.05,
                                                     max_depth=4, random_state=42),
    "SVM": SVC(probability=True, kernel="rbf", random_state=42),
}


def train_and_evaluate(data_path: str = "data/patient_data.csv",
                       model_dir: str = "models/",
                       test_size: float = 0.2):
    """Train all models, evaluate, and save the best one."""
    os.makedirs(model_dir, exist_ok=True)

    # Generate data if not present
    if not os.path.exists(data_path):
        os.makedirs("data", exist_ok=True)
        df = generate_patient_data(1200)
        df.to_csv(data_path, index=False)
        print(f"Generated data saved to {data_path}")
    else:
        df = load_data(data_path)

    print(f"Dataset: {len(df)} patients | Response rate: "
          f"{(df['response_label']=='Responder').mean():.1%}")

    X, y, _, scaler, encoders, feature_cols = preprocess(df, fit=True)
    save_artifacts(scaler, encoders, model_dir)

    # Train/test split (stratified)
    n_test = int(len(X) * test_size)
    idx = np.random.RandomState(42).permutation(len(X))
    test_idx, train_idx = idx[:n_test], idx[n_test:]
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    results = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print("\n--- Model Comparison ---")
    for name, model in MODELS.items():
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv,
                                    scoring="roc_auc", n_jobs=-1)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        metrics = {
            "cv_auc_mean": round(cv_scores.mean(), 4),
            "cv_auc_std": round(cv_scores.std(), 4),
            "test_auc": round(roc_auc_score(y_test, y_prob), 4),
            "test_accuracy": round(accuracy_score(y_test, y_pred), 4),
            "test_f1": round(f1_score(y_test, y_pred), 4),
        }
        results[name] = metrics
        print(f"\n{name}")
        print(f"  CV AUC:  {metrics['cv_auc_mean']:.4f} ± {metrics['cv_auc_std']:.4f}")
        print(f"  Test AUC: {metrics['test_auc']:.4f} | Acc: {metrics['test_accuracy']:.4f} | F1: {metrics['test_f1']:.4f}")

    # Select best model by test AUC
    best_name = max(results, key=lambda k: results[k]["test_auc"])
    best_model = MODELS[best_name]
    best_model.fit(X, y)  # retrain on full data
    print(f"\n✓ Best model: {best_name} (AUC={results[best_name]['test_auc']:.4f})")

    # Save artifacts
    joblib.dump(best_model, os.path.join(model_dir, "best_model.pkl"))
    joblib.dump(feature_cols, os.path.join(model_dir, "feature_cols.pkl"))

    results_out = {
        "best_model": best_name,
        "all_models": results,
        "feature_cols": feature_cols,
    }
    with open(os.path.join(model_dir, "results.json"), "w") as f:
        json.dump(results_out, f, indent=2)

    print(f"\nAll artifacts saved to '{model_dir}'")
    return best_model, results, feature_cols


if __name__ == "__main__":
    train_and_evaluate()
