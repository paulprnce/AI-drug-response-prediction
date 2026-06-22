"""
SHAP-based explainability for drug response predictions.
Generates global feature importance and per-patient explanations.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
import joblib
import os


def load_model_artifacts(model_dir: str = "models/"):
    model = joblib.load(os.path.join(model_dir, "best_model.pkl"))
    scaler = joblib.load(os.path.join(model_dir, "scaler.pkl"))
    encoders = joblib.load(os.path.join(model_dir, "encoders.pkl"))
    feature_cols = joblib.load(os.path.join(model_dir, "feature_cols.pkl"))
    return model, scaler, encoders, feature_cols


def compute_shap_values(model, X: np.ndarray, feature_names: list,
                         sample_size: int = 200):
    """Compute SHAP values using TreeExplainer or KernelExplainer."""
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X[:sample_size])
        # For classifiers, shap_values may be list [class0, class1]
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
    except Exception:
        # Fallback to KernelExplainer for non-tree models
        background = shap.sample(X, 50)
        explainer = shap.KernelExplainer(model.predict_proba, background)
        shap_values = explainer.shap_values(X[:sample_size])
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
    return shap_values, explainer


def plot_shap_summary(shap_values: np.ndarray, X: np.ndarray,
                      feature_names: list, save_path: str = "models/shap_summary.png"):
    """Generate and save SHAP beeswarm summary plot."""
    plt.figure(figsize=(10, 7))
    shap.summary_plot(shap_values, X, feature_names=feature_names,
                      show=False, plot_size=None)
    plt.title("SHAP Feature Importance — Drug Response Prediction", fontsize=13, pad=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"SHAP summary plot saved: {save_path}")


def plot_shap_bar(shap_values: np.ndarray, feature_names: list,
                  save_path: str = "models/shap_bar.png", top_n: int = 15):
    """Generate and save SHAP mean absolute bar chart."""
    mean_abs = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs
    }).sort_values("mean_abs_shap", ascending=True).tail(top_n)

    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.barh(importance_df["feature"], importance_df["mean_abs_shap"],
                   color="#2196F3", edgecolor="none", height=0.65)
    ax.set_xlabel("Mean |SHAP Value|", fontsize=11)
    ax.set_title(f"Top {top_n} Features by SHAP Importance", fontsize=13)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"SHAP bar chart saved: {save_path}")
    return importance_df


def explain_patient(model, scaler, patient_features: np.ndarray,
                    feature_names: list, X_background: np.ndarray,
                    save_path: str = "models/shap_patient.png"):
    """Generate SHAP waterfall/force plot for a single patient."""
    try:
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(patient_features)
        if isinstance(sv, list):
            sv = sv[1]
        base_val = explainer.expected_value
        if isinstance(base_val, (list, np.ndarray)):
            base_val = base_val[1]
    except Exception:
        bg = shap.sample(X_background, 50)
        explainer = shap.KernelExplainer(model.predict_proba, bg)
        sv = explainer.shap_values(patient_features)[1]
        base_val = explainer.expected_value[1]

    shap_exp = shap.Explanation(
        values=sv[0],
        base_values=base_val,
        data=patient_features[0],
        feature_names=feature_names
    )
    plt.figure()
    shap.waterfall_plot(shap_exp, show=False)
    plt.title("Patient-Level SHAP Explanation", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Patient SHAP waterfall saved: {save_path}")

    return dict(zip(feature_names, sv[0]))


def run_shap_analysis(data_path: str = "data/patient_data.csv",
                       model_dir: str = "models/"):
    from src.preprocessing import load_data, preprocess, load_artifacts

    model, _, _, feature_cols = load_model_artifacts(model_dir)
    scaler, encoders = load_artifacts(model_dir)
    df = load_data(data_path)
    X, y, _, _, _, _ = preprocess(df, fit=False, scaler=scaler, encoders=encoders)

    print("Computing SHAP values (this may take a moment)...")
    shap_values, _ = compute_shap_values(model, X, feature_cols)

    plot_shap_summary(shap_values, X[:200], feature_cols,
                       save_path=os.path.join(model_dir, "shap_summary.png"))
    importance_df = plot_shap_bar(shap_values, feature_cols,
                                   save_path=os.path.join(model_dir, "shap_bar.png"))

    print("\nTop 10 features:")
    print(importance_df.tail(10).to_string(index=False))

    return shap_values, importance_df


if __name__ == "__main__":
    run_shap_analysis()
