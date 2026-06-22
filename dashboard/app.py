"""
Streamlit Dashboard — Drug Response Prediction
Compare drug effectiveness across patient profiles,
view SHAP explanations, and simulate treatment decisions.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import os
import json
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Drug Response Prediction",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_DIR = "models/"
DATA_PATH = "data/patient_data.csv"

# ── Load artifacts ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model     = joblib.load(os.path.join(MODEL_DIR, "best_model.pkl"))
    scaler    = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    encoders  = joblib.load(os.path.join(MODEL_DIR, "encoders.pkl"))
    feat_cols = joblib.load(os.path.join(MODEL_DIR, "feature_cols.pkl"))
    with open(os.path.join(MODEL_DIR, "results.json")) as f:
        results = json.load(f)
    return model, scaler, encoders, feat_cols, results

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


def preprocess_patient(patient_dict, scaler, encoders):
    from src.preprocessing import engineer_features, CLINICAL_FEATURES, GENETIC_FEATURES

    df_p = pd.DataFrame([patient_dict])
    df_p = engineer_features(df_p)

    cat_cols = ["gender", "drug_assigned", "renal_function", "age_group", "bmi_category"]
    for col in cat_cols:
        le = encoders[col]
        val = str(df_p[col].iloc[0])
        if val not in le.classes_:
            df_p[col + "_enc"] = 0
        else:
            df_p[col + "_enc"] = le.transform([val])[0]

    encoded_cats = [c + "_enc" for c in cat_cols]
    feature_cols_order = (
        CLINICAL_FEATURES
        + ["BRCA1_mut", "CYP2D6_pm", "EGFR_mut", "KRAS_mut", "TP53_mut"]
        + ["genetic_burden"]
        + encoded_cats
    )
    X = df_p[feature_cols_order].values
    X_scaled = scaler.transform(X)
    return X_scaled


# ── Sidebar — Patient Profile ───────────────────────────────────────────────────
st.sidebar.header("🧬 Patient Profile")
age     = st.sidebar.slider("Age", 18, 90, 55)
gender  = st.sidebar.selectbox("Gender", ["Male", "Female"])
weight  = st.sidebar.slider("Weight (kg)", 40, 130, 72)
height  = st.sidebar.slider("Height (cm)", 150, 200, 170)
bmi     = weight / ((height / 100) ** 2)
st.sidebar.metric("BMI", f"{bmi:.1f}")

st.sidebar.subheader("Clinical Markers")
creatinine = st.sidebar.slider("Creatinine (mg/dL)", 0.4, 4.0, 1.0, 0.1)
egfr       = st.sidebar.slider("eGFR (mL/min)", 15, 120, 80)
alt        = st.sidebar.slider("ALT Liver Enzyme (U/L)", 10, 200, 40)
albumin    = st.sidebar.slider("Albumin (g/dL)", 2.0, 5.5, 4.0, 0.1)
hemoglobin = st.sidebar.slider("Hemoglobin (g/dL)", 7.0, 18.0, 13.5, 0.5)

st.sidebar.subheader("Genetic Markers")
gene_cols = ["BRCA1_mut", "CYP2D6_pm", "EGFR_mut", "KRAS_mut", "TP53_mut"]
genes = {g: int(st.sidebar.checkbox(g.replace("_", " "), False)) for g in gene_cols}

# ── Main UI ─────────────────────────────────────────────────────────────────────
st.title("💊 AI Drug Response Prediction Dashboard")
st.caption("Personalized treatment decision support using ML + SHAP explainability")

try:
    model, scaler, encoders, feat_cols, results = load_artifacts()
    df_all = load_data()
    artifacts_ready = True
except Exception as e:
    st.warning(f"⚠️ Models not trained yet. Run `python run_pipeline.py` first.\n\n`{e}`")
    artifacts_ready = False

if artifacts_ready:
    # ── Tab layout ──────────────────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs(["🔮 Drug Comparison", "📊 Model Performance", "📁 Dataset Explorer"])

    # ── Tab 1: Drug Comparison ──────────────────────────────────────────────────
    with tab1:
        st.subheader("Drug Effectiveness Prediction for This Patient")

        drugs  = ["DrugA", "DrugB", "DrugC", "DrugD"]
        probs  = []
        labels = []

        for drug in drugs:
            patient = {
                "age": age, "gender": gender, "weight_kg": weight,
                "bmi": bmi, "creatinine": creatinine, "egfr": egfr,
                "alt_liver_enzyme": alt, "albumin": albumin,
                "hemoglobin": hemoglobin, "drug_assigned": drug,
                **genes,
            }
            X_p = preprocess_patient(patient, scaler, encoders)
            prob = model.predict_proba(X_p)[0][1]
            probs.append(prob)
            labels.append("✅ Responder" if prob >= 0.5 else "❌ Non-Responder")

        # Metric cards
        cols = st.columns(4)
        best_drug_idx = int(np.argmax(probs))
        for i, (drug, prob, label) in enumerate(zip(drugs, probs, labels)):
            with cols[i]:
                border = "2px solid #4CAF50" if i == best_drug_idx else "1px solid #ddd"
                st.markdown(
                    f"""<div style='border:{border};border-radius:10px;padding:16px;text-align:center'>
                    <h3 style='margin:0'>{drug}</h3>
                    <h1 style='color:{"#4CAF50" if prob>=0.5 else "#f44336"};margin:4px 0'>
                        {prob*100:.0f}%</h1>
                    <p style='margin:0;font-size:13px'>{label}</p>
                    {"<p style='color:#4CAF50;font-weight:bold;font-size:12px'>⭐ RECOMMENDED</p>" if i==best_drug_idx else ""}
                    </div>""", unsafe_allow_html=True
                )

        st.markdown("---")
        # Bar chart
        fig, ax = plt.subplots(figsize=(8, 3.5))
        colors = ["#4CAF50" if i == best_drug_idx else "#90CAF9" for i in range(4)]
        bars = ax.bar(drugs, [p * 100 for p in probs], color=colors, edgecolor="none", width=0.55)
        ax.axhline(50, color="gray", linestyle="--", linewidth=1, alpha=0.7, label="50% threshold")
        ax.set_ylabel("Response Probability (%)")
        ax.set_title("Predicted Response Probability by Drug", fontsize=13)
        ax.set_ylim(0, 100)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        for bar, p in zip(bars, probs):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                    f"{p*100:.1f}%", ha="center", va="bottom", fontsize=10)
        ax.legend(fontsize=9)
        st.pyplot(fig, use_container_width=True)

        # SHAP plot if available
        shap_path = os.path.join(MODEL_DIR, "shap_bar.png")
        if os.path.exists(shap_path):
            st.markdown("### 🔍 Key Factors Driving Predictions (SHAP)")
            st.image(shap_path, caption="Top features by mean |SHAP value|", use_column_width=True)

    # ── Tab 2: Model Performance ────────────────────────────────────────────────
    with tab2:
        st.subheader("Model Comparison & Performance Metrics")

        best_name   = results["best_model"]
        all_results = results["all_models"]

        perf_df = pd.DataFrame(all_results).T.reset_index()
        perf_df.columns = ["Model", "CV AUC Mean", "CV AUC Std", "Test AUC",
                           "Test Accuracy", "Test F1"]
        perf_df = perf_df.sort_values("Test AUC", ascending=False)

        def highlight_best(row):
            return ["background-color: #e8f5e9" if row["Model"] == best_name
                    else "" for _ in row]

        st.dataframe(
            perf_df.style.apply(highlight_best, axis=1).format({
                "CV AUC Mean": "{:.4f}", "CV AUC Std": "{:.4f}",
                "Test AUC": "{:.4f}", "Test Accuracy": "{:.4f}", "Test F1": "{:.4f}"
            }),
            use_container_width=True
        )
        st.success(f"✓ Best model in production: **{best_name}**")

        # AUC comparison bar
        fig2, ax2 = plt.subplots(figsize=(8, 3.5))
        model_names = perf_df["Model"].tolist()
        aucs = perf_df["Test AUC"].tolist()
        bar_colors = ["#4CAF50" if m == best_name else "#90CAF9" for m in model_names]
        ax2.barh(model_names, aucs, color=bar_colors, edgecolor="none", height=0.55)
        ax2.axvline(0.5, color="gray", linestyle="--", linewidth=1, alpha=0.6)
        ax2.set_xlabel("Test AUC")
        ax2.set_title("Model AUC Comparison", fontsize=12)
        ax2.set_xlim(0.4, 1.0)
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)
        st.pyplot(fig2, use_container_width=True)

        shap_sum_path = os.path.join(MODEL_DIR, "shap_summary.png")
        if os.path.exists(shap_sum_path):
            st.markdown("### SHAP Summary (Beeswarm)")
            st.image(shap_sum_path, use_column_width=True)

    # ── Tab 3: Dataset Explorer ──────────────────────────────────────────────────
    with tab3:
        st.subheader("Patient Dataset Overview")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Patients", len(df_all))
        col2.metric("Responders", (df_all["response_label"] == "Responder").sum())
        col3.metric("Non-Responders", (df_all["response_label"] == "Non-Responder").sum())
        col4.metric("Drugs Studied", df_all["drug_assigned"].nunique())

        st.dataframe(df_all.head(50), use_container_width=True, height=300)

        fig3, axes = plt.subplots(1, 3, figsize=(14, 4))
        df_all.groupby(["drug_assigned", "response_label"]).size().unstack().plot(
            kind="bar", ax=axes[0], color=["#f44336", "#4CAF50"], edgecolor="none")
        axes[0].set_title("Response by Drug")
        axes[0].tick_params(axis="x", rotation=30)
        axes[0].spines["top"].set_visible(False)
        axes[0].spines["right"].set_visible(False)

        df_all["age"].hist(ax=axes[1], bins=25, color="#2196F3", edgecolor="none")
        axes[1].set_title("Age Distribution")
        axes[1].spines["top"].set_visible(False)
        axes[1].spines["right"].set_visible(False)

        df_all.groupby("drug_assigned")["response_score"].mean().plot(
            kind="bar", ax=axes[2], color="#7986CB", edgecolor="none")
        axes[2].set_title("Mean Response Score by Drug")
        axes[2].tick_params(axis="x", rotation=30)
        axes[2].spines["top"].set_visible(False)
        axes[2].spines["right"].set_visible(False)

        plt.tight_layout()
        st.pyplot(fig3, use_container_width=True)
