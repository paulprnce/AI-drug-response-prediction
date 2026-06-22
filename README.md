# 💊 AI System for Drug Response Prediction

> Personalized treatment decision support using machine learning, genetic data, and SHAP explainability.

---

## Overview

This project builds an end-to-end ML pipeline that predicts how individual patients respond to different drugs based on their **clinical** and **genetic** profiles. It includes:

- **Synthetic patient data generator** simulating real gene-drug interactions
- **Multiple ML models** with cross-validation and automated best-model selection
- **SHAP explainability** to identify which clinical/genetic factors drive predictions
- **Interactive Streamlit dashboard** for comparing drug effectiveness per patient profile
- **Personalized treatment simulation** — rank all drugs for any new patient

---

## Project Structure

```
drug-response-prediction/
├── data/                        # Generated patient data (CSV)
├── models/                      # Trained model + SHAP plots (auto-created)
├── notebooks/
│   └── exploratory_analysis.ipynb
├── src/
│   ├── data_generator.py        # Synthetic clinical + genetic data
│   ├── preprocessing.py         # Feature engineering & scaling
│   ├── train.py                 # Multi-model training & evaluation
│   └── explainability.py        # SHAP analysis & plots
├── dashboard/
│   └── app.py                   # Streamlit interactive dashboard
├── tests/
│   └── test_pipeline.py         # Unit & integration tests
├── run_pipeline.py              # One-command pipeline runner
└── requirements.txt
```

---

## Quickstart

### 1. Clone & install
```bash
git clone https://github.com/YOUR_USERNAME/drug-response-prediction.git
cd drug-response-prediction
pip install -r requirements.txt
```

### 2. Run the full pipeline
```bash
python run_pipeline.py
```
This will:
- Generate 1,200 synthetic patient records
- Train and compare 4 ML models (Logistic Regression, Random Forest, Gradient Boosting, SVM)
- Select and save the best model by AUC
- Run SHAP analysis and save feature importance plots

### 3. Launch the dashboard
```bash
streamlit run dashboard/app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Models Compared

| Model | Description |
|-------|-------------|
| Logistic Regression | Baseline linear model |
| Random Forest | Ensemble of decision trees |
| **Gradient Boosting** | Typically best performer |
| SVM | Kernel-based classifier |

Best model selected automatically by test AUC and retrained on full data.

---

## Features Used

**Clinical:** Age, BMI, eGFR, Creatinine, ALT, Albumin, Hemoglobin, Weight

**Genetic:** BRCA1, CYP2D6 (Poor Metabolizer), EGFR, KRAS, TP53 mutations

**Engineered:** Genetic burden score, Renal function category, Age group, BMI category

---

## Gene–Drug Interaction Logic

| Gene Variant | Drug | Effect |
|-------------|------|--------|
| CYP2D6 Poor Metabolizer | DrugA | −20 response score |
| EGFR mutation | DrugB | +25 response score |
| BRCA1 mutation | DrugC | +30 response score |
| KRAS mutation | DrugD | −15 response score |

Clinical penalties applied for low eGFR (<30), high BMI (>30), and age >70.

---

## Dashboard Features

- **Drug Comparison Tab:** Input any patient profile → see response probability for all 4 drugs
- **Model Performance Tab:** Compare all models by AUC, Accuracy, F1; view SHAP plots
- **Dataset Explorer Tab:** Browse the patient dataset with distribution charts

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Tech Stack

- **Python 3.10+**
- **scikit-learn** — model training & evaluation
- **SHAP** — model explainability
- **Streamlit** — interactive dashboard
- **pandas / numpy / matplotlib / seaborn**

---

## Results

Typical performance on 1,200 synthetic patients (80/20 split):

| Model | CV AUC | Test AUC | Test Acc |
|-------|--------|----------|----------|
| Logistic Regression | ~0.72 | ~0.73 | ~0.67 |
| Random Forest | ~0.84 | ~0.85 | ~0.77 |
| **Gradient Boosting** | **~0.86** | **~0.87** | **~0.79** |
| SVM | ~0.80 | ~0.81 | ~0.73 |

---

## Author

**Bigil** — B.Tech Biotechnology & Chemical Engineering, KTU  
Internship experience in Data Science & AI | Open to opportunities in Gulf & Germany

---

## License

MIT License
