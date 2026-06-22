"""
Synthetic clinical & genetic data generator for drug response prediction.
Simulates realistic patient profiles for model training/testing.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

np.random.seed(42)

DRUGS = ["DrugA", "DrugB", "DrugC", "DrugD"]
GENE_VARIANTS = ["BRCA1_mut", "CYP2D6_pm", "EGFR_mut", "KRAS_mut", "TP53_mut"]


def generate_patient_data(n_patients: int = 1000) -> pd.DataFrame:
    """Generate synthetic patient clinical + genetic profiles."""
    data = []

    for pid in range(n_patients):
        age = int(np.random.normal(55, 15))
        age = max(18, min(90, age))

        gender = np.random.choice(["Male", "Female"])
        weight = np.random.normal(72 if gender == "Male" else 63, 12)
        bmi = weight / ((np.random.normal(1.70, 0.08)) ** 2)

        # Clinical markers
        creatinine = np.random.lognormal(0.1, 0.3)
        egfr = max(15, 140 - age * 0.8 + np.random.normal(0, 10))
        liver_enzyme_alt = np.random.lognormal(3.5, 0.5)
        albumin = np.random.normal(4.0, 0.5)
        hemoglobin = np.random.normal(13.5 if gender == "Male" else 12.0, 1.5)

        # Genetic markers (binary presence)
        genes = {g: int(np.random.random() < 0.2) for g in GENE_VARIANTS}

        # Drug assigned
        drug = np.random.choice(DRUGS)

        # Simulate response score (0-100, higher = better response)
        base_response = np.random.normal(50, 20)

        # Gene-drug interactions
        if drug == "DrugA" and genes["CYP2D6_pm"]:
            base_response -= 20
        if drug == "DrugB" and genes["EGFR_mut"]:
            base_response += 25
        if drug == "DrugC" and genes["BRCA1_mut"]:
            base_response += 30
        if drug == "DrugD" and genes["KRAS_mut"]:
            base_response -= 15

        # Clinical interactions
        if egfr < 30:
            base_response -= 10
        if bmi > 30:
            base_response -= 5
        if age > 70:
            base_response -= 8

        response_score = float(np.clip(base_response, 0, 100))
        response_label = "Responder" if response_score >= 50 else "Non-Responder"

        row = {
            "patient_id": f"P{pid:04d}",
            "age": age,
            "gender": gender,
            "weight_kg": round(weight, 1),
            "bmi": round(bmi, 2),
            "creatinine": round(creatinine, 2),
            "egfr": round(egfr, 1),
            "alt_liver_enzyme": round(liver_enzyme_alt, 1),
            "albumin": round(albumin, 2),
            "hemoglobin": round(hemoglobin, 1),
            **genes,
            "drug_assigned": drug,
            "response_score": round(response_score, 2),
            "response_label": response_label,
        }
        data.append(row)

    df = pd.DataFrame(data)
    return df


if __name__ == "__main__":
    df = generate_patient_data(1000)
    df.to_csv("data/patient_data.csv", index=False)
    print(f"Generated {len(df)} patient records.")
    print(df.head())
    print("\nResponse distribution:")
    print(df["response_label"].value_counts())
