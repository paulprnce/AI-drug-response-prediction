"""
run_pipeline.py — End-to-end pipeline runner.
Run this once before launching the dashboard.

Usage:
    python run_pipeline.py
    python run_pipeline.py --skip-shap   # skip SHAP (faster)
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))


def main():
    parser = argparse.ArgumentParser(description="Drug Response Prediction Pipeline")
    parser.add_argument("--skip-shap", action="store_true",
                        help="Skip SHAP analysis (saves time)")
    parser.add_argument("--n-patients", type=int, default=1200,
                        help="Number of synthetic patients to generate")
    args = parser.parse_args()

    print("=" * 55)
    print("  Drug Response Prediction — Pipeline")
    print("=" * 55)

    # Step 1: Generate data
    print("\n[1/3] Generating synthetic patient data...")
    os.makedirs("data", exist_ok=True)
    from src.data_generator import generate_patient_data
    df = generate_patient_data(args.n_patients)
    df.to_csv("data/patient_data.csv", index=False)
    print(f"    ✓ {len(df)} patients written to data/patient_data.csv")

    # Step 2: Train models
    print("\n[2/3] Training & evaluating models...")
    from src.train import train_and_evaluate
    model, results, feature_cols = train_and_evaluate(
        data_path="data/patient_data.csv",
        model_dir="models/"
    )
    best = max(results, key=lambda k: results[k]["test_auc"])
    print(f"    ✓ Best model: {best} | AUC={results[best]['test_auc']:.4f}")

    # Step 3: SHAP analysis
    if not args.skip_shap:
        print("\n[3/3] Running SHAP explainability analysis...")
        from src.explainability import run_shap_analysis
        run_shap_analysis()
        print("    ✓ SHAP plots saved to models/")
    else:
        print("\n[3/3] SHAP analysis skipped.")

    print("\n" + "=" * 55)
    print("  Pipeline complete! Launch the dashboard with:")
    print("  streamlit run dashboard/app.py")
    print("=" * 55)


if __name__ == "__main__":
    main()
