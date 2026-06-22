"""
Unit tests for drug response prediction pipeline.
Run with: pytest tests/ -v
"""

import os
import sys
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestDataGenerator:
    def test_generates_correct_count(self):
        from src.data_generator import generate_patient_data
        df = generate_patient_data(100)
        assert len(df) == 100

    def test_required_columns_present(self):
        from src.data_generator import generate_patient_data
        df = generate_patient_data(50)
        required = [
            "patient_id", "age", "gender", "bmi", "egfr",
            "drug_assigned", "response_score", "response_label",
            "BRCA1_mut", "CYP2D6_pm", "EGFR_mut", "KRAS_mut", "TP53_mut"
        ]
        for col in required:
            assert col in df.columns, f"Missing column: {col}"

    def test_response_score_range(self):
        from src.data_generator import generate_patient_data
        df = generate_patient_data(200)
        assert df["response_score"].between(0, 100).all()

    def test_response_label_valid(self):
        from src.data_generator import generate_patient_data
        df = generate_patient_data(100)
        assert set(df["response_label"].unique()).issubset({"Responder", "Non-Responder"})

    def test_genetic_markers_binary(self):
        from src.data_generator import generate_patient_data, GENE_VARIANTS
        df = generate_patient_data(100)
        for g in GENE_VARIANTS:
            assert df[g].isin([0, 1]).all(), f"{g} not binary"

    def test_drug_column_valid(self):
        from src.data_generator import generate_patient_data, DRUGS
        df = generate_patient_data(100)
        assert df["drug_assigned"].isin(DRUGS).all()


class TestPreprocessing:
    @pytest.fixture
    def sample_df(self):
        from src.data_generator import generate_patient_data
        return generate_patient_data(200)

    def test_preprocess_returns_correct_shape(self, sample_df):
        from src.preprocessing import preprocess
        X, y, y_score, scaler, encoders, feature_cols = preprocess(sample_df, fit=True)
        assert X.shape[0] == len(sample_df)
        assert X.shape[1] == len(feature_cols)
        assert y.shape[0] == len(sample_df)

    def test_labels_binary(self, sample_df):
        from src.preprocessing import preprocess
        _, y, _, _, _, _ = preprocess(sample_df, fit=True)
        assert set(np.unique(y)).issubset({0, 1})

    def test_feature_engineering_adds_columns(self, sample_df):
        from src.preprocessing import engineer_features
        df_eng = engineer_features(sample_df)
        assert "genetic_burden" in df_eng.columns
        assert "renal_function" in df_eng.columns
        assert "age_group" in df_eng.columns
        assert "bmi_category" in df_eng.columns

    def test_genetic_burden_correct(self, sample_df):
        from src.preprocessing import engineer_features, GENETIC_FEATURES
        df_eng = engineer_features(sample_df)
        expected = sample_df[GENETIC_FEATURES].sum(axis=1)
        pd.testing.assert_series_equal(
            df_eng["genetic_burden"].reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False
        )

    def test_scaled_data_has_no_nan(self, sample_df):
        from src.preprocessing import preprocess
        X, _, _, _, _, _ = preprocess(sample_df, fit=True)
        assert not np.isnan(X).any()

    def test_refit_false_uses_existing_scaler(self, sample_df):
        from src.preprocessing import preprocess
        X1, _, _, scaler, encoders, _ = preprocess(sample_df, fit=True)
        X2, _, _, _, _, _ = preprocess(sample_df, fit=False, scaler=scaler, encoders=encoders)
        np.testing.assert_array_almost_equal(X1, X2)


class TestTrainPipeline:
    """Integration tests — run model train on small dataset."""

    @pytest.fixture(scope="class")
    def trained_artifacts(self, tmp_path_factory):
        from src.data_generator import generate_patient_data
        from src.train import train_and_evaluate

        tmp = tmp_path_factory.mktemp("pipeline")
        data_path = str(tmp / "patients.csv")
        model_dir = str(tmp / "models")

        df = generate_patient_data(300)
        df.to_csv(data_path, index=False)
        model, results, feat_cols = train_and_evaluate(data_path, model_dir)
        return model, results, feat_cols, model_dir

    def test_model_returns(self, trained_artifacts):
        model, results, feat_cols, _ = trained_artifacts
        assert model is not None
        assert isinstance(results, dict)
        assert len(feat_cols) > 0

    def test_results_have_expected_models(self, trained_artifacts):
        _, results, _, _ = trained_artifacts
        for name in ["Logistic Regression (Baseline)", "Random Forest"]:
            assert name in results

    def test_model_auc_above_baseline(self, trained_artifacts):
        _, results, _, _ = trained_artifacts
        for name, metrics in results.items():
            assert metrics["test_auc"] >= 0.45, f"{name} AUC too low: {metrics['test_auc']}"

    def test_model_files_saved(self, trained_artifacts):
        _, _, _, model_dir = trained_artifacts
        assert os.path.exists(os.path.join(model_dir, "best_model.pkl"))
        assert os.path.exists(os.path.join(model_dir, "scaler.pkl"))
        assert os.path.exists(os.path.join(model_dir, "results.json"))

    def test_prediction_output_shape(self, trained_artifacts):
        from src.data_generator import generate_patient_data
        from src.preprocessing import preprocess, load_artifacts
        model, _, _, model_dir = trained_artifacts

        df_test = generate_patient_data(10)
        scaler, encoders = load_artifacts(model_dir)
        X, _, _, _, _, _ = preprocess(df_test, fit=False, scaler=scaler, encoders=encoders)
        preds = model.predict(X)
        probs = model.predict_proba(X)

        assert preds.shape == (10,)
        assert probs.shape == (10, 2)
        assert (probs.sum(axis=1) - 1.0 < 1e-5).all()
