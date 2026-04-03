import numpy as np
import pandas as pd
import pytest
from pandas import Series

import ml_baselines.modelling.train as train_module
from ml_baselines.modelling.train import balance_dataset, generate_sample_weights


class DummyClassifier:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.fit_X = None
        self.fit_y = None
        self.fit_sample_weight = None

    def fit(self, X, y, sample_weight=None):
        self.fit_X = X.copy()
        self.fit_y = y.copy()
        self.fit_sample_weight = None if sample_weight is None else sample_weight.copy()
        return self

    def predict_proba(self, X):
        positive_probability = np.full(len(X), 0.8)
        return np.column_stack([1.0 - positive_probability, positive_probability])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


@pytest.fixture
def patched_training_data(monkeypatch):
    def make_frame(baseline_values):
        return pd.DataFrame(
            {
                "baseline": baseline_values,
                "feature1": np.arange(len(baseline_values), dtype=float),
                "feature2": np.arange(len(baseline_values), dtype=float) + 10.0,
            },
            index=pd.date_range("2020-01-01", periods=len(baseline_values), freq="h"),
        )

    train_df = make_frame([1, 0, 1, 0])
    validation_df = make_frame([1, 0, 1, 0])
    test_df = make_frame([1, 0, 0, 1])

    def fake_get_train_test_data(site, test_train, **kwargs):
        if test_train == "train":
            df = train_df
        elif test_train == "validation":
            df = validation_df
        elif test_train == "test":
            df = test_df
        else:
            raise AssertionError(f"Unexpected split requested: {test_train}")

        return df.drop(columns=["baseline"]), df["baseline"]

    monkeypatch.setattr(train_module, "get_train_test_data", fake_get_train_test_data)
    monkeypatch.setattr(train_module, "MLPClassifier", DummyClassifier)
    monkeypatch.setattr(train_module, "RandomForestClassifier", DummyClassifier)
    monkeypatch.setattr(train_module, "GradientBoostingClassifier", DummyClassifier)


class TestTrainBaselineModel:
    def test_train_baseline_model_returns_scores_and_selected_model(self, patched_training_data):
        model, X, y, scores = train_module.train_baseline_model(
            "dummy-site",
            model_type="random_forest",
            model_params={"max_depth": 7},
            return_scores=True,
            verbose=False,
        )

        assert isinstance(model, DummyClassifier)
        assert "max_depth" in model.kwargs
        assert model.kwargs["max_depth"] == 7
        assert list(X.columns) == ["feature1", "feature2"]
        assert list(y.tolist()) == [1, 0, 1, 0]
        assert set(scores) == {
            "precision_train",
            "precision_val",
            "precision_test",
            "recall_train",
            "recall_val",
            "recall_test",
            "f1_train",
            "f1_val",
            "f1_test",
        }
        for score in scores.values():
            assert 0.0 <= score <= 1.0

    def test_train_baseline_model_passes_sample_weights(self, patched_training_data):
        model, _, y = train_module.train_baseline_model(
            "dummy-site",
            model_type="mlp",
            sample_weights=3.0,
            verbose=False,
        )

        assert isinstance(model, DummyClassifier)
        assert model.fit_sample_weight is not None
        expected_weights = pd.Series([3.0, 1.0, 3.0, 1.0], index=y.index)
        pd.testing.assert_series_equal(model.fit_sample_weight, expected_weights)

    def test_train_baseline_model_uses_custom_prediction_threshold(self, patched_training_data):
        _, _, _, scores = train_module.train_baseline_model(
            "dummy-site",
            model_type="gradient_boosting",
            prediction_threshold=0.9,
            return_scores=True,
            verbose=False,
        )

        assert scores["precision_train"] == 0.0
        assert scores["precision_val"] == 0.0
        assert scores["precision_test"] == 0.0
        assert scores["recall_train"] == 0.0
        assert scores["recall_val"] == 0.0
        assert scores["recall_test"] == 0.0
        assert scores["f1_train"] == 0.0
        assert scores["f1_val"] == 0.0
        assert scores["f1_test"] == 0.0

    def test_train_baseline_model_rejects_unknown_model_type(self, patched_training_data):
        with pytest.raises(ValueError, match="Unknown model type"):
            train_module.train_baseline_model(
                "dummy-site",
                model_type="not-a-model",
                verbose=False,
            )


def test_generate_sample_weights():
    # Test the generate_sample_weights function
    y = Series([0, 0, 1, 1, 1])
    weights = generate_sample_weights(y, baseline_weight="auto", verbose=False)
    assert np.isclose(weights[y == 1].iloc[0], 1.0 / (y == 1).mean()), "Baseline weight is not set correctly for 'auto'"
    assert np.isclose(weights[y == 0].iloc[0], 1.0), "Non-baseline weight is not set correctly for 'auto'"

    weights = generate_sample_weights(y, baseline_weight=2.0, non_baseline_weight=0.5, verbose=False)
    assert np.isclose(weights[y == 1].iloc[0], 2.0), "Baseline weight is not set correctly for specified float"
    assert np.isclose(weights[y == 0].iloc[0], 0.5), "Non-baseline weight is not set correctly for specified float"


def test_balance_dataset():
    # Test function for balance_dataset

    # Create a sample DataFrame with an imbalanced dataset. Start with too many baseline values.
    ##################

    np.random.seed(42)
    data = {
        'baseline': np.random.choice([1, 0], size=1000, p=[0.9, 0.1]),  # 90% baseline, 10% non-baseline
        'feature1': np.random.rand(1000),
        'feature2': np.random.rand(1000)
    }
    df = pd.DataFrame(data)

    df_out = balance_dataset(df, target_baseline_ratio=0.8)

    # Check if the output DataFrame has the expected ratio of baseline to non-baseline
    baseline_count = (df_out['baseline'] == 1).sum()
    non_baseline_count = (df_out['baseline'] == 0).sum()

    assert np.isclose(baseline_count / (baseline_count + non_baseline_count), 0.8, atol=0.01), "Baseline ratio is not 0.8"
    assert np.isclose(non_baseline_count / (baseline_count + non_baseline_count), 0.2, atol=0.01), "Non-baseline ratio is not 0.2"

    # Now check that it works with the deterministic method
    df_out = balance_dataset(df, target_baseline_ratio=0.8, method="deterministic")

    # Check if the output DataFrame has the expected ratio of baseline to non-baseline
    baseline_count = (df_out['baseline'] == 1).sum()
    non_baseline_count = (df_out['baseline'] == 0).sum()

    assert np.isclose(baseline_count / (baseline_count + non_baseline_count), 0.8, atol=0.01), "Baseline ratio is not 0.8"
    assert np.isclose(non_baseline_count / (baseline_count + non_baseline_count), 0.2, atol=0.01), "Non-baseline ratio is not 0.2"

    # Now start with too many non-baseline values.
    ##################
    data = {
        'baseline': np.random.choice([1, 0], size=1000, p=[0.1, 0.9]),  # 10% baseline, 90% non-baseline
        'feature1': np.random.rand(1000),
        'feature2': np.random.rand(1000)
    }
    df = pd.DataFrame(data)

    df_out = balance_dataset(df, target_baseline_ratio=0.8)

    # Check if the output DataFrame has the expected ratio of baseline to non-baseline
    baseline_count = (df_out['baseline'] == 1).sum()
    non_baseline_count = (df_out['baseline'] == 0).sum()

    assert np.isclose(baseline_count / (baseline_count + non_baseline_count), 0.8, atol=0.01), "Baseline ratio is not 0.8"
    assert np.isclose(non_baseline_count / (baseline_count + non_baseline_count), 0.2, atol=0.01), "Non-baseline ratio is not 0.2"

    # Check the deterministic version
    df_out = balance_dataset(df, target_baseline_ratio=0.8, method="deterministic")

    # Check if the output DataFrame has the expected ratio of baseline to non-baseline
    baseline_count = (df_out['baseline'] == 1).sum()
    non_baseline_count = (df_out['baseline'] == 0).sum()

    assert np.isclose(baseline_count / (baseline_count + non_baseline_count), 0.8, atol=0.01), "Baseline ratio is not 0.8"
    assert np.isclose(non_baseline_count / (baseline_count + non_baseline_count), 0.2, atol=0.01), "Non-baseline ratio is not 0.2"

