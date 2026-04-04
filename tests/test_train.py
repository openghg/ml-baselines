import numpy as np
from pandas import Series
import pandas as pd
import pytest

from ml_baselines.modelling.train import (
    InputPerVariableScaler,
    balance_dataset,
    generate_sample_weights,
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

class TestInputPerVariableScaler:
    def test_fit_transform_dataframe(self):
        # Goal: verify grouped dataframe features are standardized and columns are preserved.
        np.random.seed(0)
        n = 200
        df = pd.DataFrame(
            {
                "u10_0": np.random.normal(loc=5.0, scale=2.0, size=n),
                "u10_6": np.random.normal(loc=-2.0, scale=4.0, size=n),
                "v10_0": np.random.normal(loc=10.0, scale=3.0, size=n),
                "v10_6": np.random.normal(loc=0.0, scale=5.0, size=n),
                "hour_of_day": np.random.randint(0, 24, size=n),
                "day_of_year": np.random.randint(1, 366, size=n),
            }


        )

        scaler = InputPerVariableScaler(aux_variables=["hour_of_day", "day_of_year"])
        transformed = scaler.fit_transform(df)

        assert list(transformed.columns) == list(df.columns)

        for col in transformed.columns:
            assert np.isclose(transformed[col].mean(), 0.0, atol=1e-10), f"{col} mean not ~0"
            assert np.isclose(transformed[col].std(ddof=0), 1.0, atol=1e-10), f"{col} std not ~1"

    def test_transform_matches_fit_transform(self):
        # Goal: ensure fit+transform and fit_transform return identical outputs on same data.
        np.random.seed(1)
        n = 120
        df = pd.DataFrame(
            {
                "u850_0": np.random.normal(size=n),
                "u850_6": np.random.normal(size=n),
                "v850_0": np.random.normal(size=n),
                "v850_6": np.random.normal(size=n),
                "hour_of_day": np.random.randint(0, 24, size=n),
                "day_of_year": np.random.randint(1, 366, size=n),
            }
        )

        scaler_a = InputPerVariableScaler()
        fit_transform_out = scaler_a.fit_transform(df)

        scaler_b = InputPerVariableScaler()
        scaler_b.fit(df)
        transform_out = scaler_b.transform(df)

        assert np.allclose(fit_transform_out.values, transform_out.values)

    def test_numpy_input_with_feature_names(self):
        # Goal: verify numpy input path works when feature names are explicitly provided.
        np.random.seed(2)
        n = 100
        df = pd.DataFrame(
            {
                "u10_0": np.random.normal(size=n),
                "u10_6": np.random.normal(size=n),
                "v10_0": np.random.normal(size=n),
                "v10_6": np.random.normal(size=n),
                "hour_of_day": np.random.randint(0, 24, size=n),
                "day_of_year": np.random.randint(1, 366, size=n),
            }
        )

        feature_names = list(df.columns)
        X = df.values

        scaler = InputPerVariableScaler()
        scaler.fit(X, feature_names=feature_names)
        transformed = scaler.transform(X, feature_names=feature_names)

        assert isinstance(transformed, pd.DataFrame)
        assert list(transformed.columns) == feature_names
        assert transformed.shape == df.shape

    def test_raises_before_fit(self):
        # Goal: confirm transform fails with a clear error when fit has not been called.
        df = pd.DataFrame(
            {
                "u10_0": [1.0, 2.0, 3.0],
                "u10_6": [2.0, 3.0, 4.0],
                "hour_of_day": [0, 6, 12],
                "day_of_year": [10, 11, 12],
            }
        )

        scaler = InputPerVariableScaler()
        with pytest.raises(ValueError, match="has not been fitted"):
            scaler.transform(df)

    def test_numpy_requires_feature_names(self):
        # Goal: ensure numpy input without feature names raises informative errors.
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        scaler = InputPerVariableScaler()

        with pytest.raises(ValueError, match="feature_names must be provided"):
            scaler.fit(X)

        with pytest.raises(ValueError, match="feature_names must be provided"):
            scaler.transform(X)


## TODO add tests checking it works within the pipeline after file merging with main branch