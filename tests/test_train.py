import numpy as np
from pandas import Series

from ml_baselines.modelling.train import balance_dataset, generate_sample_weights


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
    import pandas as pd
    
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

