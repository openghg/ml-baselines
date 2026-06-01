import numpy as np
import pandas as pd
import pytest
from pandas import Series

from ml_baselines.modelling.train import (
    InputPerVariableScaler,
    balance_dataset,
    generate_sample_weights,
)
import ml_baselines.modelling.train as train_module


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
        positive_probability = 0.8 if self.fit_sample_weight is not None else 0.4
        pos = np.full(len(X), positive_probability)
        return np.column_stack([1.0 - pos, pos])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class DummyGridSearchCV:
    instances = []

    def __init__(self, estimator, param_grid, scoring=None, cv=None, refit=True, verbose=0, n_jobs=None, return_train_score=False):
        self.estimator = estimator
        self.param_grid = param_grid
        self.scoring = scoring
        self.cv = cv
        self.refit = refit
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.return_train_score = return_train_score

        self.fit_X = None
        self.fit_y = None
        self.fit_sample_weight = None
        self.best_params_ = None
        self.best_score_ = None
        self.cv_results_ = None

        DummyGridSearchCV.instances.append(self)

    def fit(self, X, y, sample_weight=None):
        self.fit_X = X.copy()
        self.fit_y = y.copy()
        self.fit_sample_weight = None if sample_weight is None else np.array(sample_weight, copy=True)

        self.best_params_ = {
            key: (value[0] if isinstance(value, list) else value)
            for key, value in self.param_grid.items()
        }
        self.best_score_ = 0.0 if sample_weight is None else float(np.mean(sample_weight))
        self.cv_results_ = {
            "mean_test_score": np.array([self.best_score_]),
            "rank_test_score": np.array([1]),
            "params": [self.best_params_],
        }
        fitted = DummyClassifier(**self.estimator.kwargs)
        fitted.fit(X, y, sample_weight=sample_weight)
        self.best_estimator_ = fitted

        return self


@pytest.fixture
def patched_training_data(monkeypatch):
    def make_frame(baseline_values):
        return pd.DataFrame(
            {
                "baseline": baseline_values,
                "u10_0": np.arange(len(baseline_values), dtype=float),
                "u10_6": np.arange(len(baseline_values), dtype=float) + 1.0,
                "v10_0": np.arange(len(baseline_values), dtype=float) + 10.0,
                "v10_6": np.arange(len(baseline_values), dtype=float) + 11.0,
                "hour_of_day": np.random.randint(0, 24, size=len(baseline_values)),
                "day_of_year": np.random.randint(1, 366, size=len(baseline_values)),
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


@pytest.fixture
def patched_grid_search_setup(monkeypatch):
    DummyGridSearchCV.instances = []
    get_data_calls = []

    train_df = pd.DataFrame(
        {
            "baseline": [1, 0, 1, 0],
            "u10_0": [1.0, 2.0, 3.0, 4.0],
            "v10_0": [10.0, 11.0, 12.0, 13.0],
        },
        index=pd.date_range("2020-01-01", periods=4, freq="h"),
    )
    val_df = pd.DataFrame(
        {
            "baseline": [1, 0],
            "u10_0": [5.0, 6.0],
            "v10_0": [14.0, 15.0],
        },
        index=pd.date_range("2020-01-05", periods=2, freq="h"),
    )

    def fake_get_train_test_data(site, test_train, **kwargs):
        get_data_calls.append((test_train, kwargs.copy()))
        if test_train == "train":
            df = train_df
        elif test_train == "validation":
            df = val_df
        else:
            raise AssertionError(f"Unexpected split requested in grid-search test: {test_train}")
        return df.drop(columns=["baseline"]), df["baseline"]

    monkeypatch.setattr(train_module, "get_train_test_data", fake_get_train_test_data)
    monkeypatch.setattr(train_module, "GridSearchCV", DummyGridSearchCV)
    monkeypatch.setattr(train_module, "MLPClassifier", DummyClassifier)

    return {
        "calls": get_data_calls,
        "train_index": train_df.index,
        "val_index": val_df.index,
    }


class TestTrainBaselineModel:
    def test_train_baseline_model_returns_scores_and_selected_model(self, patched_training_data):
        model, X, y, extra_info = train_module.train_baseline_model(
            "dummy-site",
            model_type="random_forest",
            model_params={"max_depth": 7},
            return_scores=True,
            verbose=False,
        )

        assert isinstance(model, DummyClassifier)
        assert "max_depth" in model.kwargs
        assert model.kwargs["max_depth"] == 7
        assert list(X.columns) == ["u10_0", "u10_6", "v10_0", "v10_6", "hour_of_day", "day_of_year"]
        assert list(y.tolist()) == [1, 0, 1, 0]
        assert "scores" in extra_info
        scores = extra_info["scores"]
        assert np.all(x in scores for x in ["precision_train", "precision_val", "precision_test", "recall_train", "recall_val", "recall_test", "f1_train", "f1_val", "f1_test"]), "Missing expected score keys"
        for score in scores.values():
            assert 0.0 <= score <= 1.0

    def test_train_baseline_model_passes_sample_weights(self, patched_training_data):
        model, _, y, _ = train_module.train_baseline_model(
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
        _, _, _, extra_info = train_module.train_baseline_model(
            "dummy-site",
            model_type="gradient_boosting",
            prediction_threshold=0.9,
            return_scores=True,
            verbose=False,
        )
        scores = extra_info["scores"]
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

    def test_train_baseline_model_trains_and_returns_scaler(self, patched_training_data):
        # Goal: verify scaler is trained, inputs are normalized, and scaler is returned in extra_info.
        model, X_train, y_train, extra_info = train_module.train_baseline_model(
            "dummy-site",
            model_type="mlp",
            normalise_inputs=True,
            return_scaler=True,
            verbose=False,
        )

        # Verify scaler is in extra_info
        assert "scaler" in extra_info
        scaler = extra_info["scaler"]
        assert isinstance(scaler, InputPerVariableScaler)

        # Verify X_train is normalized: each feature should have mean ~0 and std ~1
        for col in X_train.columns:
            assert np.isclose(X_train[col].mean(), 0.0, atol=1e-10), f"{col} mean not ~0"
            assert np.isclose(X_train[col].std(ddof=0), 1.0, atol=1e-10), f"{col} std not ~1"


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

def test_get_train_test_data_removes_overlap_from_test_split(monkeypatch):
    site = "dummy-site"

    monkeypatch.setattr(train_module.cfg, "training_period", {site: (2019, 2019)}, raising=False)
    monkeypatch.setattr(train_module.cfg, "validation_period", {site: (2020, 2020)}, raising=False)
    monkeypatch.setattr(train_module.cfg, "testing_period", {site: (2019, 2021)}, raising=False)
    monkeypatch.setattr(train_module.cfg, "full_period", {site: None}, raising=False)

    idx = pd.to_datetime(
        [
            "2019-01-01 00:00", "2019-06-01 00:00",
            "2020-01-01 00:00", "2020-06-01 00:00",
            "2021-01-01 00:00", "2021-06-01 00:00",
        ]
    )

    df_features = pd.DataFrame(
        {
            "u10_0": np.arange(len(idx), dtype=float),
            "v10_0": np.arange(len(idx), dtype=float) + 10.0,
        },
        index=idx,
    )
    df_intem = pd.DataFrame({"baseline": [0, 1, 0, 1, 0, 1]}, index=idx)

    monkeypatch.setattr(train_module, "open_features", lambda *args, **kwargs: df_features)
    monkeypatch.setattr(train_module, "read_intem", lambda *args, **kwargs: df_intem)

    X_test, y_test = train_module.get_train_test_data(site, "test", verbose=False)

    assert set(X_test.index.year) == {2021}
    assert len(X_test) == 2
    assert len(y_test) == 2

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

    def test_fit_on_one_dataset_transform_another(self):
        # Goal: verify transform uses statistics learned from the fit dataset on new data.
        train_df = pd.DataFrame(
            {
                "u10_0": [0.0, 1.0, 2.0, 3.0],
                "u10_6": [10.0, 11.0, 12.0, 13.0],
                "v10_0": [20.0, 21.0, 22.0, 23.0],
                "v10_6": [30.0, 31.0, 32.0, 33.0],
                "hour_of_day": [0, 6, 12, 18],
                "day_of_year": [100, 101, 102, 103],
            }
        )
        test_df = pd.DataFrame(
            {
                "u10_0": [1.5, 2.5],
                "u10_6": [11.5, 12.5],
                "v10_0": [21.5, 22.5],
                "v10_6": [31.5, 32.5],
                "hour_of_day": [3, 15],
                "day_of_year": [100.5, 102.5],
            }
        )

        scaler = InputPerVariableScaler(aux_variables=["hour_of_day", "day_of_year"])
        scaler.fit(train_df)
        transformed = scaler.transform(test_df)

        expected = (test_df - train_df.mean()) / train_df.std(ddof=0)

        assert list(transformed.columns) == list(test_df.columns)
        assert np.allclose(transformed.values, expected.values, atol=1e-12)


class TestTrainBaselineModelGridSearch:
    def test_grid_search_passes_sample_weights_to_cv_fit(self, patched_grid_search_setup):
        train_module.train_baseline_model_grid_search(
            site="dummy-site",
            model_type="mlp",
            param_grid={"alpha": [0.1]},
            data_kwargs={"time_shift_hours": [[6]], "sample_weights": [None, 3.0]},
            validation_keys=["time_shift_hours"],
        )

        assert len(DummyGridSearchCV.instances) == 2
        weighted_fit = [inst for inst in DummyGridSearchCV.instances if inst.fit_sample_weight is not None]
        assert len(weighted_fit) == 1

        sample_weight = weighted_fit[0].fit_sample_weight
        assert len(sample_weight) == 6
        assert np.allclose(sample_weight, np.array([3.0, 1.0, 3.0, 1.0, 1.0, 1.0]))

    def test_grid_search_does_not_forward_sample_weights_to_data_loader(self, patched_grid_search_setup):
        train_module.train_baseline_model_grid_search(
            site="dummy-site",
            model_type="mlp",
            param_grid={"alpha": [0.1]},
            data_kwargs={"time_shift_hours": [[6]], "sample_weights": [None, 2.0]},
            validation_keys=["time_shift_hours"],
        )

        for _, kwargs in patched_grid_search_setup["calls"]:
            assert "sample_weights" not in kwargs

    def test_grid_search_rejects_unknown_data_kwargs_key(self, patched_grid_search_setup):
        with pytest.raises(ValueError, match="Unknown data_kwargs keys"):
            train_module.train_baseline_model_grid_search(
                site="dummy-site",
                model_type="mlp",
                param_grid={"alpha": [0.1]},
                data_kwargs={"time_shift_hours": [[6]], "bad_key": [123]},
            )

    def test_grid_search_uses_winning_sample_weights_in_final_fit(self, patched_grid_search_setup):
        best_model, _, _, best_combo_kw = train_module.train_baseline_model_grid_search(
            site="dummy-site",
            model_type="mlp",
            param_grid={"alpha": [0.1]},
            data_kwargs={"time_shift_hours": [[6]], "sample_weights": [None, 3.0]},
            validation_keys=["time_shift_hours"],
        )

        assert isinstance(best_model, DummyClassifier)
        assert best_combo_kw["sample_weights"] == 3.0
        assert best_model.fit_sample_weight is not None

        expected_weights = pd.Series([3.0, 1.0, 3.0, 1.0], index=patched_grid_search_setup["train_index"])
        pd.testing.assert_series_equal(best_model.fit_sample_weight, expected_weights)

    def test_grid_search_returns_cv_scores_with_data_kwarg_metadata(self, patched_grid_search_setup):
        _, _, _, best_combo_kw, all_cv_results = train_module.train_baseline_model_grid_search(
            site="dummy-site",
            model_type="mlp",
            param_grid={"alpha": [0.1]},
            data_kwargs={"balance": [-1], "time_shift_hours": [[6]], "sample_weights": [None, 2.0]},
            validation_keys=["time_shift_hours"],
            return_cv_scores=True,
        )

        assert "sample_weights" in best_combo_kw
        assert "data_kwarg" in all_cv_results.columns
        assert "param_data_sample_weights" in all_cv_results.columns
        assert "param_data_balance" in all_cv_results.columns
        assert set(all_cv_results["param_data_sample_weights"].unique()) == {"None", "2.0"}


## TODO add tests checking it works within the pipeline after file merging with main branch