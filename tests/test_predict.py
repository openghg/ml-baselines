import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier

from ml_baselines.modelling import predict as predict_baselines
from ml_baselines.modelling import train as train_baselines


def fake_get_train_test_data(site, prediction_mode, time_shift_hours, verbose):
    fake_X = pd.DataFrame({
        "u10_0": [-6.923574, -9.106857, -11.585869, -11.628951],
        "v10_0": [6.368086, 3.760457, 4.375037, 5.610230]
    }, index=pd.to_datetime(["2016-01-01 07:00:00", "2016-01-01 09:00:00", "2016-01-01 11:00:00", "2016-01-01 13:00:00"]))
    fake_y = pd.Series([0, 1, 0, 1], index=fake_X.index, name="baseline")

    return fake_X, fake_y


def test_predict_baselines_threshold(monkeypatch):

    monkeypatch.setattr(train_baselines, "get_train_test_data", fake_get_train_test_data)
    monkeypatch.setattr(predict_baselines, "get_train_test_data", train_baselines.get_train_test_data)

    fake_X, fake_y = train_baselines.get_train_test_data("MHD", "validation", [6], False)

    dummy_model = DummyClassifier(strategy="constant", constant=1)
    dummy_model.fit(fake_X, fake_y)
    
    y_true, y_pred, y_proba, metrics = predict_baselines.predict_baselines(
        site="MHD",
        model=dummy_model,
        prediction_threshold=0.5,
        verbose=False,
        return_proba=True
    )

    assert y_pred.tolist() == [1, 1, 1, 1]
    assert y_pred.index.equals(y_true.index)
    assert set(metrics.keys()) == {"precision", "recall", "f1"}
    ## assert that precision is 0.5 and recall is 1.0
    assert np.isclose(metrics["precision"], 0.5)
    assert np.isclose(metrics["recall"], 1)
    assert np.allclose(y_proba, 1.0)


def test_calculate_monthly_means_groups_and_calculates_stats():
    labelled_df = pd.DataFrame(
        {
            "mf": [10.0, 14.0, 20.0, 30.0, 40.0, 50.0],
            "baseline": [1, 0, 1, 0, 1, 1],
            "predicted_baseline": [1, 1, 0, 0, 1, 0],
        },
        index=pd.to_datetime(
            [
                "2016-01-05",
                "2016-01-15",
                "2016-01-20",
                "2016-02-01",
                "2016-02-10",
                "2016-02-20",
            ]
        ),
    )

    monthly_means = predict_baselines.calculate_monthly_means(labelled_df)

    assert list(monthly_means.columns) == [
        "pred_monthly_mf",
        "pred_monthly_std",
        "pred_monthly_count",
        "true_monthly_mf",
        "true_monthly_std",
        "true_monthly_count",
        "true_coeffvariation",
        "MAE",
        "MAPE",
        "bias",
    ]

    assert monthly_means.index.equals(pd.to_datetime(["2016-01-01", "2016-02-01"]))

    january = monthly_means.loc[pd.Timestamp("2016-01-01")]
    february = monthly_means.loc[pd.Timestamp("2016-02-01")]

    assert np.isclose(january["pred_monthly_mf"], 12.0)
    assert np.isclose(january["pred_monthly_std"], np.sqrt(8.0))
    assert january["pred_monthly_count"] == 2
    assert np.isclose(january["true_monthly_mf"], 15.0)
    assert np.isclose(january["true_monthly_std"], np.sqrt(50.0))
    assert january["true_monthly_count"] == 2
    assert np.isclose(january["MAE"], 3.0)
    assert np.isclose(january["MAPE"], 0.2)
    assert np.isclose(january["bias"], -3.0)

    assert np.isclose(february["pred_monthly_mf"], 40.0)
    assert np.isnan(february["pred_monthly_std"])
    assert february["pred_monthly_count"] == 1
    assert np.isclose(february["true_monthly_mf"], 45.0)
    assert np.isclose(february["true_monthly_std"], np.sqrt(50.0))
    assert february["true_monthly_count"] == 2
    assert np.isclose(february["MAE"], 5.0)
    assert np.isclose(february["MAPE"], 5.0 / 45.0)
    assert np.isclose(february["bias"], -5.0)


def test_calculate_monthly_means_handles_no_baseline_data():
    labelled_df = pd.DataFrame(
        {
            "mf": [11.0, 12.0],
            "baseline": [0, 0],
            "predicted_baseline": [0, 0],
        },
        index=pd.to_datetime(["2016-01-05", "2016-01-15"]),
    )

    monthly_means = predict_baselines.calculate_monthly_means(labelled_df)

    assert monthly_means.empty
    assert np.all([x in list(monthly_means.columns) for x in [
        "pred_monthly_mf",
        "pred_monthly_std",
        "pred_monthly_count",
        "true_monthly_mf",
        "true_monthly_std",
        "true_monthly_count",
    ]])
    assert monthly_means.index.dtype == "datetime64[ns]"


def test_calculate_monthly_means_handles_true_but_no_predicted_baselines():
    labelled_df = pd.DataFrame(
        {
            "mf": [21.0, 23.0, 25.0],
            "baseline": [1, 1, 1],
            "predicted_baseline": [0, 0, 0],
        },
        index=pd.to_datetime(["2016-03-05", "2016-03-15", "2016-03-25"]),
    )

    monthly_means = predict_baselines.calculate_monthly_means(labelled_df)

    assert monthly_means.index.equals(pd.to_datetime(["2016-03-01"]))

    march = monthly_means.loc[pd.Timestamp("2016-03-01")]

    assert np.isnan(march["pred_monthly_mf"])
    assert np.isnan(march["pred_monthly_std"])
    assert march["pred_monthly_count"] == 0
    assert np.isclose(march["true_monthly_mf"], 23.0)
    assert np.isclose(march["true_monthly_std"], 2)
    assert march["true_monthly_count"] == 3
    assert np.isnan(march["MAE"])
    assert np.isnan(march["MAPE"])
    assert np.isnan(march["bias"])