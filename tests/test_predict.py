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