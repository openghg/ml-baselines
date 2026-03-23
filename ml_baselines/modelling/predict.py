import numpy as np
import pandas as pd

from sklearn.metrics import precision_score, recall_score, f1_score

from ml_baselines.modelling.train import get_train_test_data

def predict_baselines(site, model, time_shift_hours=[6], prediction_threshold=0.5, prediction_mode="validation", verbose=True, save_preds = False):
    """
    Predict baseline events for a given site using a trained model.
    
    Parameters:
    - site: The site for which to make predictions.
    - model: The trained model to use for predictions.
    - time_shift_hours: List of time shifts (in hours) to apply to the data for prediction.
    - prediction_threshold: Threshold for converting predicted probabilities to binary predictions.
    - prediction_mode: Whether to predict on the "validation" or "test" set.
    - verbose: Whether to print detailed information about the prediction process.
    - save_preds: Whether to save the predictions to disk (not yet implemented).
    
    Returns:
    - y: The true labels for the prediction set.
    - y_pred: The predicted labels for the prediction set.
    - metrics: A dictionary containing precision, recall, and F1 score.
    """

    if prediction_mode not in ["validation", "test"]:
        raise ValueError("mode must be either 'validation' or 'test'")  
    
    if verbose: print(f"Predicting on {prediction_mode} set for site: {site} with time shifts: {time_shift_hours} and prediction threshold: {prediction_threshold}")

    X, y = get_train_test_data(site, prediction_mode, time_shift_hours=time_shift_hours, verbose=verbose)    

    y_pred = (model.predict_proba(X)[:, 1] >= prediction_threshold).astype(int)

    y_pred = pd.Series(y_pred, index=y.index)

    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)

    if verbose:
        print(f"Precision: {precision:.3f}")
        print(f"Recall: {recall:.3f}")
        print(f"F1 Score: {f1:.3f}")

    if save_preds:
        raise NotImplementedError("Saving predictions is not yet implemented!")
    
    return y, y_pred, {"precision": precision, "recall": recall, "f1": f1}



