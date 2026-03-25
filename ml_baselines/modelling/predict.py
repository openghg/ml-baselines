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

    if prediction_mode not in ["train", "validation", "test", "full"]:
        raise ValueError("mode must be either 'train', 'validation', 'test', or 'full'")  
    
    if verbose: print(f"Predicting on {prediction_mode} set for site: {site} with prediction threshold: {prediction_threshold}")

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


def align_predictions_and_obs(y, y_pred, df_obs):
    """
    Align predictions with observed data, using the indeces of both timeseries.

    Parameters:
    - y: The true labels (observed baselines) as a pandas Series with a datetime index.
    - y_pred: The predicted labels (predicted baselines) pandas Series with a datetime index.
    - df_obs: A DataFrame containing the observed molefractions in a column named "mf" with a datetime index. Use read_agage to read in the observed data.

    Returns:
    - labelled_df: A DataFrame containing the observed molefractions, true baseline labels, and predicted baseline labels, all aligned by their datetime index.
    """

    y = y[(y.index.year >= min(df_obs.index.year)) & (y.index.year <= max(df_obs.index.year))]
    labelled_df = pd.merge_asof(pd.DataFrame({"baseline": y}), df_obs["mf"], left_index=True, right_index=True, direction='nearest')
    labelled_df = labelled_df[["mf", "baseline"]]
    labelled_df["predicted_baseline"] = y_pred
    return labelled_df



def calculate_monthly_means(labelled_df):
    """
    Calculate monthly means of the observed molefractions, and the percentage of baseline labels in each month.

    Parameters:
    - labelled_df: A DataFrame containing the observed molefractions in a column named "mf", true baseline labels in a column named "baseline", and predicted baseline labels in a column named "predicted_baseline". The DataFrame should have a datetime index.

    Returns:
    - monthly_means: A DataFrame containing the monthly mean molefractions and the percentage of baseline labels for each month.
    """
    df_pred = labelled_df[labelled_df["predicted_baseline"] == 1]
    df_true = labelled_df[labelled_df["baseline"] == 1]

    ## TODO print a warning if there are months with no predicted or true baselines

    monthly_pred_mean = df_pred.resample('ME').agg({'mf': 'mean'})
    monthly_pred_std = df_pred.resample('ME').agg({'mf': 'std'})
    monthly_true_mean = df_true.resample('ME').agg({'mf': 'mean'})
    monthly_true_std = df_true.resample('ME').agg({'mf': 'std'})
    monthly_means = pd.concat([monthly_pred_mean, monthly_pred_std, monthly_true_mean, monthly_true_std], axis=1)
    monthly_means.columns = ["pred_monthly_mf", "pred_monthly_std", "true_monthly_mf", "true_monthly_std"]
    monthly_means.index = monthly_means.index.to_period('M').to_timestamp()
    return monthly_means