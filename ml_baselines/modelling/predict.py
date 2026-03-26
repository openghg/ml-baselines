import numpy as np
import pandas as pd

from sklearn.metrics import precision_score, recall_score, f1_score

from ml_baselines.modelling.train import get_train_test_data
from ml_baselines.modelling.plot import plot_confusion_matrix, plot_obs, plot_obs_with_labels, plot_monthly_means, plot_baseline_count_hist

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

    monthly_pred = df_pred.resample("ME").agg({"mf": ["mean", "std", "count"]})
    monthly_true = df_true.resample("ME").agg({"mf": ["mean", "std", "count"]})

    monthly_pred.columns = ["pred_monthly_mf", "pred_monthly_std", "pred_monthly_count"]
    monthly_true.columns = ["true_monthly_mf", "true_monthly_std", "true_monthly_count"]

    monthly_means = pd.concat([monthly_pred, monthly_true], axis=1)
    monthly_means.index = monthly_means.index.to_period("M").to_timestamp()

    return monthly_means

class BaselineLabelledObservations:
    def __init__(self, y, y_pred, df_obs, site, species):
        """
        Object to hold observed molefractions along with true and predicted baseline labels, and provide methods for plotting these observations the labels. Also provides a method to calculate monthly means of the observed molefractions for true and predicted baselines.

        Parameters:
        - y: The true labels (observed baselines) as a pandas Series with a datetime index.
        - y_pred: The predicted labels (predicted baselines) pandas Series with a datetime
        index.
        - df_obs: A DataFrame containing the observed molefractions in a column named "mf" with a datetime index. Use read_agage to read in the observed data.
        - site: the site the obs and baselines refer to
        - species: the species the obs refer to

        Attributes:
        - labelled_df: A DataFrame containing the observed molefractions, true baseline labels,
        and predicted baseline labels, all aligned by their datetime index.
        - monthly_means: A DataFrame containing the monthly mean molefractions. The attribute is calculated using the calculate_monthly_means method

        """
        self.site = site
        self.species = species
        self.labelled_df = align_predictions_and_obs(y, y_pred, df_obs)
        
    def plot_confusion_matrix(self, normalise=True, title="Confusion Matrix"):
        plot_confusion_matrix(self.labelled_df["baseline"], self.labelled_df["predicted_baseline"], normalise=normalise, title=title)
        
    def plot_obs(self, title=None):
        if title is None:
            title = f"MF of {self.species.upper()} at {self.site} with InTEM baseline labels"
        plot_obs(self.labelled_df, title=title)

    def plot_obs_with_labels(self, title=None, plot_true_negatives=True):
        if title is None:
            title = f"MF of {self.species.upper()} at {self.site} with baseline and model Predictions"
        plot_obs_with_labels(self.labelled_df, title=title, site=self.site, plot_true_negatives=plot_true_negatives)

    def calculate_monthly_means(self):
        if not hasattr(self, "monthly_means"):
            self.monthly_means = calculate_monthly_means(self.labelled_df)
        else:
            print("Monthly means have already been calculated. Use the 'monthly_means' attribute to access the DataFrame containing these means.")


    def plot_monthly_means(self, shade_train_and_val_periods=True, plot_obs=True, plot_count_hist=False):
        if not hasattr(self, "monthly_means"):
            self.calculate_monthly_means()

        obs_df = self.labelled_df if plot_obs else None

        plot_monthly_means(self.monthly_means, shade_train_and_val_periods=shade_train_and_val_periods, site=self.site, obs_df=obs_df, plot_count_hist=plot_count_hist)

    def plot_baseline_count_hist(self):
        if not hasattr(self, "monthly_means"):
            self.calculate_monthly_means()

        plot_baseline_count_hist(self.monthly_means)