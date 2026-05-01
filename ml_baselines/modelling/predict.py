import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib
import glob
from pathlib import Path

from scipy import stats
from sklearn.metrics import precision_score, recall_score, f1_score
from statsmodels.tsa.seasonal import STL

from ml_baselines.modelling.train import get_train_test_data
from ml_baselines.modelling.plot import plot_confusion_matrix, plot_obs, plot_obs_with_labels, plot_monthly_means, plot_baseline_count_hist, plot_model_confidence, plot_stl_components

from ml_baselines.config import Config
cfg = Config()

def load_baseline_model(site, model_type="mlp", models_folder=cfg.models_path, timestamp=None, suffix=None, verbose=True):
    """
    Load a trained model and its extra info from a specified folder. If timestamp is provided, it will look for a model file with that timestamp; otherwise, it will load the most recent model file for the given site and model type.

    Args:
    site (str): The site for which to load the model.
    model_type (str): The type of model to load (default is "mlp").
    models_folder (str): The folder where models are stored (default is cfg.models_path).
    timestamp (str, optional): The timestamp to look for in the model filename. If None, loads the most recent model.
    suffix (str, optional): A suffix to append to the model filename. Useful to identify files instead of relying on timestamps, or to distinguish between different versions of models trained at the same time.
    Returns:
    model: The loaded model object.
    extra_info: A dictionary containing extra information about the model (e.g., scores, scaler, model inputs).
    """

    model_path = Path(models_folder) / site / f"{model_type}_model_*.joblib"
    if timestamp is not None and suffix is not None:
        model_path = model_path.with_name(f"{model_type}_model_{timestamp}_{suffix}.joblib")
    elif timestamp is not None:
        model_path = model_path.with_name(f"{model_type}_model_{timestamp}*.joblib")
    elif suffix is not None:
        model_path = model_path.with_name(f"{model_type}_model_*{suffix}.joblib")
        
    model_file = sorted(glob.glob(str(model_path)))
    if not model_file:
        raise FileNotFoundError(f"No model found at {model_path}")

    model_dict = joblib.load(model_file[-1])  # Load the most recent model file

    if verbose: print(f"Loaded model from {model_file[-1]}")

    return model_dict['model'], model_dict['info']


def predict_baselines(site, model, time_shift_hours=[6, 12, 18, 24], prediction_threshold=0.5, prediction_mode="validation", scaler=None, verbose=True, save_preds = False, return_proba=False):
    """
    Predict baseline events for a given site using a trained model.

    Args:
        site: The site for which to make predictions.
        model: The trained model to use for predictions.
        time_shift_hours: List of time shifts (in hours) to apply to the data
            for prediction.
        prediction_threshold: Threshold for converting predicted probabilities
            to binary predictions.
        prediction_mode: Whether to predict on the "validation", "test" or "full" set, where "full" is either the custom period defined in the config or the entire period from the start of training to the end of testing.
        verbose: Whether to print detailed information about the prediction
            process.
        save_preds: Whether to save the predictions to disk (not yet implemented).
        return_proba: Whether to return predicted probabilities as well as
            binary predictions.

    Returns:
        tuple: The true labels, predicted labels, and metrics dictionary. If
            ``return_proba`` is True, predicted probabilities are included as
            well.
    """

    if verbose: print(f"Predicting on {prediction_mode} set for site: {site} with prediction threshold: {prediction_threshold}")

    X, y = get_train_test_data(site, prediction_mode, time_shift_hours=time_shift_hours, verbose=verbose)    

    if scaler is not None:
        X = scaler.transform(X)

    y_pred = (model.predict_proba(X)[:, 1] >= prediction_threshold).astype(int)
    y_proba = model.predict_proba(X)[:, 1]

    y_pred = pd.Series(y_pred, index=y.index)
    y_proba = pd.Series(y_proba, index=y.index)

    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)

    if verbose:
        print(f"Precision: {precision:.3f}")
        print(f"Recall: {recall:.3f}")
        print(f"F1 Score: {f1:.3f}")

    if save_preds:
        raise NotImplementedError("Saving predictions is not yet implemented!")
    
    if return_proba:
        return y, y_pred, y_proba, {"precision": precision, "recall": recall, "f1": f1}
    else:
        return y, y_pred, {"precision": precision, "recall": recall, "f1": f1}


def align_predictions_and_obs(y, y_pred, df_obs, y_proba=None):
    """
    Align predictions with observed data using the indices of both time series.

    Args:
        y: The true labels (observed baselines) as a pandas Series with a
            datetime index.
        y_pred: The predicted labels (predicted baselines) as a pandas Series
            with a datetime index.
        df_obs: A DataFrame containing the observed molefractions in a column
            named "mf" with a datetime index.
        y_proba: Optional predicted probabilities aligned with ``y_pred``.

    Returns:
        pd.DataFrame: A DataFrame containing the observed molefractions, true
            baseline labels, and predicted baseline labels, all aligned by
            their datetime index.
    """
    y = y.copy()
    df_obs = df_obs.copy()
    y.index = y.index.astype("datetime64[ns]")
    df_obs.index = df_obs.index.astype("datetime64[ns]")

    y = y[(y.index.year >= min(df_obs.index.year)) & (y.index.year <= max(df_obs.index.year))]
    if y.empty:
        raise ValueError(f"No overlap between prediction period and observation period.")
    labelled_df = pd.merge_asof(pd.DataFrame({"baseline": y}), df_obs["mf"], left_index=True, right_index=True, direction='nearest')
    labelled_df = labelled_df[["mf", "baseline"]]
    y_pred = y_pred.reindex(labelled_df.index, method="nearest")
    labelled_df["predicted_baseline"] = y_pred

    if y_proba is not None:
        y_proba = y_proba.reindex(labelled_df.index, method="nearest")
        labelled_df["predicted_proba"] = y_proba
    return labelled_df


def calculate_true_baseline_cv(labelled_df, remove_seasonality=True, plot_stl=False):
    """
    Quantifies the noise in the baseline molefractions as an assessment of model utility.

    Args:
        labelled_df (pd.DataFrame): A DataFrame containing the observed molefractions in a
            column named "mf", true baseline labels in a column named
            "baseline", and predicted baseline labels in a column named
            "predicted_baseline". The DataFrame should have a datetime index.
        remove_seasonality (bool): Whether to remove seasonal trends from the data using STL decomposition.
            If True, also removes long-term trend. If False, data is detrended using a simple linear fit.
        plot_stl (bool): Whether to plot the STL decomposition components (observed, trend, seasonal, and residuals).
            Only valid when remove_seasonality=True.

    Returns:
        cv_dict (dict): A dictionary containing:
            "pct_monthly_coverage" (float): The percentage of months with sufficient baseline points for a monthly mean. 
                If less than 80%, the calculated CV may be unreliable if removing seasonal trends.
            "cv" (float): The calculated coefficient of variation
    """

    cv_dict = {}

    # extract true baselines and resample to monthly means
    true_baselines = labelled_df[labelled_df["baseline"] == 1]["mf"]
    orig_monthly_means = true_baselines.resample("ME").mean().dropna()

    # calculate coverage and store
    monthly_coverage = len(orig_monthly_means) / len(pd.date_range(true_baselines.index.min(), true_baselines.index.max(), freq="ME"))
    cv_dict['pct_monthly_coverage'] = monthly_coverage * 100

    if remove_seasonality:
        # remove seasonal variation and long-term trends
        monthly_means_filled = orig_monthly_means.resample("ME").asfreq().interpolate(method="time")
        stl = STL(monthly_means_filled, period=12, robust=True)
        stl_result = stl.fit()
        residuals = stl_result.resid

        if plot_stl:
            plot_stl_components(true_baselines, orig_monthly_means, monthly_means_filled, stl_result)

    else:
        # detrend by removing a simple linear fit
        x = np.arange(len(orig_monthly_means))
        slope, intercept, _, _, _ = stats.linregress(x, orig_monthly_means.values)
        residuals = orig_monthly_means.values - (intercept + slope * x)

    cv = residuals.std() / orig_monthly_means.mean()
    cv_dict['cv'] = cv.item()

    return cv_dict


def assess_true_baselines(labelled_df):
    """
    Assesses the InTEM/"true" baselines by identifying anomalies and months with a low baseline ratio.
    Considers points outside 3 standard deviations from the mean for a given month to be anomalous.
    Also quantifies the percentage of these true anomalies also considered baseline by the model.

    Args:
        labelled_df (pd.DataFrame): A DataFrame containing the observed molefractions in a
            column named "mf", true baseline labels in a column named
            "baseline", and predicted baseline labels in a column named
            "predicted_baseline". The DataFrame should have a datetime index.

    Returns:
        assessment_dict (dict): A dictionary containing:
            - "low_baseline_months" (list): List of months where baseline points are less than 5% of total observations that month.
            - "pct_low_baseline_months": Percentage of total months that have a low baseline ratio (<5%)
            - "true_baseline_anomalies" (pd.DataFrame): DataFrame containing the anomalous observations.
            - "pct_anomalous" (float): Percentage of true baseline points that are anomalous (outside 3 std), given as an average across months.
            - "pct_anomalies_true_pos" (float): Percentage of true baseline points considered to be anomalous that are also classified as baseline by the model.
    """

    low_baseline_months = []
    monthly_anomalies = []
    total_months = 0
    for month, group in labelled_df.groupby(pd.Grouper(freq='ME')):
        total = len(group)
        if total == 0:
            continue
        total_months += 1
        baselines = group[group["baseline"]==1]
        n_baselines = len(baselines)

        # Check for low baseline months
        if n_baselines / total < 0.05:
            low_baseline_months.append(month)

        # Identify baseline anomalies
        if n_baselines > 0:
            monthly_mean = baselines["mf"].mean()
            monthly_std = baselines["mf"].std()
            anomalies = baselines[np.abs(baselines["mf"] - monthly_mean) > 3 * monthly_std]
            monthly_anomalies.append(anomalies)

    pct_low_baseline_months = 100 * len(low_baseline_months) / total_months
    true_baseline_anomalies = pd.concat(monthly_anomalies) if monthly_anomalies else pd.DataFrame()
    pct_anomalous = 100 * len(true_baseline_anomalies) / len(labelled_df[labelled_df["baseline"] == 1]) if len(true_baseline_anomalies) > 0 else 0.0
    pct_anomalies_true_pos = 100 * len(true_baseline_anomalies[true_baseline_anomalies["predicted_baseline"] == 1]) / len(true_baseline_anomalies) if len(true_baseline_anomalies) > 0 else 0.0

    assessment_dict = {
        'low_baseline_months': low_baseline_months,
        'pct_low_baseline_months': pct_low_baseline_months,
        'true_baseline_anomalies': true_baseline_anomalies,
        'pct_anomalous': pct_anomalous,
        'pct_anomalies_true_pos': pct_anomalies_true_pos
    }

    return assessment_dict


def calculate_monthly_means(labelled_df, add_stats=True):
    """
    Calculate monthly means of the observed molefractions.

    Args:
        labelled_df (pd.DataFrame): A DataFrame containing the observed molefractions in a
            column named "mf", true baseline labels in a column named
            "baseline", and predicted baseline labels in a column named
            "predicted_baseline". The DataFrame should have a datetime index.
        add_stats (bool): Whether to add MAE, MAPE, bias, and coefficient of variation for each month as additional columns.

    Returns:
        pd.DataFrame: A DataFrame containing the monthly mean molefractions and
            related statistics for each month.
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
    monthly_means["pred_monthly_count"] = monthly_means["pred_monthly_count"].fillna(0).astype(int)
    monthly_means["true_monthly_count"] = monthly_means["true_monthly_count"].fillna(0).astype(int)

    if add_stats:
        monthly_means["true_coeffvariation"] = monthly_means["true_monthly_std"] / monthly_means["true_monthly_mf"]
        monthly_means["MAE"] = np.abs(monthly_means["pred_monthly_mf"] - monthly_means["true_monthly_mf"])
        monthly_means["MAPE"] = monthly_means["MAE"] / monthly_means["true_monthly_mf"]
        monthly_means["bias"] = monthly_means["pred_monthly_mf"] - monthly_means["true_monthly_mf"]

    return monthly_means


class BaselineLabelledObservations:
    def __init__(self, y, y_pred, df_obs, site, species, y_proba=None):
        """
        Object to hold observed molefractions along with true and predicted
        baseline labels, and provide methods for plotting these observations and
        the labels. Also provides a method to calculate monthly means of the
        observed molefractions for true and predicted baselines.

        Args:
            y: The true labels (observed baselines) as a pandas Series with a
                datetime index.
            y_pred: The predicted labels (predicted baselines) as a pandas
                Series with a datetime index.
            df_obs: A DataFrame containing the observed molefractions in a
                column named "mf" with a datetime index.
            site: The site the observations and baselines refer to.
            species: The species the observations refer to.
            y_proba: Optional predicted probabilities aligned with ``y_pred``.

        Attributes:
            labelled_df: A DataFrame containing the observed molefractions,
                true baseline labels, and predicted baseline labels, as well as
                other columns such as predicted probabilities if available, all
                aligned by their datetime index.
            monthly_means: A DataFrame containing the monthly mean
                molefractions. The attribute is calculated using the
                calculate_monthly_means method.
        """
        self.site = site
        self.species = species
        self.labelled_df = align_predictions_and_obs(y, y_pred, df_obs, y_proba=y_proba)
        if len(self.labelled_df) == 0:
            raise ValueError(f"No overlapping time period between model predictions and AGAGE data for {species}.")

        self.data_periods = {
        "train": cfg.training_period[site], 
        "validation": cfg.validation_period[site],
        "test": cfg.testing_period[site],
        "full": cfg.full_period[site] if cfg.full_period[site] is not None else (cfg.training_period[site][0], cfg.testing_period[site][1])
        }


    def get_prediction_scores(self, verbose=True):
        """
        Calculate precision, recall, and F1 score for each data period.

        The scores are stored in a dictionary attribute called ``scores``, with
        keys corresponding to each data period if available.
        """
        scores = {}
        for period in self.data_periods:
            period_df = self.labelled_df[(self.labelled_df.index.year >= self.data_periods[period][0]) & (self.labelled_df.index.year <= self.data_periods[period][1])]
            if len(period_df) == 0:
                continue
            precision = precision_score(period_df["baseline"], period_df["predicted_baseline"])
            recall = recall_score(period_df["baseline"], period_df["predicted_baseline"])
            f1 = f1_score(period_df["baseline"], period_df["predicted_baseline"])
            if verbose: print(f"{period[:6]} set - Precision: {precision:.3f}, Recall: {recall:.3f}, F1 Score: {f1:.3f}")

            scores[period] = {"precision": precision, "recall": recall, "f1": f1}
        
            self.scores = scores


    def calculate_true_baseline_cv(self, remove_seasonality=True, plot_stl=False, verbose=True):
        if not hasattr(self, "baseline_cv"):
            self.baseline_cv = calculate_true_baseline_cv(self.labelled_df, remove_seasonality=remove_seasonality, plot_stl=plot_stl)
            if remove_seasonality and self.baseline_cv['pct_monthly_coverage'] < 75:
                print(f"WARNING: Monthly coverage is {self.baseline_cv['pct_monthly_coverage']:.2f}%. STL decomposition may be unreliable.")
            if verbose: print(f"Baseline CV: {self.baseline_cv['cv']:.4f}")
        else:
            print("Baseline CV has already been calculated. Use the 'baseline_cv' attribute to access the result.")


    def assess_true_baselines(self, verbose=True):
        if not hasattr(self, "true_baseline_assessment"):
            self.true_baseline_assessment = assess_true_baselines(self.labelled_df)
            if verbose: print(f"Number of months with a low baseline ratio (<5%): {len(self.true_baseline_assessment['low_baseline_months'])} ({self.true_baseline_assessment['pct_low_baseline_months']:.2f}% of all months)")
            if self.true_baseline_assessment['pct_anomalous'] > 0:
                if verbose: print(f"Percentage of true baselines considered anomalous: {self.true_baseline_assessment['pct_anomalous']:.2f}%")
                pct_anomalies_true_pos = self.true_baseline_assessment['pct_anomalies_true_pos']
                if pct_anomalies_true_pos > 50:
                    if verbose: print(f"    WARNING: {self.true_baseline_assessment['pct_anomalies_true_pos']:.2f}% of true baseline points considered anomalous are also classified as baseline by the model. "
                                       "The model may be learning from these 'anomalous' InTEM labels.")
                    else: print(f"WARNING: {self.true_baseline_assessment['pct_anomalous']:.2f}% of true baselines are considered anomalous and {self.true_baseline_assessment['pct_anomalies_true_pos']:.2f}% of these are also classified as baseline by the model. "
                                 "The model may be learning from these 'anomalous' InTEM labels.")
        else:
            print("True baselines have already been assessed. Use the 'true_baseline_assessment' attribute to access the results.")


    def calculate_monthly_means(self, verbose=True):
        if not hasattr(self, "monthly_means"):
            self.monthly_means = calculate_monthly_means(self.labelled_df)
            self.find_monthly_anomalies(verbose=verbose)
        else:
            print("Monthly means have already been calculated. Use the 'monthly_means' attribute to access the DataFrame containing these means.")


    def find_monthly_anomalies(self, verbose=True):
        if not hasattr(self, "monthly_means"):
            self.calculate_monthly_means()

        missing_months = self.monthly_means[((self.monthly_means["pred_monthly_count"].isna()) | (self.monthly_means["pred_monthly_count"] == 0)) & (self.monthly_means["true_monthly_count"] > 0)].index
            
        # find months where the predicted monthly mean is 1,3 and 5 standard deviations away from the true monthly mean

        deviation_thresholds = [1, 3, 5]
        anomaly_months = {}
        for threshold in deviation_thresholds:
            anomaly_months[threshold] = self.monthly_means[(np.abs(self.monthly_means["pred_monthly_mf"] - self.monthly_means["true_monthly_mf"]) > threshold * self.monthly_means["true_monthly_std"]) & (self.monthly_means["pred_monthly_count"] > 0) & (self.monthly_means["true_monthly_count"] > 0)].index
        self.anomaly_months = anomaly_months
        self.missing_months = missing_months

        # add a column to monthly_means indicating whether each month is an anomaly month indicating which threshold or a missing month
        self.monthly_means["is_anomaly"] = 0
        for threshold in deviation_thresholds:
            self.monthly_means.loc[self.monthly_means.index.isin(anomaly_months[threshold]), "is_anomaly"] = threshold

        self.monthly_means["is_missing"] = self.monthly_means.index.isin(missing_months)

        if verbose: print(f"Found {len(missing_months)} missing months, and {sum(self.monthly_means['is_anomaly'] > 0)} anomaly months (with {sum(self.monthly_means['is_anomaly'] == 1)}, {sum(self.monthly_means['is_anomaly'] == 3)}, and {sum(self.monthly_means['is_anomaly'] == 5)} months with deviations greater than 1, 3, and 5 standard deviations respectively).")

    def print_monthly_stats(self, verbose=True):
        # self.labelled_df already has columns mae, mape etc so just need to print the mean of these columns across the dataset
        if not hasattr(self, "monthly_means"):
            self.calculate_monthly_means(verbose=verbose)

        scores = {}
        for period in self.data_periods:
            period_df = self.monthly_means[(self.monthly_means.index.year >= self.data_periods[period][0]) & (self.monthly_means.index.year <= self.data_periods[period][1])]

            if len(period_df) == 0:
                continue    
            
            mae = np.mean(period_df["MAE"])
            mape = np.mean(np.abs(period_df["MAPE"]))
            bias = np.mean(period_df["bias"])
            rmse = np.sqrt(np.mean((period_df["MAE"] ** 2)))

            if verbose: print(f"{period[:6]} set - MAE: {mae:.3f}, MAPE: {100*mape:.3f}%, bias: {bias:.3f}, RMSE: {rmse:.3f}")

            scores[period] = {"MAE": mae, "MAPE": mape, "bias": bias, "RMSE": rmse}

        self.monthly_scores = scores

    ## CALLS TO PLOTTING FUNCTIONS
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

    def plot_model_confidence(self, title=None, cmap=None):
        if "predicted_proba" not in self.labelled_df.columns:
            raise ValueError("Predicted probabilities are not available in the labelled_df. Please ensure that the predict_baselines function is called with return_proba=True, and that the resulting y_proba is included in the BaselineLabelledObservations object.")
        
        if title is None:
            title = f"Predicted probability of baseline events for {self.species.upper()} at {self.site}"
        
        plot_model_confidence(self.labelled_df, title=title, cmap=cmap, site=self.site, shade_train_and_val_periods=True)

    def plot_monthly_means(self, shade_train_and_val_periods=True, plot_obs=True, plot_count_hist=False, show_anomalies=False):
        if not hasattr(self, "monthly_means"):
            self.calculate_monthly_means()

        obs_df = self.labelled_df if plot_obs else None

        plot_monthly_means(self.monthly_means, shade_train_and_val_periods=shade_train_and_val_periods, site=self.site, obs_df=obs_df, plot_count_hist=plot_count_hist, show_anomalies=show_anomalies, title=f"Monthly Mean Molefractions for Predicted and True Baselines for {self.species.upper()} at {self.site}")

    def plot_baseline_count_hist(self):
        if not hasattr(self, "monthly_means"):
            self.calculate_monthly_means()

        plot_baseline_count_hist(self.monthly_means)