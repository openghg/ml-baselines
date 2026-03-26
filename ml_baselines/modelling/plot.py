import datetime

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.dates as mdates

from ml_baselines.config import Config
from ml_baselines.modelling.predict import align_predictions_and_obs, calculate_monthly_means

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


cfg = Config()

models_path = cfg.models_path

def plot_confusion_matrix(y, y_pred, normalise=False, title="", labels = ["non-baseline", "baseline"]):

    if normalise:
        normalise = "all"
    else:
        normalise = None
        
    cm = confusion_matrix(y, y_pred, labels=np.unique(y), normalize=normalise)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                display_labels=["non-baseline", "baseline"])
    # change ax labels to "Predicted label" and "Intem label "

    
    disp.plot(cmap=plt.cm.Blues)

    disp.ax_.set_title(title)
    disp.ax_.set(xlabel='Predicted label', ylabel='InTEM label')


def plot_obs_with_labels(labelled_df, shade_train_and_val_periods=True, title="", site=None, plot_true_negatives=True):
    """
    Plot observed molefractions and predicted baseline labels over time, using confusion matrix labels (true positive, false positive etc)

    Parameters:
    - labelled_df: A DataFrame containing the observed molefractions in a column named "mf", true baseline labels in a column named "baseline", and predicted baseline labels in a column named "predicted_baseline". The DataFrame should have a datetime index.

    Returns:
    - A plot showing the observed molefractions and highlighting false positives, false negatives, and true positives.
    """

    true_positives_mask = (labelled_df.baseline == 1) & (labelled_df.predicted_baseline == 1)
    false_positives_mask = (labelled_df.baseline == 0) & (labelled_df.predicted_baseline == 1)
    false_negatives_mask = (labelled_df.baseline == 1) & (labelled_df.predicted_baseline == 0)
    true_negatives_mask = (labelled_df.baseline == 0) & (labelled_df.predicted_baseline == 0)

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(labelled_df.index, labelled_df.mf, c="gray", label="AGAGE Observations", zorder=1, lw=0.5, alpha=0.5)

    if plot_true_negatives:
        ax.scatter(labelled_df.index[true_negatives_mask], labelled_df.mf[true_negatives_mask], color="lightcoral", label="Correctly predicted as non-baseline (True Negatives)", zorder=5, s=0.5, alpha=0.5, marker='x')

    ax.scatter(labelled_df.index[false_positives_mask], labelled_df.mf[false_positives_mask], color="firebrick", label="Incorrectly predicted as baseline (False Positives)", zorder=5, s=0.5, marker='x')
    ax.scatter(labelled_df.index[false_negatives_mask], labelled_df.mf[false_negatives_mask], color="mediumaquamarine", label="Missed label (False Negatives)", zorder=5, s=0.5, marker='x', alpha=1)
    ax.scatter(labelled_df.index[true_positives_mask], labelled_df.mf[true_positives_mask], color="darkgreen", label="Correctly predicted as baseline (True Positives)", zorder=5, s=0.5, marker='x', alpha=0.8)

    ax.legend(markerscale=6)

    ax.set_ylabel("mole fraction in air / ppt")

    locator = mdates.MonthLocator() 
    #ax.set_minor_locator(locator)

    # set montly tick labels on x axis

    if shade_train_and_val_periods:
        if site is not None:
            ax.axvspan(pd.to_datetime(f"{cfg.training_period[site][0]}-01-01"), pd.to_datetime(f"{cfg.training_period[site][1]}-12-31"), alpha=0.1, label="Training Set", color='grey', zorder=0)
            ax.axvspan(pd.to_datetime(f"{cfg.validation_period[site][0]}-01-01"), pd.to_datetime(f"{cfg.validation_period[site][1]}-12-31"), alpha=0.1, label="Validation Set", color='purple', zorder=0)
        else:
            print("Could not shade training and validation periods because site is None. Please provide a site name to shade these periods.")
    
    ax.set_title(title)

    plt.show()


def plot_obs(labelled_df, labels_from="InTEM", title=""):
    """
    Plot observed molefractions.

    Parameters:
    - labelled_df: A DataFrame containing the observed molefractions in a column named "mf", true baseline labels in a column named "baseline", and predicted baseline labels in a column named "predicted_baseline". The DataFrame should have a datetime index.
    - labels_from: The source of the labels to plot (e.g., "InTEM").
    - site: The site for which to plot observed molefractions.

    Returns:
    - A plot showing the observed molefractions and highlighting false positives, false negatives, and true positives.
    """
    if labels_from == "InTEM":
        true_mask = (labelled_df.baseline == 1)
        false_mask = (labelled_df.baseline == 0)
    elif labels_from == "model":
        true_mask = (labelled_df.predicted_baseline == 1)
        false_mask = (labelled_df.predicted_baseline == 0)
    else:
        raise ValueError("Invalid value for 'labels_from'. Please choose 'InTEM' or 'model'.")

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(labelled_df.index, labelled_df.mf, c="gray", label="AGAGE Observations", zorder=1, lw=0.5, alpha=0.5)

    ax.scatter(labelled_df.index[false_mask], labelled_df.mf[false_mask], color="lightcoral", label=f"{labels_from} non-baseline", zorder=5, s=0.5, alpha=0.5, marker='x')
    ax.scatter(labelled_df.index[true_mask], labelled_df.mf[true_mask], color="darkgreen", label=f"{labels_from} baseline", zorder=5, s=0.5, marker='x', alpha=0.8)

    # make markers bigger in the legend
    ax.legend(markerscale=6)

    ax.set_ylabel("mole fraction in air / ppt")

    locator = mdates.MonthLocator()
    ax.xaxis.set_minor_locator(locator)
    ax.tick_params(axis='x', which='minor', length=2)

    ax.set_title(title)
    plt.show()

def plot_monthly_means(monthly_means, shade_train_and_val_periods=True, site=None, obs_df=None, plot_count_hist=False):
    """
    plot monthly means of the observed molefractions, and one standard deviation. Generate the dataset for this plot using the calculate_monthly_means function.

    Parameters:
    - monthly_means: A DataFrame containing the monthly mean molefractions with a timeindex
    - shade_train_and_val_periods: Whether to shade the training and validation periods on the plot (requires parameter site to be provided)
    - site: The site for which to plot observed molefractions. Required if shade_train_and_val_periods is True.
    - obs_df: Optional. If passed (a DataFrame containing the observed molefractions in a column named "mf" with a datetime index), plots the observed molefractions as a gray line in the background of the plot.
    - plot_count_hist: Optional. If True, plots a histogram of the monthly baseline counts on a twin axis.
    """
    fig, ax = plt.subplots(figsize=(12,6))

    ax.plot(monthly_means.index,
                monthly_means["true_monthly_mf"], label="InTEM Baseline Monthly Mean", marker='o', color='darkgreen', markersize=3)
    
    ax.plot(monthly_means.index, monthly_means["pred_monthly_mf"], label="Predicted Baseline Monthly Mean", color='royalblue', marker='o', markersize=3, alpha=0.8)

    # if predicted data is missing for a month, plot a red marker with the same value as the true monthly mean for that month, to show that the model is missing data for that month
    missing_pred_data_mask = monthly_means["pred_monthly_mf"].isna()
    if missing_pred_data_mask.any():
        ax.scatter(monthly_means.index[missing_pred_data_mask], monthly_means["true_monthly_mf"][missing_pred_data_mask], color='red', label="Missing Predicted Monthly Mean", marker='o', s=5, zorder=5)


    ax.fill_between(monthly_means.index, 
                    monthly_means["true_monthly_mf"] - monthly_means["true_monthly_std"],
                    monthly_means["true_monthly_mf"] + monthly_means["true_monthly_std"],
                    color='darkgreen', alpha=0.3, label="InTEM Baseline Monthly Std")

    ax.fill_between(monthly_means.index, 
                    monthly_means["pred_monthly_mf"] - monthly_means["pred_monthly_std"],
                    monthly_means["pred_monthly_mf"] + monthly_means["pred_monthly_std"],
                    color='royalblue', alpha=0.3, label="Predicted Baseline Monthly Std")
    
    ax.set_title("Monthly Mean Molefractions for Predicted and True Baselines")
    ax.set_xlabel("Month")
    ax.set_ylabel("Mean Molefraction / ppt")
    

    locator = mdates.MonthLocator()
    ax.xaxis.set_minor_locator(locator)
    ax.tick_params(axis='x', which='minor', length=2)

    if shade_train_and_val_periods:
        if site is not None:
            ax.axvspan(pd.to_datetime(f"{cfg.training_period[site][0]}-01-01"), pd.to_datetime(f"{cfg.training_period[site][1]}-12-31"), alpha=0.1, label="Training Set", color='grey', zorder=0)
            ax.axvspan(pd.to_datetime(f"{cfg.validation_period[site][0]}-01-01"), pd.to_datetime(f"{cfg.validation_period[site][1]}-12-31"), alpha=0.1, label="Validation Set", color='purple', zorder=0)
        else:
            print("Could not shade training and validation periods because site is None. Please provide a site name to shade these periods.")

    if obs_df is not None:
        ax.plot(obs_df.index, obs_df.mf, c="gray", label="AGAGE Observations", zorder=1, lw=0.5, alpha=0.5)
        
    if plot_count_hist:
        ax2 = ax.twinx()
        plot_baseline_count_hist(monthly_means, ax=ax2)
        ax2.set_ylabel("Baseline count")

    ax.legend(loc="upper left")

    plt.show()

def plot_baseline_count_hist(monthly_means, ax=None):
    """
    Plot a histogram of the monthly baseline counts for true and predicted baselines, using the DataFrame generated by the calculate_monthly_means function.
    """
    import datetime
    if ax is None:
        _, axis = plt.subplots(figsize=(12, 4))
    else:
        axis = ax

    counts = monthly_means[["pred_monthly_count", "true_monthly_count"]].copy().fillna(0)

    x = monthly_means.index.date
    width = datetime.timedelta(days=10) 

    axis.bar(x, counts["true_monthly_count"], width=width, label="Count InTEM baselines", color="royalblue")
    axis.bar(x+datetime.timedelta(days=10), counts["pred_monthly_count"], width=width, label="Count predicted baselines", color="darkgreen")
    if ax is None:
        axis.set_xticks(x[::6])
        axis.set_xticklabels([d.strftime("%Y-%m") for d in counts.index][::6], rotation=45, ha="right")
        axis.set_ylabel("Baseline count")
        axis.set_xlabel("Month")
        axis.set_title("Monthly baseline counts")
        axis.legend()
    else:
        axis.set_ylim(0, max(counts.max())*4)
        axis.set_yticks(axis.get_yticks()[axis.get_yticks() <= max(counts.max())])
        axis.legend(loc="center right")



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