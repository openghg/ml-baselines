'''
This script defines plotting functions for visualising model predictions and performance.
These include plotting raw observations with baseline labels, model confidence scores, monthly baseline means, baseline counts and STL decomposition components.
'''

import datetime
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import AutoMinorLocator
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from ml_baselines.config import Config
cfg = Config()

models_path = cfg.models_path

def plot_confusion_matrix(y, y_pred, labels = ["non-baseline", "baseline"], normalise=False,  title=""):
    """
    Plot a confusion matrix for the true labels and predicted labels.

    Args:
        y (pd.Series): True labels.
        y_pred (pd.Series): Predicted labels.
        normalise (bool): If True, the confusion matrix will be normalised to show
            proportions of the total dataset instead of counts.
        title (str): The title of the plot.
        labels (list of str): The labels for the axes. Defaults to ["non-baseline", "baseline"].
    """
    if normalise:
        normalise = "all"
    else:
        normalise = None
        
    cm = confusion_matrix(y, y_pred, labels=[0,1], normalize=normalise)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                display_labels=labels)

    disp.plot(cmap=plt.cm.Blues)

    disp.ax_.set_title(title)
    disp.ax_.set(xlabel='Predicted label', ylabel='InTEM label')

    return disp.figure_, disp.ax_


def plot_obs(labelled_df, labels_from="InTEM",
             shade_train_and_val_periods=True, site=None,
             title="", show_legend=True):
    """
    Plot observed molefractions.

    Args:
        labelled_df (pd.DataFrame): A DataFrame containing the observed molefractions in a
            column named "mf", true baseline labels in a column named
            "baseline", and predicted baseline labels in a column named
            "predicted_baseline". The DataFrame should have a datetime index.
        labels_from (str): The source of the labels to plot (e.g., "InTEM"). Must be one of 'InTEM' or 'model'.
        shade_train_and_val_periods (bool): Whether to shade the training and validation periods.
        site (str): The site name. Used to shade training and validation periods.
        title (str): The title of the plot.
        legend (bool): Whether to show the legend.

    """
    if labels_from == "InTEM":
        true_mask = (labelled_df.baseline == 1)
        false_mask = (labelled_df.baseline == 0)
        colours = ['lightcoral', 'darkgreen']
    elif labels_from == "model":
        true_mask = (labelled_df.predicted_baseline == 1)
        false_mask = (labelled_df.predicted_baseline == 0)
        colours = ['lightcoral', 'darkblue']
    else:
        raise ValueError("Invalid value for 'labels_from'. Please choose 'InTEM' or 'model'.")


    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(labelled_df.index, labelled_df.mf, c="gray", label="AGAGE Observations", zorder=1, lw=0.5, alpha=0.5)
    ax.scatter(labelled_df.index[false_mask], labelled_df.mf[false_mask],
               color=colours[0], label=f"{labels_from.title() if labels_from == 'model' else labels_from} non-baseline",
               zorder=5, s=0.5, alpha=0.5, marker='x')
    ax.scatter(labelled_df.index[true_mask], labelled_df.mf[true_mask],
               color=colours[1], label=f"{labels_from.title() if labels_from == 'model' else labels_from} baseline",
               zorder=5, s=0.5, marker='x', alpha=0.8)

    ax.set_ylabel("Molefraction in air / ppt", fontstyle="italic", fontsize=11)
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(axis='y', which='major', labelsize=10)
    ax.tick_params(axis='y', which='minor', length=2)

    ax.set_xlim(labelled_df.index.min() - pd.DateOffset(months=3),
            labelled_df.index.max() + pd.DateOffset(months=3))
    n_years = labelled_df.index.max().year - labelled_df.index.min().year
    if n_years <= 10:
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[4, 7, 10]))
        ax.tick_params(axis='x', which='minor', length=3)
    else:
        ax.xaxis.set_major_locator(mdates.YearLocator(base=4))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.xaxis.set_minor_locator(mdates.YearLocator())
    ax.tick_params(axis='x', which='major', labelsize=10)
    ax.tick_params(axis='x', which='minor', length=2)

    if shade_train_and_val_periods:
        if site is not None:
            x_min = labelled_df.index.min() - pd.DateOffset(months=3)
            x_max = labelled_df.index.max() + pd.DateOffset(months=3)

            train_start = pd.to_datetime(f"{cfg.training_period[site][0]}-01-01")
            train_end = pd.to_datetime(f"{cfg.training_period[site][1]}-12-31")
            val_start = pd.to_datetime(f"{cfg.validation_period[site][0]}-01-01")
            val_end = pd.to_datetime(f"{cfg.validation_period[site][1]}-12-31")

            if train_start <= x_max and train_end >= x_min:
                ax.axvspan(train_start, train_end, alpha=0.1, label="Training Set", color='grey', zorder=0)
            if val_start <= x_max and val_end >= x_min:
                ax.axvspan(val_start, val_end, alpha=0.1, label="Validation Set", color='purple', zorder=0)
        else:
            print("Could not shade training and validation periods because site is None. Please provide a site name to shade these periods.")

    ax.set_title(title)

    if show_legend:
        ax.legend(markerscale=6,
                  fontsize=11)

    return fig, ax


def plot_obs_with_labels(labelled_df,
                         plot_true_negatives=True,
                         shade_train_and_val_periods=True, site=None,
                         title=""):
    """
    Plot observed molefractions and predicted baseline labels over time.

    Args:
        labelled_df (pd.DataFrame): A DataFrame containing the observed molefractions in a
            column named "mf", true baseline labels in a column named
            "baseline", and predicted baseline labels in a column named
            "predicted_baseline". The DataFrame should have a datetime index.
        plot_true_negatives (bool): Whether to plot true negatives.
        shade_train_and_val_periods (bool): Whether to shade the training and
            validation periods.
        site (str): The site name used to shade training and validation periods.
        title (str): The title of the plot.

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
    ax.scatter(labelled_df.index[false_negatives_mask], labelled_df.mf[false_negatives_mask], color="mediumaquamarine", label="Missed baseline label (False Negatives)", zorder=5, s=0.5, marker='x', alpha=0.8)
    ax.scatter(labelled_df.index[true_positives_mask], labelled_df.mf[true_positives_mask], color="darkgreen", label="Correctly predicted as baseline (True Positives)", zorder=5, s=0.5, marker='x', alpha=0.8)

    ax.set_ylabel("Molefraction in air / ppt", fontstyle="italic", fontsize=11)
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(axis='y', which='major', labelsize=10)
    ax.tick_params(axis='y', which='minor', length=2)

    ax.set_xlim(labelled_df.index.min() - pd.DateOffset(months=3),
            labelled_df.index.max() + pd.DateOffset(months=3))
    n_years = labelled_df.index.max().year - labelled_df.index.min().year
    if n_years <= 10:
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[4, 7, 10]))
        ax.tick_params(axis='x', which='minor', length=3)
    else:
        ax.xaxis.set_major_locator(mdates.YearLocator(base=4))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.xaxis.set_minor_locator(mdates.YearLocator())
    ax.tick_params(axis='x', which='major', labelsize=10)
    ax.tick_params(axis='x', which='minor', length=2)

    if shade_train_and_val_periods:
        if site is not None:
            x_min = labelled_df.index.min() - pd.DateOffset(months=3)
            x_max = labelled_df.index.max() + pd.DateOffset(months=3)

            train_start = pd.to_datetime(f"{cfg.training_period[site][0]}-01-01")
            train_end = pd.to_datetime(f"{cfg.training_period[site][1]}-12-31")
            val_start = pd.to_datetime(f"{cfg.validation_period[site][0]}-01-01")
            val_end = pd.to_datetime(f"{cfg.validation_period[site][1]}-12-31")

            if train_start <= x_max and train_end >= x_min:
                ax.axvspan(train_start, train_end, alpha=0.1, label="Training Set", color='grey', zorder=0)
            if val_start <= x_max and val_end >= x_min:
                ax.axvspan(val_start, val_end, alpha=0.1, label="Validation Set", color='purple', zorder=0)
        else:
            print("Could not shade training and validation periods because site is None. Please provide a site name to shade these periods.")

    ax.set_title(title)

    ax.legend(markerscale=6, fontsize=11)

    return fig, ax


def plot_model_confidence(labelled_df, cmap=None, shade_train_and_val_periods=True, site=None, title=None):
    """
    Plot model confidence for baseline classification.

    Args:
        labelled_df (pd.DataFrame): A DataFrame containing the observed molefractions in a
            column named "mf", true baseline labels in a column named
            "baseline", and predicted baseline labels in a column named
            "predicted_baseline". The DataFrame should have a datetime index.
        cmap (str, matplotlib colormap, or None): The colour map to use.
        shade_train_and_val_periods=True (bool): Whether to shade the training and validation periods on the plot.
        title (str): The title of the plot.

    """
    if "predicted_proba" not in labelled_df.columns:
        raise ValueError("Predicted probabilities are not available in the labelled_df. Please ensure that the predict_baselines function is called with return_proba=True, and that the resulting y_proba is included in the BaselineLabelledObservations object.")

    if cmap is None:
        baseline_cmap = LinearSegmentedColormap.from_list(
            "baseline_prob_cmap",
            ["lightcoral", "darkgreen"]
        )
    elif isinstance(cmap, str):
        baseline_cmap = plt.get_cmap(cmap)
    else:
        baseline_cmap = cmap

    plot_df = labelled_df.sort_values("predicted_proba")

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(labelled_df.index, labelled_df.mf, 
            c="gray", label="AGAGE Observations", 
            zorder=1, lw=0.5, alpha=0.5
        )
    sc = ax.scatter(plot_df.index, plot_df.mf,
            c=plot_df["predicted_proba"],
            cmap=baseline_cmap,
            vmin=0.0,
            vmax=1.0,
            zorder=5,
            s=0.5,
            marker="x",
            alpha=0.9,
            label="Points colored by\npredicted baseline probability"
        )

    cbar = plt.colorbar(sc, ax=ax, pad=0.01, fraction=0.03)
    cbar.set_label("P(baseline)", fontstyle="italic", fontsize=11)
    cbar.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    cbar.ax.tick_params(labelsize=11)
    cbar.ax.yaxis.set_minor_locator(AutoMinorLocator())
    cbar.ax.tick_params(which='minor', length=3)
    cbar.ax.tick_params(which='major', length=5)

    ax.set_ylabel("Molefraction in air / ppt", fontstyle="italic", fontsize=11)
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(axis='y', which='major', labelsize=10)
    ax.tick_params(axis='y', which='minor', length=2)

    ax.set_xlim(labelled_df.index.min() - pd.DateOffset(months=3),
            labelled_df.index.max() + pd.DateOffset(months=3))
    n_years = labelled_df.index.max().year - labelled_df.index.min().year
    if n_years <= 10:
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[4, 7, 10]))
        ax.tick_params(axis='x', which='minor', length=3)
    else:
        ax.xaxis.set_major_locator(mdates.YearLocator(base=4))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.xaxis.set_minor_locator(mdates.YearLocator())
    ax.tick_params(axis='x', which='major', labelsize=10)
    ax.tick_params(axis='x', which='minor', length=2)

    if shade_train_and_val_periods:
        if site is not None:
            x_min = labelled_df.index.min() - pd.DateOffset(months=3)
            x_max = labelled_df.index.max() + pd.DateOffset(months=3)

            train_start = pd.to_datetime(f"{cfg.training_period[site][0]}-01-01")
            train_end = pd.to_datetime(f"{cfg.training_period[site][1]}-12-31")
            val_start = pd.to_datetime(f"{cfg.validation_period[site][0]}-01-01")
            val_end = pd.to_datetime(f"{cfg.validation_period[site][1]}-12-31")

            if train_start <= x_max and train_end >= x_min:
                ax.axvspan(train_start, train_end, alpha=0.1, label="Training Set", color='grey', zorder=0)
            if val_start <= x_max and val_end >= x_min:
                ax.axvspan(val_start, val_end, alpha=0.1, label="Validation Set", color='purple', zorder=0)
        else:
            print("Could not shade training and validation periods because site is None. Please provide a site name to shade these periods.")

    ax.set_title(title if title else "Model confidence over observations")

    ax.legend(loc="best", fontsize=11)

    return fig, ax


def plot_monthly_means(monthly_means, shade_train_and_val_periods=True, site=None,
                       obs_df=None, plot_count_hist=False,
                       show_anomalies=True, show_missing=True, show_legend=True,
                       date_range=None,
                       title="Monthly Mean Molefractions for Predicted and True Baselines"):
    """
    Plot monthly means of the observed molefractions and one standard deviation.

    Args:
        monthly_means (pd.DataFrame): A DataFrame containing the monthly mean molefractions with a time index.
        shade_train_and_val_periods (bool): Whether to shade the training and validation periods on the plot.
        site (str): The site for which to plot observed molefractions. Required if shade_train_and_val_periods is True.
        obs_df (pd.DataFrame): If passed, plots the observed molefractions as a gray line in the background of the plot.
        plot_count_hist (bool): If True, plots a histogram of the monthly baseline counts on a twin axis.
        show_anomalies (bool): Whether to highlight anomaly months.
        date_range (list): ADD DOCSTRING - MENTION INCLUSIVE
        title (str): The title of the plot.
    """

    if date_range is not None:
        # Crop monthly means based on date range
        start_date = pd.Timestamp(str(date_range[0]))
        end_date = pd.Timestamp(str(date_range[1])) + pd.offsets.YearEnd(1)
        monthly_means = monthly_means[(monthly_means.index >= start_date) & (monthly_means.index <= end_date)]

    fig, ax = plt.subplots(figsize=(12,6))

    ax.plot(monthly_means.index, monthly_means["true_monthly_mf"],
            label="InTEM Baseline Monthly Mean", marker='o', color='darkgreen', markersize=3)
    ax.plot(monthly_means.index, monthly_means["pred_monthly_mf"],
            label="Predicted Baseline Monthly Mean", color='royalblue', marker='o', markersize=3, alpha=0.8)

    missing_pred_data_mask = monthly_means["pred_monthly_mf"].isna()
    if show_missing and missing_pred_data_mask.any():
        ax.scatter(monthly_means.index[missing_pred_data_mask], monthly_means["true_monthly_mf"][missing_pred_data_mask], 
                   color='#FBF719', edgecolor='black', linewidth=0.25,
                   label="Missing Predicted Monthly Mean", marker='o', s=25, zorder=5)

    ax.fill_between(monthly_means.index, 
                    monthly_means["true_monthly_mf"] - monthly_means["true_monthly_std"],
                    monthly_means["true_monthly_mf"] + monthly_means["true_monthly_std"],
                    color='darkgreen', alpha=0.3, label="InTEM Baseline Monthly Std")

    ax.fill_between(monthly_means.index, 
                    monthly_means["pred_monthly_mf"] - monthly_means["pred_monthly_std"],
                    monthly_means["pred_monthly_mf"] + monthly_means["pred_monthly_std"],
                    color='royalblue', alpha=0.3, label="Predicted Baseline Monthly Std")

    if obs_df is not None:
        ax.plot(obs_df.index, obs_df.mf, c="gray", label="AGAGE Observations", zorder=1, lw=0.5, alpha=0.5)
        
    if plot_count_hist:
        ax2 = ax.twinx()
        plot_baseline_count_hist(monthly_means, ax=ax2)
        ax2.set_ylabel("Baseline count")

    y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
    offset = 0.01 * y_range
    if show_anomalies and 'is_anomaly' in monthly_means.columns:
        anomaly_colors = {3: 'orange', 5: 'red'}
        for threshold in sorted(np.unique(monthly_means["is_anomaly"])):
            if threshold > 1:
                anomaly_months = monthly_means[monthly_means["is_anomaly"] == threshold]
                color = anomaly_colors.get(threshold, 'red')
                ax.scatter(anomaly_months.index, anomaly_months["true_monthly_mf"] - offset,
                        label=f"Anomaly > {threshold} std",
                        color=color, edgecolor='black', linewidth=0.25,
                        marker='^',
                        s=50, zorder=5)

    ax.set_ylabel("Mean Molefraction / ppt", fontstyle="italic", fontsize=11)
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(axis='y', which='major', labelsize=10)
    ax.tick_params(axis='y', which='minor', length=2)

    ax.set_xlim(monthly_means.index.min() - pd.DateOffset(months=3),
            monthly_means.index.max() + pd.DateOffset(months=3))
    n_years = monthly_means.index.max().year - monthly_means.index.min().year
    if n_years <= 10:
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[4, 7, 10]))
        ax.tick_params(axis='x', which='minor', length=3)
    else:
        ax.xaxis.set_major_locator(mdates.YearLocator(base=4))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.xaxis.set_minor_locator(mdates.YearLocator())
    ax.tick_params(axis='x', which='major', labelsize=10)
    ax.tick_params(axis='x', which='minor', length=2)

    if shade_train_and_val_periods:
        if site is not None:
            x_min = monthly_means.index.min() - pd.DateOffset(months=3)
            x_max = monthly_means.index.max() + pd.DateOffset(months=3)

            train_start = pd.to_datetime(f"{cfg.training_period[site][0]}-01-01")
            train_end = pd.to_datetime(f"{cfg.training_period[site][1]}-12-31")
            val_start = pd.to_datetime(f"{cfg.validation_period[site][0]}-01-01")
            val_end = pd.to_datetime(f"{cfg.validation_period[site][1]}-12-31")

            if train_start <= x_max and train_end >= x_min:
                ax.axvspan(train_start, train_end, alpha=0.1, label="Training Set", color='grey', zorder=0)
            if val_start <= x_max and val_end >= x_min:
                ax.axvspan(val_start, val_end, alpha=0.1, label="Validation Set", color='purple', zorder=0)
        else:
            print("Could not shade training and validation periods because site is None. Please provide a site name to shade these periods.")

    ax.set_title(title)

    if show_legend:
        ax.legend(loc="best" if not plot_count_hist else "upper left",
                  fontsize=11, ncols=2)

    return fig, ax


def plot_baseline_count_hist(monthly_means, ax=None):
    """
    Plot a histogram of the monthly baseline counts for true and predicted baselines.

    Args:
        monthly_means (pd.DataFrame): A DataFrame generated by the calculate_monthly_means function.
        ax (matplotlib.axes.Axes): Optional matplotlib axis to draw on.

    """
    if ax is None:
        fig, axis = plt.subplots(figsize=(12, 4))
    else:
        axis = ax
        fig = ax.get_figure()

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
        
    return fig, axis


def plot_stl_components(true_baselines, orig_monthly_means, monthly_means_filled, stl_result):
    """
    Plots the components of the STL decomposition.
    Computed when removing seasonality during the true baseline CV calculation.

    Args:
        true_baselines (pd.Series): The raw baseline observations with a datetime index.
        orig_monthly_means (pd.Series): The monthly means of the baseline observations.
        monthly_means_filled (pd.Series): The interpolated monthly means passed to STL.
        stl_result: The fitted STL result object.
        species (str): The species name. Used in the title if provided.
        site (str): The site name. Used in the title if provided.
        title (bool): Whether to add a suptitle. Default is True.

    """

    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

    # plot 1 - raw observations (scattered) and monthly means with any interpolated regions highlights
    interpolated_mask = ~monthly_means_filled.index.isin(orig_monthly_means.index)
    axes[0].scatter(true_baselines.index, true_baselines.values,
                    label='Raw observations', color='lightblue', s=2, zorder=1)
    axes[0].plot(orig_monthly_means.index, orig_monthly_means.values,
                    label='Raw monthly mean', color='blue', linewidth=1)
    if interpolated_mask.any():
        axes[0].scatter(monthly_means_filled.index[interpolated_mask], monthly_means_filled.values[interpolated_mask],
                        label='Interpolation for STL fit', color='red', zorder=5, s=15)
    axes[0].legend()
    axes[0].set_title("Observations")

    # plot 2 - overall trend
    axes[1].plot(monthly_means_filled.index, stl_result.trend, color='red')
    axes[1].set_title("Trend")

    # plot 3 - seasonal trend
    axes[2].plot(monthly_means_filled.index, stl_result.seasonal, color='purple')
    axes[2].axhline(0, color='black', linestyle='--', linewidth=0.8)
    axes[2].set_title("Seasonal")

    # plot 4 - residuals
    axes[3].plot(monthly_means_filled.index, stl_result.resid, color='green')
    axes[3].axhline(0, color='black', linestyle='--', linewidth=0.8)
    axes[3].set_title("Residuals")

    for ax in axes:
        axes[0].spines[['top', 'right']].set_visible(False)

    fig.supylabel("Mole fraction in air / ppt")
    fig.tight_layout()

    return fig, axes