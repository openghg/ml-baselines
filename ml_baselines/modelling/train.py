import itertools

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.preprocessing import StandardScaler

from ml_baselines.data import read_intem
from ml_baselines.config import Config
from ml_baselines.features import open_features

cfg = Config()
site_coords_dict = cfg.site_coords_dict
models_path = cfg.models_path


def get_train_test_data(site, test_train,
                        balance=-1,
                        undersample=0,
                        time_shift_hours=[6],
                        return_dataframe=False,
                        balance_method="random", verbose=True):
    """ Get the training, testing or validation data for a given site.

    Args:
        site (str): The site for which to get the data.
        test_train (str): The type of data to get. Must be one of 'train', 'test', or 'validation'.
        balance (bool): If True, balance the dataset by undersampling the majority class.
            NOTE: This is only applied to training data (ignored for test or validation).
        undersample (float or bool): If a float between 0 and 1, randomly undersample the dataset to this fraction.
            NOTE: This is only applied to training data (ignored for test or validation).
        time_shift_hours (list of int): List of time shifts in hours to create lagged features for. For example, [6, 24] will create features shifted by 6 and 24 hours.
        return_dataframe (bool): If True, return the data as a DataFrame. If False, return the features and target separately.
        balance_method (str): The method to use for balancing the dataset. Must be one of 'random' or 'deterministic'.
        verbose (bool): If True, print verbose output.
    Returns:
        pd.DataFrame or tuple: If return_dataframe is True, returns a DataFrame with the features and target.
                                If return_dataframe is False, returns a tuple (X, y) where X is the features and y is the target.

    Raises:
        ValueError: If the test_train argument is not one of 'train', 'test', or 'validation'.
    """

    if test_train == "train":
        start_year = cfg.training_period[site][0]
        end_year = cfg.training_period[site][1]
    elif test_train == "test":
        start_year = cfg.testing_period[site][0]
        end_year = cfg.testing_period[site][1]
    elif test_train == "validation":
        start_year = cfg.validation_period[site][0]
        end_year = cfg.validation_period[site][1]
    else:
        raise ValueError("test_train must be either 'train', 'test', or 'validation'")

    df_features = open_features(site,
                            start_year = start_year,
                            end_year = end_year,
                            time_shift_hours = time_shift_hours)
    df_intem = read_intem(site,
                        start_year = start_year,
                        end_year = end_year)

    # Merge features and intem data on time. The intem flags should be merged with the nearest features in time
    df = df_intem.merge(df_features, how='left', left_index=True, right_index=True)
    
    # Drop any nan values
    df = df.dropna()

    if df.shape[0] == 0:
        raise ValueError(f"No data available for {site} in the specified period.")

    # Check if the data is continuous
    if not df.index.is_monotonic_increasing:
        raise ValueError(f"Data for {site} is not continuous in the specified period.")
    if not df.index.is_unique:
        raise ValueError(f"Data for {site} has duplicate timestamps in the specified period.")

    # Balance the dataset if required FOR TRAINING DATA ONLY
    if balance > 0. and test_train == "train":
        if verbose: print(f"... balancing dataset to target baseline ratio of {balance}")
        if balance > 1:
            raise ValueError("Balance must be between 0 and 1.")
        # If balance is True, balance the dataset
        df = balance_dataset(df, target_baseline_ratio=balance, method=balance_method)

    # Undersample the dataset if required FOR TRAINING DATA ONLY
    if undersample and test_train == "train":
        if verbose: print(f"... undersampling dataset to fraction {undersample}")
        if not (0. < undersample <= 1.):
            raise ValueError("Undersample must be a float between 0 and 1.")
        # Randomly undersample the dataset
        # First shuffle the DataFrame
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        # Then, just keep the first "undersample" rows
        df = df.iloc[:int(undersample*len(df))]

    if return_dataframe:
        return df
    else:
        # Split the data into features and target
        X = df.drop(columns=["baseline"])
        y = df["baseline"]
        return X, y


def balance_dataset(df, target_baseline_ratio=0.5, method="random"):
    """ Balance the dataset by undersampling the majority class (baseline or non-baseline) to achieve a target ratio.
    
    Args:
        df (pd.DataFrame): The input DataFrame containing a 'baseline' column with values 0 or 1.
        target_baseline_ratio (float): The desired ratio of baseline (1) to non-baseline (0) values in the output DataFrame.
        
    Returns:
        pd.DataFrame: A balanced DataFrame with the specified ratio of baseline to non-baseline values.
    
    Raises:
        ValueError: If the input DataFrame does not contain a 'baseline' column or if the target ratio is not between 0 and 1.
    """

    def undersample(df,
                    majority_indices_to_subsample,
                    minority_indices,
                    target_majority_ratio,
                    majority_count,
                    minority_count,
                    method="random"):
        undersample_ratio = target_majority_ratio * minority_count / \
                            (majority_count * (1 - target_majority_ratio))

        if method == "random":
            # Randomly sample the majority values
            sampled_majority_indices = np.random.choice(majority_indices_to_subsample,
                                                        size=int(undersample_ratio * majority_count),
                                                        replace=False)
        elif method == "deterministic":
            # Deterministically sample the majority values (sample evenly)
            desired_count = int(np.round(undersample_ratio * majority_count))
            if desired_count < 1:
                desired_count = 1
                print("Warning: Desired count for majority class is less than 1. Setting to 1.")
            indices = np.linspace(0, len(majority_indices_to_subsample) - 1, desired_count).astype(int)
            indices = np.unique(indices)  # Ensure unique indices
            sampled_majority_indices = majority_indices_to_subsample[indices]
        else:
            raise ValueError(f"Unknown undersampling method: {method}, must be 'random' or 'deterministic'.")

        # Add the minority values
        return pd.concat([df.loc[sampled_majority_indices], df.loc[minority_indices]])

    if 'baseline' not in df.columns:
        raise ValueError("Input DataFrame must contain a 'baseline' column.")

    if not (0 <= target_baseline_ratio <= 1):
        raise ValueError("Target baseline ratio must be between 0 and 1.")

    # counting number of baseline&non-baseline data points
    baseline_count = (df['baseline']==1).sum()
    non_baseline_count = (df['baseline']==0).sum()

    baseline_ratio = baseline_count / (baseline_count + non_baseline_count)

    # If there are too many baseline values, we need to undersample them
    if baseline_ratio > target_baseline_ratio:

        df_balanced = undersample(df,
                                  df[df['baseline'] == 1].index,
                                  df[df['baseline'] == 0].index,
                                  target_baseline_ratio,
                                  baseline_count,
                                  non_baseline_count,
                                  method=method)

    else:
        # If there are too many non-baseline values, we need to undersample them
        df_balanced = undersample(df,
                                  df[df['baseline'] == 0].index,
                                  df[df['baseline'] == 1].index,
                                  1 - target_baseline_ratio,
                                  non_baseline_count,
                                  baseline_count,
                                  method=method)

    # Sort the DataFrame
    df_balanced = df_balanced.sort_index()

    return df_balanced


def generate_sample_weights(y, baseline_weight=1.0, non_baseline_weight=1.0, verbose=True):
    """ 
    Generate a Series of sample weights for the baseline dataset, using the specified weights.  
    
    Samples with a higher weight will have more influence on the model during training. This can be used to address class imbalance by giving more weight to the minority class (baseline). If baseline_weight == "auto", it will be set to 1/class_frequency, and non_baseline_weight will be set to 1.0. If baseline_weight is a float, it will be used directly, and non_baseline_weight will be used for the other class.

    Args:
        y (pd.Series): The target variable containing 0s and 1s, where 1 indicates the baseline class and 0 indicates the non-baseline class.
        baseline_weight (float or str): The weight to assign to the baseline class (1s). If "auto", it will be set to 1/class_frequency. Default is 1.0.
        non_baseline_weight (float): The weight to assign to the non-baseline class (0s). Default is 1.0.

    Returns:
        pd.Series: A Series of sample weights corresponding to each entry in y, with the same index. 
    """
    weights = pd.Series(np.ones_like(y, dtype=float), index=y.index)
    if baseline_weight == "auto":
        if np.sum(y == 1) == 0:
            raise ValueError("Cannot set baseline_weight to 'auto' because there are no baseline samples (1s) in y!")
        
        baseline_weight = 1.0 / (y == 1).mean()
        non_baseline_weight = 1.0
        if verbose:
            print("Assigning baseline sample weights automatically based on class frequency: {:.3f}".format(baseline_weight))

    else:
        if not isinstance(baseline_weight, (int, float)):
            raise ValueError("baseline_weight must be a float or 'auto'")
        elif not isinstance(non_baseline_weight, (int, float)):
            raise ValueError("non_baseline_weight must be a float")
        elif baseline_weight < 0 or non_baseline_weight < 0:
            raise ValueError("baseline_weight and non_baseline_weight must be non-negative")
        
        if verbose:
            print("Assigning baseline sample weight: {:.3f} and non-baseline sample weight: {:.3f}".format(baseline_weight, non_baseline_weight))

    weights[y == 1] = float(baseline_weight)
    weights[y == 0] = float(non_baseline_weight)
    
    return weights


class InputPerVariableScaler:
    def __init__(self, aux_variables=["hour_of_day", "day_of_year"]):
        """
        Scale the input features separately for each variable (e.g. u10, v10, u850, etc.) using StandardScaler.
        This normalises each variable independently while still keeping the different time-shifted features of the same variable on the same scale.
        The aux_variables argument specifies any additional variables that should be treated as separate groups and scaled independently (e.g. hour_of_day and day_of_year).
        """

        self.scalers = {}
        self.aux_variables = aux_variables
    
    def fit(self, X, feature_names=None):
        if isinstance(X, np.ndarray) and feature_names is None:
            raise ValueError("If X is a numpy array, feature_names must be provided as a list of column names.")
        elif isinstance(X, np.ndarray) and feature_names is not None:
            X = pd.DataFrame(X, columns=feature_names)
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input X must be a pandas DataFrame with relevant column names.")
        
        col_names = np.unique([col.split("_")[0] if col not in self.aux_variables else col for col in X.columns])
        col_names = sorted(col_names, key=lambda x: int(''.join(filter(str.isdigit, x))) if any(c.isdigit() for c in x) else float('inf'))
        col_names = [col for col in col_names if col not in self.aux_variables] + [col for col in self.aux_variables if col in col_names]
        self.col_names = col_names

        column_groups = { }
        for col in col_names:
            if col in self.aux_variables:
                column_groups[col] = [col]
            else:
                column_groups[col] = [c for c in X.columns if c.startswith(col + "_")]

        for group, columns in column_groups.items():
            scaler = StandardScaler() # Use same scaler on all three sets

            train_data = X[columns] if len(columns) > 1 else X[columns].values.reshape(-1, 1)
            scaler.fit(train_data)
            self.scalers[group] = scaler
        

    def transform(self, X, feature_names=None):
        if isinstance(X, np.ndarray) and feature_names is None:
            raise ValueError("If X is a numpy array, feature_names must be provided as a list of column names.")
        elif isinstance(X, np.ndarray) and feature_names is not None:
            X = pd.DataFrame(X, columns=feature_names)
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input X must be a pandas DataFrame with relevant column names.")
        
        # check that the model has been fitted
        if self.scalers == {}:
            raise ValueError("The scaler has not been fitted yet. Please call fit() before transform().")

        transformed_groups = []
        for group in self.col_names:
            columns = [col for col in X.columns if col.startswith(group + "_")] if group not in self.aux_variables else [group]
            scaler = self.scalers[group]

            data = X[columns] if len(columns) > 1 else X[columns].values.reshape(-1, 1)
            transformed_data = pd.DataFrame(scaler.transform(data), columns=columns, index=X.index)
            transformed_groups.append(transformed_data)

        X_transformed = pd.concat(transformed_groups, axis=1)
        return X_transformed
    
    def fit_transform(self, X, feature_names=None):
        self.fit(X, feature_names=feature_names)
        return self.transform(X, feature_names=feature_names)


def train_baseline_model(site, model_type="mlp",
            balance=0.5,
            balance_method="random",
            undersample=0,
            sample_weights=None,
            normalise_inputs=False,
            time_shift_hours=[6], prediction_threshold=0.5,
            model_params=None, return_scores=False, verbose=True):
    """ Train a model to classify baseline events for a given site.

    Args:
        site (str): The site for which to train the model.
        model_type (str): The type of model to train. Currently accepts "mlp", "random_forest", or "gradient_boosting". Note that the model_params need to be appropriate for the chosen model type  
        balance (float): The target ratio of baseline to non-baseline values in the training data. Must be between 0 and 1. Only applied to training data.
        balance_method (str): The method to use for balancing the dataset. Must be one of 'random' or 'deterministic'. Only applied to training data.
        undersample (float): If a float between 0 and 1, randomly undersample the training dataset to this fraction. Only applied to training data.
        sample_weights (float or str "auto"): If a float, the weight to assign to the baseline class (1s) during training, where non-baseline instances receive a weight of 1.0. If "auto", it will be set to 1/class_frequency. If None, no sample weights will be used.
        normalise_inputs (bool): Whether to normalise the input features based on category using InputPerVariableScaler.
        time_shift_hours (list of int): List of time shifts in hours to create lagged features for. For example, [6, 24] will create features shifted by 6 and 24 hours.
        model_params (dict): A dictionary of hyperparameters to pass to the model. If None, default parameters will be used.
        return_scores (bool): Whether to return the evaluation scores as a dictionary.
        verbose (bool): Whether to print verbose output.
    Returns:
        model: The trained model.
        X_train (pd.DataFrame): The feature matrix used for training.
        y_train (pd.DataFrame): The target labels used for training.
        scaler (InputPerVariableScaler or None): If  ``normalise_inputs`` is true, returns fitted scaler used to normalise input features to be applied to new data. Otherwise, returns None.

        If ``return_scores`` is True, a fourth element is returned:
        scores (dict): A dictionary of evaluation scores (e.g. precision, recall, F1).
    """

    # Get the training data
    if verbose: print(f"Training {model_type.upper() if model_type == 'mlp' else model_type} model for site: {site}")
    X_train, y_train = get_train_test_data(site, "train", balance=balance, balance_method=balance_method,
                               time_shift_hours=time_shift_hours, undersample=undersample, verbose=verbose)
    
    if sample_weights is not None:
        if verbose: print("... calculating sample weights")
        weights = generate_sample_weights(y_train, baseline_weight=sample_weights, non_baseline_weight=1.0, verbose=verbose)

    if verbose:
        print(f"Number of training points: {len(y_train)}")
        print(f"... number of baseline points: {sum(y_train == 1)} ({sum(y_train == 1) / len(y_train):.1%})")

    # If non-specified, use default hyperparameters
    if model_params is None:
        model_params = {}

    # Create model object
    valid_model_types = ["mlp", "random_forest", "gradient_boosting"]
    if model_type not in valid_model_types:
        raise ValueError(f"Unknown model type: {model_type}! must be one of {valid_model_types}.")
    if model_type == "mlp":
        model = MLPClassifier(**model_params, random_state=42)
    elif model_type == "random_forest":
        model = RandomForestClassifier(**model_params, random_state=42)
    elif model_type == "gradient_boosting":
        model = GradientBoostingClassifier(**model_params, random_state=42)
    
    # Get the validation and testing data
    X_val, y_val = get_train_test_data(site, "validation",
                                       time_shift_hours=time_shift_hours, verbose=verbose)
    #TODO: Testing on unbalanced data?
    X_test, y_test = get_train_test_data(site, "test",
                                         time_shift_hours=time_shift_hours, verbose=verbose)

    if normalise_inputs:
        if verbose: print("... normalising inputs")
        scaler = InputPerVariableScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)
    else:
        scaler = None

    # Fit the model
    if verbose: print("... fitting")
    if sample_weights is not None:
        model.fit(X_train, y_train, sample_weight=weights)
    else:
        model.fit(X_train, y_train)

    # Make predictions
    if verbose: print("... predicting")
    if prediction_threshold != 0.5:
        if verbose: print(f"Using custom prediction threshold of {prediction_threshold} instead of default 0.5")
    y_pred_val = (model.predict_proba(X_val)[:, 1] >= prediction_threshold).astype(int)
    y_pred_train = (model.predict_proba(X_train)[:, 1] >= prediction_threshold).astype(int)
    y_pred_test = (model.predict_proba(X_test)[:, 1] >= prediction_threshold).astype(int)

    # Calculate scores
    precision_val = precision_score(y_val, y_pred_val)
    precision_train = precision_score(y_train, y_pred_train)
    precision_test = precision_score(y_test, y_pred_test)
    recall_val = recall_score(y_val, y_pred_val)
    recall_train = recall_score(y_train, y_pred_train)
    recall_test = recall_score(y_test, y_pred_test)

    f1_val = f1_score(y_val, y_pred_val)
    f1_train = f1_score(y_train, y_pred_train)
    f1_test = f1_score(y_test, y_pred_test)

    if verbose:
        print(f"Precision on Training Set = {precision_train:.3f}")
        print(f"Precision on Validation Set = {precision_val:.3f}")
        print(f"Precision on Test Set = {precision_test:.3f}")
        print(f"Recall on Training Set = {recall_train:.3f}")
        print(f"Recall on Validation Set = {recall_val:.3f}")
        print(f"Recall on Test Set = {recall_test:.3f}")
        print(f"F1 Score on Training Set = {f1_train:.3f}")
        print(f"F1 Score on Validation Set = {f1_val:.3f}")
        print(f"F1 Score on Test Set = {f1_test:.3f}")

    if return_scores: 
        scores = {
            "precision_train": precision_train,
            "precision_val": precision_val,
            "precision_test": precision_test,
            "recall_train": recall_train,
            "recall_val": recall_val,
            "recall_test": recall_test,
            "f1_train": f1_train,
            "f1_val": f1_val,
            "f1_test": f1_test
        }
        return model, X_train, y_train, scores, scaler
    else:
        return model, X_train, y_train, scaler


def train_baseline_model_grid_search(site,
                          model_type="mlp",
                          scoring="f1",
                          param_grid=None,
                          data_kwargs=None,
                          validation_keys=None):
    """ Train a model to classify baseline events using grid search for hyperparameter tuning.

    The grid search explores both the model hyperparameters in ``param_grid`` and
    all combinations of data-loading options supplied via ``data_kwargs``.

    Args:
        site (str): The site for which to train the model.
        model_type (str): The type of model to train. Currently accepts "mlp", 
            "random_forest", or "gradient_boosting". Note that the param_grid
            needs to be appropriate for the chosen model type. 
        scoring (str): The metric being optimised for.
        param_grid (dict, optional): A dictionary containing model
            hyperparameters to tune. If None, default values are used.
        data_kwargs (dict, optional): A dictionary where each key is a keyword
            argument accepted by :func:`get_train_test_data` and each value is a
            list of options to explore. All combinations of options are tried.
            For example::

                {
                    "balance": [-1, 0.5],
                    "time_shift_hours": [[6], [6, 24]],
                }

            Valid keys are ``balance``, ``undersample``, ``time_shift_hours``,
            and ``balance_method``. If None, defaults to
            ``{"balance": [-1], "time_shift_hours": [[6]]}``.
        validation_keys (list of str, optional): Keys from ``data_kwargs`` that
            should also be forwarded when loading the validation set. Only
            keywords that affect the feature representation (e.g.
            ``"time_shift_hours"``) should be included here; training-only
            options such as ``"balance"`` or ``"undersample"`` should be
            omitted. If None, defaults to ``["time_shift_hours"]``.

    Returns:
        tuple: ``(best_model, best_params, best_data_kwargs)`` — the fitted
            model, the winning hyperparameter dict, and the winning data-kwargs dict.
    """

    valid_model_types = ["mlp", "random_forest", "gradient_boosting"]
    if model_type not in valid_model_types:
        raise ValueError(f"Unknown model type: {model_type}! must be one of {valid_model_types}.")
    if model_type == "mlp":
        model = MLPClassifier(random_state=42)
    elif model_type == "random_forest":
        model = RandomForestClassifier(random_state=42)
    elif model_type == "gradient_boosting":
        model = GradientBoostingClassifier(random_state=42)

    if param_grid is None:
        if model_type == "mlp":
            param_grid = {
                'hidden_layer_sizes': [(50,),],
                'activation': ['relu'],
                'solver': ['adam'],
                'alpha': [0.0001],
                'batch_size': [5, 10],
                'max_iter': [1000],
                'early_stopping': [True],
                'shuffle': [False]
            }
        if model_type == "random_forest":
            param_grid = {
                'n_estimators': [100, 50, 200,],
                'criterion': ['gini', 'entropy',],
                'max_depth': [None, 5, 10,],
                'min_samples_split': [2, 5, 10,],
                'max_features': ['log2', 'sqrt', None],
                'bootstrap': [True, False],
            }
        if model_type == "gradient_boosting":
            param_grid = {
                'loss': ['log_loss', 'exponential'],
                'learning_rate': [0.1, 0.2, 0.5,],
                'n_estimators': [100, 50, 200],
                'criterion': ['friedman_mse', 'squared_error'],
                'min_samples_split': [2, 5, 10,],
                'max_depth': [3, 5, 8, 10,],
                'max_features': ['sqrt', 'log2'],  
            }

    if data_kwargs is None:
        data_kwargs = {"balance": [-1],
                       "time_shift_hours": [[6]]}

    if validation_keys is None:
        validation_keys = ["time_shift_hours"]

    # Build the cartesian product of all data-kwarg options
    keys = list(data_kwargs.keys())
    combos = list(itertools.product(*[data_kwargs[k] for k in keys]))

    best_params_list = []
    best_scores_list = []
    best_data_kwargs_list = []

    for combo in combos:
        combo_kw = dict(zip(keys, combo))
        print(f"... data kwargs: {combo_kw}")

        val_kw = {k: combo_kw[k] for k in validation_keys if k in combo_kw}
        X_train, y_train = get_train_test_data(site, "train", **combo_kw)
        X_val, y_val = get_train_test_data(site, "validation", **val_kw)

        assert set(X_train.columns) == set(X_val.columns), "Feature columns in training and validation sets do not match. Check that the data kwargs affecting features are included in validation_keys."

        # Combine training and validation sets; use PredefinedSplit so validation
        # rows are never used for fitting during cross-validation
        X_all = pd.concat([X_train, X_val])
        y_all = pd.concat([y_train, y_val])
        test_fold = [-1] * len(X_train) + [0] * len(X_val)
        ps = PredefinedSplit(test_fold=test_fold)

        grid_search = GridSearchCV(
            model,
            param_grid,
            scoring=scoring,
            cv=ps,
            refit=False,
            verbose=2,
            n_jobs=-1
        )

        print(f"Training {model_type.upper() if model_type == 'mlp' else model_type} model for site: {site} with grid search...")
        grid_search.fit(X_all, y_all)

        print("Best parameters found: ", grid_search.best_params_)
        print("Best score: ", grid_search.best_score_)

        best_params_list.append(grid_search.best_params_)
        best_scores_list.append(grid_search.best_score_)
        best_data_kwargs_list.append(combo_kw)

    # Find the best combination across all data-kwarg combos
    best_index = best_scores_list.index(max(best_scores_list))
    best_best_params = best_params_list[best_index]
    best_combo_kw = best_data_kwargs_list[best_index]

    print(f"\nBest data kwargs: {best_combo_kw}")
    print(f"Best {model_type.upper() if model_type == 'mlp' else model_type} parameters: {best_best_params}")

    # Train final model with the winning combination
    best_val_kw = {k: best_combo_kw[k] for k in validation_keys if k in best_combo_kw}
    X_train_final, y_train_final = get_train_test_data(site, "train", **best_combo_kw)
    X_val_final, y_val_final = get_train_test_data(site, "validation", **best_val_kw)

    best_model = model.__class__(random_state=42, **best_best_params)
    best_model.fit(X_train_final, y_train_final)

    # Evaluate on validation set
    pred_val = best_model.predict(X_val_final)

    precision_val = precision_score(y_val_final, pred_val)
    recall_val = recall_score(y_val_final, pred_val)
    f1_val = f1_score(y_val_final, pred_val)

    print("Validation metrics of winning combination:")
    print(f"    Precision = {precision_val:.3f}")
    print(f"    Recall = {recall_val:.3f}")
    print(f"    F1 Score = {f1_val:.3f}")

    return best_model, best_best_params, best_combo_kw
