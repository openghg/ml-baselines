import numpy as np
import pandas as pd

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.metrics import precision_score, recall_score, f1_score

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


def train_mlp(site,
            balance=0.5,
            balance_method="random",
            undersample=0,
            sample_weights=None,
            time_shift_hours=[6],
            prediction_threshold=0.5,
            mlp_params=None, return_scores=False, verbose=True):
    """ Train a MLP model for a given site.

    Args:
        site (str): The site for which to train the model.
        balance (float): The target ratio of baseline to non-baseline
            values in the training data. Must be between 0 and 1. Only applied to training data.
        balance_method (str): The method to use for balancing the dataset. Must be one of 'random' or 'deterministic'. Only applied to training data.
        undersample (float): If a float between 0 and 1, randomly undersample the training dataset to this fraction. Only applied to training data.
        sample_weights (float or str): If a float, the weight to assign to the baseline class (1s) during training. If "auto", it will be set to 1/class_frequency. 
            If None, no sample weights will be used. Only applied to training data.
        time_shift_hours (list of int): List of time shifts in hours to create lagged features for. For example, [6, 24] will create features shifted by 6 and 24 hours.
        prediction_threshold (float): Decision threshold in the range [0, 1] applied to the
            predicted probabilities (e.g., from ``predict_proba``) to classify examples as positive.
        time_shift_hours (list of int): List of time shifts in hours to create lagged features for. For example, [6, 24] will create features shifted by 6 and 24 hours.
        mlp_params (dict): A dictionary of hyperparameters to pass to the MLPClassifier. If None, default parameters will be used.
    Returns:
        tuple: A tuple containing:
            - model (MLPClassifier): The trained MLP model.
            - X (ndarray): The feature matrix used for training.
            - y (ndarray): The target labels used for training.
            If ``return_scores`` is True, a fourth element is returned:
            - scores (dict): A dictionary of evaluation scores (e.g. precision, recall, F1).
        verbose (bool): Whether to print verbose output.
    """

    # Get the training data
    if verbose: print(f"Training MLP model for site: {site}")
    X, y = get_train_test_data(site, "train", balance=balance, balance_method=balance_method,
                               time_shift_hours=time_shift_hours, undersample=undersample, verbose=verbose)
    
    if sample_weights is not None:
        if verbose: print("Calculating sample weights...")
        weights = generate_sample_weights(y, baseline_weight=sample_weights, non_baseline_weight=1.0, verbose=verbose)

    if mlp_params is None:
        mlp_params = {}
        print(f"Number of training points: {len(y)}")
        print(f"... number of baseline points: {sum(y == 1)} ({sum(y == 1) / len(y):.1%})")

    nn_model = MLPClassifier(**mlp_params, random_state=42)

    # Fit the model
    if verbose: print("... fitting")
    if sample_weights is not None:
        nn_model.fit(X, y, sample_weight=weights)
    else:
        nn_model.fit(X, y)

    # Validation
    X_val, y_val = get_train_test_data(site, "validation",
                                       time_shift_hours=time_shift_hours, verbose=verbose)

    # Testing
    #TODO: Testing on unbalanced data?
    X_test, y_test = get_train_test_data(site, "test",
                                         time_shift_hours=time_shift_hours, verbose=verbose)
                                         

    if verbose: print("... predicting")
    if prediction_threshold != 0.5:
        print(f"Using custom prediction threshold of {prediction_threshold} instead of default 0.5")
        y_pred_val = (nn_model.predict_proba(X_val)[:, 1] >= prediction_threshold).astype(int)
        y_pred_train = (nn_model.predict_proba(X)[:, 1] >= prediction_threshold).astype(int)
        y_pred_test = (nn_model.predict_proba(X_test)[:, 1] >= prediction_threshold).astype(int)
    else:
        y_pred_val = nn_model.predict(X_val)
        y_pred_train = nn_model.predict(X)
        y_pred_test = nn_model.predict(X_test)

    # calculating scores
    precision_val = precision_score(y_val, y_pred_val)
    precision_train = precision_score(y, y_pred_train)
    precision_test = precision_score(y_test, y_pred_test)
    recall_val = recall_score(y_val, y_pred_val)
    recall_train = recall_score(y, y_pred_train)
    recall_test = recall_score(y_test, y_pred_test)

    f1_val = f1_score(y_val, y_pred_val)
    f1_train = f1_score(y, y_pred_train)
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
        return nn_model, X, y, scores
    else:
        return nn_model, X, y


def train_mlp_grid_search(site,
                          param_grid=None,
                          time_shift_hours=[6]):
    """ Train a MLP model using grid search for hyperparameter tuning.

    Args:
        site (str): The site for which to train the model.
        param_grid (dict, optional): A dictionary containing the hyperparameters to tune. If None, default values are used.
        time_shift_hours (list of int): List of time shifts in hours to create lagged features for. For example, [6, 24] will create features shifted by 6 and 24 hours.

    Returns:
        GridSearchCV: The trained model with the best hyperparameters.
    """
    
    if param_grid is None:
        param_grid = {
            'hidden_layer_sizes': [ (50), (100,)],
            'activation': ['relu'],
            'solver': ['adam'],
            'alpha': [0.0001],
            'learning_rate': ['constant'],
            'batch_size': [50],
            'max_iter': [1000],
            'early_stopping': [False],
            'shuffle': [False]
            # 'activation': ['relu'],
            # 'solver': ['adam'],
            # 'alpha': [0.0001, 0.05],
            # 'learning_rate': ['constant','adaptive'],
            # 'batch_size': [100, 200, 300],
            # 'max_iter': [1000, 2000],
            # 'early_stopping': [True, False]
        }

    X_val, y_val = get_train_test_data(site, "validation", time_shift_hours=time_shift_hours)

    best_params = []
    best_scores = []
    best_balances = []

    for balance in np.arange(0.2, 0.9, 0.1):

        #TODO: Could also do undersampling instead of balancing, or both?
        # and could also do different balance methods (random vs deterministic)
        X_train, y_train = get_train_test_data(site, "train", balance=balance,
                                            balance_method="deterministic",
                                            time_shift_hours=time_shift_hours)

        # Combine your training and validation sets
        X_all = pd.concat([X_train, X_val])
        y_all = pd.concat([y_train, y_val])

        # Create a test_fold array: assign -1 for training rows and 0 for validation rows
        test_fold = [-1] * len(X_train) + [0] * len(X_val)
        ps = PredefinedSplit(test_fold=test_fold)

        grid_search = GridSearchCV(
            MLPClassifier(random_state=42),
            param_grid,
            scoring="f1",
            cv=ps,
            refit=False, 
            verbose=2,
            n_jobs=-1
        )

        print(f"Training MLP model for site: {site} with grid search...")
        grid_search.fit(X_all, y_all)

        print("Best parameters found: ", grid_search.best_params_)
        print("Best score: ", grid_search.best_score_)

        best_params.append(grid_search.best_params_)
        best_scores.append(grid_search.best_score_)
        best_balances.append(balance)

    # Find the best combination across all balances
    best_index = best_scores.index(max(best_scores))
    best_best_params = best_params[best_index]
    best_balance = best_balances[best_index]

    print(f"Best balance {best_balance}")
    print(f"Best parameters across all balances: {best_best_params}")

    # Train final model with the best balance and parameters
    X_train_final, y_train_final = get_train_test_data(site, "train", balance=best_balance,
                                                      balance_method="deterministic",
                                                      time_shift_hours=time_shift_hours)
    
    best_model = MLPClassifier(random_state=42, **best_best_params)
    best_model.fit(X_train_final, y_train_final)

    # Evaluate on validation set
    pred_val = best_model.predict(X_val)

    precision_val = precision_score(y_val, pred_val)
    recall_val = recall_score(y_val, pred_val)
    f1_val = f1_score(y_val, pred_val)

    print(f"Validation Precision = {precision_val:.3f}")
    print(f"Validation Recall = {recall_val:.3f}")
    print(f"Validation F1 Score = {f1_val:.3f}")

    # # Save model
    # met_str = f"-{cfg.met_type}" if cfg.met_type else ""
    # joblib.dump(best_model, Path(cfg.models_path) / f"best_mlp{met_str}_{site}.joblib")

    return best_model, best_best_params, best_balance