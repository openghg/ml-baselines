import numpy as np
import xarray as xr
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score, recall_score, f1_score

from ml_baselines.data import read_intem
from ml_baselines.config import Config
from ml_baselines.features import open_features

cfg = Config()
site_coords_dict = cfg.site_coords_dict
models_path = cfg.models_path


def get_train_test_data(site, test_train):

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
                            end_year = end_year)
    df_intem = read_intem(site,
                        start_year = start_year,
                        end_year = end_year)

    # Merge features and intem data on time. The intem flags should be merged with the nearest features in time
    df = df_intem.merge(df_features, how='left', left_index=True, right_index=True)
    
    # Drop the nan values
    df = df.dropna()

    if df.shape[0] == 0:
        raise ValueError(f"No data available for {site} in the specified period.")

    # Check if the data is continuous
    if not df.index.is_monotonic_increasing:
        raise ValueError(f"Data for {site} is not continuous in the specified period.")
    if not df.index.is_unique:
        raise ValueError(f"Data for {site} has duplicate timestamps in the specified period.")

    return df


def balance_dataset(df, target_baseline_ratio=0.8):

    # counting number of baseline&non-baseline data points
    baseline_count = (df['baseline']==1).sum()
    non_baseline_count = (df['baseline']==0).sum()

    baseline_ratio = baseline_count / (baseline_count + non_baseline_count)
    non_baseline_ratio = non_baseline_count / (baseline_count + non_baseline_count)

    if baseline_ratio > target_baseline_ratio:
        print("The dataset already has a higher baseline ratio than the target. No balancing needed.")
        return df


    # calculating the majority class count based on majority_ratio and minority_count
    majority_count = int(baseline_count * (non_baseline_ratio/baseline_ratio))

    # subsetting the non-baseline data points
    undersampled_non_baseline = df[df['baseline'] == 0]


    #TODO: GOT TO HERE. Everything below this line is not working.




    # creating an array of time indices & randomly selecting some
    time_indices = undersampled_non_baseline.index
    selected_indices = np.random.choice(time_indices, majority_count, replace=False)
    selected_indices = np.sort(selected_indices)

    # setting the non-baseline data points to only include the randomly selected indices
    undersampled_non_baseline = undersampled_non_baseline.sel(time=selected_indices)

    # combining the the undersampled non-baseline with the baseline values
    balanced_df = xr.merge([df.sel(time=(df['baseline'] == 1)), undersampled_non_baseline])
    balanced_df = balanced_df.sortby('time')

    # checking balance
    new_baseline_count = balanced_df['baseline'].where(balanced_df['baseline']==1).count()
    new_non_baseline_count = balanced_df['baseline'].where(balanced_df['baseline']==0).count()

    # verifying that the ratio of baseline:non-baseline data points is as expected (within a tolerance of 1%)
    tolerance = 0.01
    upper_bound = (1+tolerance)*(majority_ratio/minority_ratio)
    lower_bound = (1-tolerance)*(majority_ratio/minority_ratio)

    if(lower_bound <= (new_non_baseline_count/new_baseline_count) <= upper_bound):
        return balanced_df
    else:
        raise ValueError("The counts of baseline and non-baseline values are not in the expected ratio.")



def train_mlp(site):

    df = get_train_test_data(site, "train")

    # Split the data into features and target
    X = df.drop(columns=["baseline"])
    y = df["baseline"]

    #TODO: BALANCE DATASET
    df = balance_dataset(df)

    nn_model = MLPClassifier(max_iter=1000, random_state=42,)

    # Hyperparameters from Kirstin's model:
    # nn_model = MLPClassifier(max_iter=1000, random_state=42,
    #                     hidden_layer_sizes=(100,), 
    #                     shuffle=False,
    #                     activation='relu', 
    #                     solver='adam', 
    #                     alpha=0.0001, 
    #                     learning_rate='constant', 
    #                     batch_size=100, 
    #                     early_stopping=False,
    #                     learning_rate_init=0.0001,
    #                     beta_2=0.9,)
    # Fit the model
    nn_model.fit(X, y)

    # Validation
    df_val = get_train_test_data(site, "validation")
    X_val = df_val.drop(columns=["baseline"])
    y_val = df_val["baseline"]

    # Testing
    df_test = get_train_test_data(site, "test")
    X_test = df_test.drop(columns=["baseline"])
    y_test = df_test["baseline"]

    y_pred_val = nn_model.predict(X_val)
    y_pred_train = nn_model.predict(X)

    # calculating scores
    precision_val = precision_score(y_val, y_pred_val)
    precision_train = precision_score(y, y_pred_train)
    recall_val = recall_score(y_val, y_pred_val)
    recall_train = recall_score(y, y_pred_train)
    f1_val = f1_score(y_val, y_pred_val)
    f1_train = f1_score(y, y_pred_train)

    print(f"Precision on Training Set = {precision_train:.3f}")
    print(f"Precision on Testing Set = {precision_val:.3f}")
    print(f"Recall on Training Set = {recall_train:.3f}")
    print(f"Recall on Testing Set = {recall_val:.3f}")
    print(f"F1 Score on Training Set = {f1_train:.3f}")
    print(f"F1 Score on Testing Set = {f1_val:.3f}")

    return nn_model, X_test, y_test

train_mlp("MHD")