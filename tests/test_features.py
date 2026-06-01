import numpy as np
import os
import pandas as pd
from pathlib import Path
from tempfile import TemporaryDirectory
import xarray as xr

from ml_baselines.features import open_features, preprocess_features_arco_era5
from ml_baselines.config import Config

cfg = Config()

surface_variable_mapping = {
    "surface_pressure": "sp",
    "boundary_layer_height": "blh",
    "10m_u_component_of_wind": "u10",
    "10m_v_component_of_wind": "v10"
}

level_variable_mapping = {
    "u_component_of_wind": "u",
    "v_component_of_wind": "v",
}

test_dir = os.path.dirname(os.path.abspath(__file__))
test_data_dir = os.path.join(test_dir, "data")

def test_preprocess_features_arco_era5():

    site = "MHD"
    year = 2020

    with TemporaryDirectory() as temp_dir:
        preprocess_features_arco_era5("MHD",
                                      input_dir=test_data_dir,
                                      output_dir=temp_dir)

        # Check that the DataFrame has been created and has expected columns
        df = pd.read_csv(Path(temp_dir) / f"features-arco-era5_{site}_{year}.csv.gz",
                         compression='gzip',
                         comment='#')


    ds = xr.open_dataset(Path(cfg.root_dir) / f"{test_data_dir}/era5test-{site}-{year}.nc")

    # Test that time index matches
    ds_time = pd.to_datetime(ds['time'].values)
    df_time = pd.to_datetime(df['time'].values)
    assert all(ds_time == df_time), "Time indices do not match between original and processed data."

    # Test that surface variables have been correctly renamed and included
    for var in surface_variable_mapping.keys():
        for point in range(ds.sizes['points']):
            var_ds = ds[var].sel(dict(points=point)).values
            var_df = df[f"{surface_variable_mapping[var]}_{point}"].values
            assert np.allclose(var_ds, var_df), f"Variable '{var}' at point {point} does not match between original and processed data."

    # Test that level variables have been correctly renamed and included
    for var in level_variable_mapping.keys():
        for level in ds.levels.values.tolist():
            for point in range(ds.sizes['points']):
                var_ds = ds[var].sel(dict(levels=level, points=point)).values
                var_df = df[f"{level_variable_mapping[var]}{level}_{point}"].values
                assert np.allclose(var_ds, var_df), f"Variable '{var}' at level {level} and point {point} does not match between original and processed data."


def test_open_features():

    with TemporaryDirectory() as temp_dir:
        preprocess_features_arco_era5("MHD",
                                    input_dir=test_data_dir,
                                    output_dir=temp_dir)

        df = open_features("MHD",
                           start_year=2020,
                           end_year=2020,
                           features_dir=temp_dir)
    
        df_6_24 = open_features("MHD",
                                start_year=2020,
                                end_year=2020,
                                time_shift_hours=[6, 24],
                                features_dir=temp_dir)

    columns = []
    for key in cfg.met_variables.keys():
        for i in range(17):
            columns.append(f"{key}_{i}")

    # Check that the hour_of_day column is correctly computed
    hour_of_day = df.index.hour.values
    assert np.all(df["hour_of_day"].values == hour_of_day), "hour_of_day column is not correctly computed."

    ### TEST LAGS
    columns_6h = []
    columns_24h = []
    for column in columns:
        columns_6h.append(f"{column}_6h")
        columns_24h.append(f"{column}_24h")

    ### Firstly, just test the 6-hour lagged features

    expected_columns = columns + columns_6h

    # Check that all expected met. columns are present
    assert all([col in df.columns for col in expected_columns]), "Not all expected columns are present in the opened features DataFrame."

    # Check that all _6h columns are indeed 6-hours lagged behind their original columns
    now_columns = [col for col in expected_columns if not col.endswith("_6h")]
    lagged_columns = [col for col in expected_columns if col.endswith("_6h")]

    future_rows = df[df.index >= df.index[0] + pd.Timedelta(hours=6)]

    assert np.all(df[now_columns].iloc[:len(future_rows)].values == future_rows[lagged_columns].values), "Lagged columns do not match the expected 6-hour lag."

    # Check that the hour_of_day column is correctly computed
    hour_of_day = df.index.hour.values
    assert np.all(df["hour_of_day"].values == hour_of_day), "hour_of_day column is not correctly computed."

    ### Now test the 6 and 24-hour lagged features

    expected_columns = columns + columns_6h + columns_24h

    # Check that all expected met. columns are present
    assert all([col in df_6_24.columns for col in expected_columns]), "Not all expected columns are present in the opened features DataFrame."

    # Check that all _6h and _24h columns are indeed 6- and 24-hours lagged behind their original columns
    now_columns = [col for col in expected_columns if not col.endswith("_6h") and not col.endswith("_24h")]
    lagged_columns_6h = [col for col in expected_columns if col.endswith("_6h")]
    lagged_columns_24h = [col for col in expected_columns if col.endswith("_24h")]

    future_rows_6h = df_6_24[df_6_24.index >= df_6_24.index[0] + pd.Timedelta(hours=6)]
    future_rows_24h = df_6_24[df_6_24.index >= df_6_24.index[0] + pd.Timedelta(hours=24)]

    assert np.all(df_6_24[now_columns].iloc[:len(future_rows_6h)].values == future_rows_6h[lagged_columns_6h].values), "Lagged columns do not match the expected 6-hour lag."
    assert np.all(df_6_24[now_columns].iloc[:len(future_rows_24h)].values == future_rows_24h[lagged_columns_24h].values), "Lagged columns do not match the expected 24-hour lag."
