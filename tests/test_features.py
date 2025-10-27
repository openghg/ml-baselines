import pandas as pd
from pathlib import Path
import xarray as xr
import numpy as np
from tempfile import TemporaryDirectory

from ml_baselines.features import preprocess_features_arco_era5
from ml_baselines.config import Config

cfg = Config()


def test_preprocess_features_arco_era5():

    site = "MHD"
    year = 2020

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

    with TemporaryDirectory() as temp_dir:
        preprocess_features_arco_era5("MHD",
                                    input_dir="tests/data",
                                    output_dir=temp_dir)

        # Check that the DataFrame has been created and has expected columns
        df = pd.read_csv(Path(temp_dir) / f"features-arco-era5_{site}_{year}.csv.gz",
                         compression='gzip',
                         comment='#')


    ds = xr.open_dataset(Path(cfg.root_dir) / f"tests/data/era5test-{site}-{year}.nc")

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
