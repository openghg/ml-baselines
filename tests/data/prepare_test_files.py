# Script to prepare test files for unit tests
from pathlib import Path
import xarray as xr
from ml_baselines.config import Config

cfg = Config()
met_path = Path(cfg.data_path + "/meteorological_data")


# Load the original ERA5 MHD dataset and create a smaller test file
with xr.open_dataset(met_path / "arco-era5/era5-MHD-2020.nc") as ds:
    
    # Select a subset of the data for testing
    ds_test = ds.isel(time=slice(0, 500))

    # Save the subset to a new NetCDF file
    ds_test.to_netcdf(Path(cfg.root_dir) / "tests/data/era5test-MHD-2020.nc")
