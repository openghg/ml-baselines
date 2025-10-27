from pathlib import Path
import numpy as np
import xarray as xr
import pandas as pd
import getpass
import gzip

from ml_baselines.config import Config
from ml_baselines.utils import longitude_to_360

cfg = Config()
site_coords_dict = cfg.site_coords_dict
met_path = Path(cfg.data_path + "/meteorological_data")
models_path = Path(cfg.models_path)

lats_grid = cfg.lats_grid
lons_grid = cfg.lons_grid
variables = cfg.met_variables

# Define the time coordinate in the met files
time_coord = "valid_time"


def preprocess_features(site, year, force=False):
    """Preprocesses the meteorological data for a given site using slices of ERA5 from the CDS API.

    Features will be extracted from the ECMWF ERA5 reanalysis data for the specified site and year.
    The data will be interpolated onto a grid system with +/- 5 and 10 degrees latitude and longitude from the site of interest.
    The processed data will be saved to a netCDF file in the models_path / features directory.
    The file will be named features_<site>_<year>.nc.
    The data will be saved in the following format:
        - time: time coordinate
        - points: grid points
        - variables: meteorological variables (e.g., temperature, humidity, wind speed)

    Args:
        site (str): Site code.
        year (int): Year to process.
        force (bool): If True, force reprocessing even if the file already exists.
    """

    # Path to the data
    data_path = met_path / "ECMWF" / site.upper()

    output_filename = models_path / "features" / f"features_{site}_{year}.csv.gz"

    # Check if the data path exists
    if not data_path.exists():
        raise ValueError(f"Data path {data_path} does not exist. Please check the site code.")

    # Check if the output file already exists
    if output_filename.exists() and not force:
        print(f"Output file {output_filename} already exists. Skipping.")
        return
    
    # Get the coordinates of the site
    site_lat, site_lon = site_coords_dict[site]

    # creating a grid system with +/- 5 and 10 degrees latitude and longitude from the site of interest
    points_lat = lats_grid + site_lat
    points_lon = lons_grid + site_lon
    points = range(17)

    # creating an xarray DataArray for the grid coordinates
    lats = xr.DataArray(points_lat, dims=["points"], coords={"points": points})
    lons = xr.DataArray(points_lon, dims=["points"], coords={"points": points})

    for var_name in variables.keys():

        var = variables[var_name]
        files = sorted((data_path / var["file"]).glob(f"{site.upper()}*{year}*.nc"))

        if len(list(files)) == 0:
            print(f"No files found for {site} in {year}: {var_name}.")
            return
        if len(list(files)) > 12:
            raise ValueError(f"More than 12 files found for {site} in {year}: {var_name}. Please check the data.")
        if len(list(files)) < 12:
            print(f"WARNING: only {len(list(files))} months available for {site} in {year}: {var_name}.")

        data = []

        for f in files:
            with xr.open_dataset(f) as ds:
                # Extract the variable and interpolate the data onto the grid
                if var["level"] is not None:
                    data_slice = ds.sel(pressure_level=var["level"])
                    # Drop the pressure_level coordinate
                    data_slice = data_slice.drop_vars(["pressure_level"])
                else:
                    data_slice = ds

                data_slice = \
                    data_slice[var["var_in_file"]]\
                        .interp(latitude=lats, longitude=lons, method="nearest")\
                        .assign_coords({time_coord: ds[time_coord].values})
                if "number" in data_slice.coords:
                    data_slice = data_slice.drop_vars(["number"])
                if "expver" in data_slice.coords:
                    data_slice = data_slice.drop_vars(["expver"])

                data.append(data_slice)

        # Concatenate the data along the time dimension, and add points as a coordinate
        ds_var = xr.concat(data, dim=time_coord)

        # Rename the time dimension to 'time'
        ds_var = ds_var.rename({time_coord: "time"})

        # If the variable has a level coordinate, rename
        if var["level"] is not None:
            ds_var.name = f"{var_name}"

        # Rename the long_name and units attributes
        ds_var.attrs["long_name"] = var["long_name"]

        # If ds_out hasn't been created yet, create it
        if "ds_out" not in locals():
            ds_out = ds_var
        else:
            # Merge the new variable with the existing dataset
            ds_out = xr.merge([ds_out, ds_var])

    # Check that all varaibles exist for all time points
    if ds_out.isnull().any():
        for var in ds_out.data_vars:
            if ds_out[var].isnull().any():
                # Check if the variable is missing for all time points
                if ds_out[var].isnull().all():
                    raise ValueError(f"All data is missing for {var} in {site} in {year}. Please check the data.")
                else:
                    # Find which months are missing data
                    missing_months = ds_out[var].isnull().groupby("time.month").sum(dim="time")
                    missing_months = list(missing_months.where(missing_months > 0, drop=True).month.values)
                    raise ValueError(f"Missing data for {var} in {site} in {year}, month {', '.join([str(m) for m in missing_months])}")

    # For each variable, pivot the (time, points) array into a wide-form DataFrame
    dfs = []
    for var in ds_out.data_vars:
        # Convert the DataArray for a variable to a DataFrame and pivot so that each grid point becomes a separate column
        df_var = (
            ds_out[var]
            .to_dataframe()
            .reset_index()
            .pivot(index="time", columns="points", values=var)
        )
        # Rename columns to include the variable name (e.g., sp_0, sp_1, etc.)
        df_var.columns = [f"{var}_{int(pt)}" for pt in df_var.columns]
        dfs.append(df_var)

    # Merge all variable-wise DataFrames on the time index
    df = pd.concat(dfs, axis=1)

    # Sort by time index
    df = df.sort_index()
    # Add the time index as a column
    df = df.reset_index()
    
    with gzip.open(output_filename, "wt") as f:
        f.write("# Subset of the ECMWF ERA5 reanalysis data for use calculating ML baselines\n")
        f.write("# Data is interpolated onto a grid system with +/- 5 and 10 degrees latitude and longitude from the site of interest\n")
        f.write("# Column names have the following format:\n")
        f.write("# - the text before the first underscore is the variable name\n")
        f.write("# - the text after the first underscore is the grid point number\n")
        f.write("# - the text after the (optional) second underscore is the time shift\n")
        f.write(f"# Processed by: {getpass.getuser()}\n")
        f.write(f"# Processed on: {pd.Timestamp.now()}\n")
        f.write(f"# Processed by ml-baselines code\n")
        f.write(f"# Site: {site}\n")
        f.write(f"# Year: {year}\n")
 
        # Save the DataFrame to a CSV file
        df.to_csv(f, index=False, header=True, mode="a")

    print(f"Preprocessed features for {site} in {year} and saved to {output_filename}")


def preprocess_features_arco_era5(site,
                                  force = False,
                                  input_dir = "",
                                  output_dir = ""):
    """Preprocess features that have been extracted from the ARCO ERA5 reanalysis data.

    These files should have already undergone some preprocessing (see gcp_era5 container), including 
    interpolation onto a grid with +/- 5 and 10 degrees latitude and longitude

    Args:
        site (str): Site code.
        force (bool): If True, force reprocessing even if the file already exists.
        input_dir (str): Directory where the input files are located if not in location specified in config. 
            Mainly used for testing purposes. If empty, uses default path.
        output_dir (str): Directory where the output files will be saved if not in location specified in config. 
            Mainly used for testing purposes. If empty, uses default path.
    Returns:
        None
    """
    
    # Path to the data
    if input_dir:
        data_path = Path(input_dir)
    else:
        data_path = met_path / "arco-era5"

    files = sorted((data_path).glob(f"era5*{site.upper()}*.nc"))

    if len(list(files)) == 0:
        raise ValueError(f"No files found for {site}.")
    
    # Check if every year between the first and last year is present
    years = [int(f.name.split("-")[-1][:4]) for f in files]
    for year in range(min(years), max(years)+1):
        if year not in years:
            print(f"WARNING: Year {year} is missing for {site}.")

    ds = xr.open_mfdataset(files, combine="by_coords")

    # Check that the dataset has the expected coordinates
    if "points" not in ds.coords:
        raise ValueError(f"Dataset for {site} does not have the 'points' coordinate. Please check the data.")
    if "latitude" not in ds.coords or "longitude" not in ds.coords:
        raise ValueError(f"Dataset for {site} does not have the expected latitude or longitude coordinates. Please check the data.")
    if "levels" not in ds.coords:
        raise ValueError(f"Dataset for {site} does not have the 'levels' coordinate. Please check the data.")

    # Check that grid points are ordered the same way as the lons_grid and lats_grid dataArray
    if not np.allclose(lats_grid + ds.latitude.values[0], ds.latitude.values, rtol=0.1):
        raise ValueError("Extracted points are not aligned with expected grid latitude points")
    lons_expected = lons_grid + ds.longitude.values[0]

    if not np.allclose(longitude_to_360(lons_expected),
                       longitude_to_360(ds.longitude.values), rtol=0.1):
        raise ValueError("Extracted points are not aligned with expected grid longitude points")

    # u_component_of_wind and v_component_of_wind are at 500hPh and 850hPa levels
    # Flatten these variables into single variables with level suffixes
    for level in [500, 850]:
        ds[f"u{level}"] = ds[f"u_component_of_wind"].sel(levels=level)
        ds[f"v{level}"] = ds[f"v_component_of_wind"].sel(levels=level)
    ds = ds.drop_vars(["u_component_of_wind", "v_component_of_wind", "levels"])

    # Simplify variable names for consistency with previous code
    ds = ds.rename({
        "surface_pressure": "sp",
        "boundary_layer_height": "blh",
        "10m_u_component_of_wind": "u10",
        "10m_v_component_of_wind": "v10"})

    # Find duplicate times and drop the first occurrence of each duplicate
    if ds.indexes["time"].duplicated().any():
        ds = ds.sel(time=~ds.indexes["time"].duplicated())

    dfs = []

    # For each variable, pivot the (time, points) array into a wide-form DataFrame
    # Use the variables defined in the variables dict to ensure consistency
    for var in variables.keys():
        # Convert the DataArray for a variable to a DataFrame and pivot so that each grid point becomes a separate column
        df_var = (
            ds[var]
            .to_dataframe()
            .reset_index()
            .pivot(index="time", columns="points", values=var)
        )
        # Rename columns to include the variable name (e.g., sp_0, sp_1, etc.)
        df_var.columns = [f"{var}_{int(pt)}" for pt in df_var.columns]
        dfs.append(df_var)

    # Merge all variable-wise DataFrames on the time index
    df = pd.concat(dfs, axis=1)

    # Sort by time index
    if not df.index.is_monotonic_increasing:
        df = df.sort_index()
    # Add the time index as a column
    df = df.reset_index()

    for year in years:
        if output_dir:
            output_filename = Path(output_dir) / f"features-arco-era5_{site}_{year}.csv.gz"
        else:
            output_filename = models_path / "features" / f"features-arco-era5_{site}_{year}.csv.gz"
        if output_filename.exists() and not force:
            print(f"Skipping {output_filename}, already exists.")
            continue

        df_year = df[(df["time"] >= pd.Timestamp(f"{year}-01-01")) & (df["time"] < pd.Timestamp(f"{year+1}-01-01"))]

        with gzip.open(output_filename, "wt", compresslevel=6) as f:
            f.write("# Subset of the ECMWF ERA5 reanalysis data for use calculating ML baselines\n")
            f.write("# Data is interpolated onto a grid system with +/- 5 and 10 degrees latitude and longitude from the site of interest\n")
            f.write("# Column names have the following format:\n")
            f.write("# - the text before the first underscore is the variable name\n")
            f.write("# - the text after the first underscore is the grid point number\n")
            f.write("# - the text after the (optional) second underscore is the time shift\n")
            f.write(f"# Processed by: {getpass.getuser()}\n")
            f.write(f"# Processed on: {pd.Timestamp.now()}\n")
            f.write(f"# Processed by ml-baselines code\n")
            f.write(f"# Site: {site}\n")
            f.write(f"# Year: {year}\n")
    
            # Save the DataFrame to a CSV file
            df_year.to_csv(f, index=False, header=True, mode="a")

        print(f"Preprocessed features for {site} in {year} and saved to {output_filename}")


def preprocess_all_features(start_year=1978, end_year=2024, force=False):
    """Preprocesses the meteorological data for all sites and years."""
    for site in site_coords_dict.keys():
        for year in range(start_year, end_year):
            try:
                preprocess_features(site, year, force=force)
            except Exception as e:
                print(f"Error processing {site} in {year}: {e}")
                continue
    print("Preprocessing complete.")


def preprocess_all_features_arco_era5(force=False):
    """Preprocesses the meteorological data for all sites and years."""
    for site in site_coords_dict.keys():
        try:
            preprocess_features_arco_era5(site, force=force)
        except Exception as e:
            print(f"Error processing {site}: {e}")
            continue
    print("Preprocessing complete.")


def open_features(site,
                start_year=1978,
                end_year=2024):
    """Opens the preprocessed features for a given site.

    Args:
        site (str): Site code.
        start_year (int): Start year to retrieve data (inclusive).
        end_year (int): End year for to retrieve data (inclusive).

    Returns:
        pd.DataFrame: Preprocessed features for the site.
    """

    features_str = "-arco-era5" if cfg.met_type == "arco-era5" else ""

    expected_columns = [key + f"_{i}" for key in cfg.met_variables.keys() for i in range(17)]

    files = [models_path / "features" / f"features{features_str}_{site.upper()}_{year}.csv.gz" for year in range(start_year, end_year+1)]

    for f in files:
        if not f.exists():
            error_message = f"File {f} does not exist. Please preprocess the data first, or set a start and end year."
            raise ValueError(error_message)

    dfs = []
    for f in files:
        df = pd.read_csv(f,
            comment="#",
            compression="gzip",
            parse_dates=["time"],
            index_col=["time"],
            )

        # Check that all varaibles exist for all time points
        for var in df.columns:
            if df[var].isnull().any():
                # Check if the variable is missing for all time points
                if df[var].isnull().all():
                    raise ValueError(f"All data is missing for {var} in file {f} . Please check the data.")
                else:
                    # Find which months are missing data
                    missing_months = df[var].isnull().groupby(df.index.month).sum()
                    missing_months = list(missing_months.where(missing_months > 0, drop=True).index)
                    raise ValueError(f"Missing data for {var} in {site}, month {', '.join([str(m) for m in missing_months])}")

        # Check for correct columns in the right order
        if list(df.columns) != expected_columns:
            # Check if it's the order that is different
            if sorted(df.columns) == sorted(expected_columns):
                raise ValueError(f"Incorrect column order in file {f} . Please check the data.")
            else:
                raise ValueError(f"Incorrect columns in file {f} . Please check the data.")

        dfs.append(df)

    df = pd.concat(dfs, axis=0)

    # Create a shifted copy: subtract 6 hours from time by equivalently shifting the index forward by 6 hours.
    # For each record at time T, the _past columns will come from time T – 6 hours.
    df_past = df.copy()
    df_past.index = df_past.index + pd.Timedelta(hours=6)
    df_past = df_past.add_suffix("_6h")

    # Merge the current and past dataframes on their time index
    df_final = pd.merge(df, df_past, left_index=True, right_index=True, how="left")

    # Add hour of day column
    df_final["hour_of_day"] = df_final.index.hour

    return df_final


if __name__ == "__main__":

    if cfg.met_type == "arco-era5":
        preprocess_all_features_arco_era5(force=True)
    else:
        preprocess_all_features(force=True)