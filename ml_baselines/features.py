from pathlib import Path
import numpy as np
import xarray as xr
import pandas as pd
import getpass
import gzip

from ml_baselines.config import Config

cfg = Config()
site_coords_dict = cfg.site_coords_dict
met_path = Path(cfg.data_path + "/meteorological_data/ECMWF")
models_path = Path(cfg.models_path)

# Define variables to be extracted
variables = {
        "sp": {
            "file": "single_level",
            "var_in_file": "sp",
            "level": None,
            "units": "hPa",
            "long_name": "Surface Pressure",
        },
        "blh": {
            "file": "single_level",
            "var_in_file": "blh",
            "level": None,
            "units": "m",
            "long_name": "Boundary Layer Height",
        },
        "u10": {
            "file": "single_level",
            "var_in_file": "u10",
            "level": None,
            "units": "m/s",
            "long_name": "10m U-component of Wind",
        },
        "v10": {
            "file": "single_level",
            "var_in_file": "v10",
            "level": None,
            "units": "m/s",
            "long_name": "10m V-component of Wind",
        },
        "u850": {
            "file": "pressure_levels",
            "var_in_file": "u",
            "level": 850,
            "units": "m/s",
            "long_name": "850hPa U-component of Wind",
        },
        "v850": {
            "file": "pressure_levels",
            "var_in_file": "v",
            "level": 850,
            "units": "m/s",
            "long_name": "850hPa V-component of Wind",
        },
        "u500": {
            "file": "pressure_levels",
            "var_in_file": "u",
            "level": 500,
            "units": "m/s",
            "long_name": "500hPa U-component of Wind",
        },
        "v500": {
            "file": "pressure_levels",
            "var_in_file": "v",
            "level": 500,
            "units": "m/s",
            "long_name": "500hPa V-component of Wind",
        },
    }

# Define the grid system
lats_grid = np.array([0, 5, 5, 0, -5, -5, -5, 0, 5, 10, 10, 0, -10, -10, -10, 0, 10])
lons_grid = np.array([0, 0, 5, 5, 5, 0, -5, -5, -5, 0, 10, 10, 10, 0, -10, -10, -10])

# Define the time coordinate in the met files
# time_coord = "valid_time"
time_coord = "time"


def preprocess_features(site, year, force=False):
    """Preprocesses the meteorological data for a given site.

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
    data_path = met_path / site.upper()

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
                    # data_slice = ds.sel(pressure_level=var["level"])
                    data_slice = ds.sel(level=var["level"])
                    # Drop the pressure_level coordinate
                    # data_slice = data_slice.drop_vars(["pressure_level"])
                    data_slice = data_slice.drop_vars(["level"])
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
        # ds_var = ds_var.rename({time_coord: "time"})

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

    files = [models_path / "features" / f"features_{site.upper()}_{year}.csv.gz" for year in range(start_year, end_year+1)]

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

        dfs.append(df)

    df = pd.concat(dfs, axis=0)

    # Create a shifted copy: subtract 6 hours from time by equivalently shifting the index forward by 6 hours.
    # For each record at time T, the _past columns will come from time T – 6 hours.
    df_past = df.copy()
    df_past.index = df_past.index + pd.Timedelta(hours=6)
    df_past = df_past.add_suffix("_6h")

    # Merge the current and past dataframes on their time index
    df_final = pd.merge(df, df_past, left_index=True, right_index=True, how="left")

    return df_final


if __name__ == "__main__":
    # Example usage
    preprocess_all_features(force=False)
