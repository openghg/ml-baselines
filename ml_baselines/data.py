import xarray as xr
import numpy as np
import pandas as pd
import io
from pathlib import Path
import zipfile

from ml_baselines.config import Config


cfg = Config()
site_coords_dict = cfg.site_coords_dict
package_path = cfg.package_dir
root_path = cfg.root_dir


def read_intem(site,
               start_year = None,
               end_year = None):
    """
    Extracting baseline flags for a given site

    Args:
    - site (str): Site code (e.g., MHD)
    - start_year (int): Start year for the data extraction (inclusive)
    - end_year (int): End year for the data extraction (inclusive)

    Returns:
    - df (pandas.DataFrame): DataFrame with baseline flags as a binary variable
    """
    
    site_translator = {"MHD":"MH",
                       "CGO":"CG",
                       "GSN":"GS",
                       "JFJ":"J1",
                       "CMN":"M5",
                       "THD":"TH",
                       "ZEP":"ZE",
                       "RPB":"BA",
                       "SMO":"SM"}

    # zip file location
    intem_zip_path = root_path / "data" / "intem_baselines.zip"

    dfs = []

    # Find the files in the zip archive
    with zipfile.ZipFile(intem_zip_path, 'r') as zip_ref:

        # Find all files in archive matching "{site_translator[site]}*.txt"
        files = [zip_ref.extract(file, path=package_path / "data") for file in zip_ref.namelist() if file.startswith(f"{site_translator[site]}") and file.endswith(".txt")]

        # If start_year is not None, filter files by year
        if start_year is not None:
            files = [file for file in files if int(file.split("_")[-1][:4]) >= start_year]
        if end_year is not None:
            files = [file for file in files if int(file.split("_")[-1][:4]) <= end_year]

        for file in files:
            # Read the data, skipping metadata, putting into pandas dataframe
            data = pd.read_csv(file, skiprows=6, sep=r'\s+')

            # Setting the index of the dataframe to be the extracted datetime and naming it time
            data.index = pd.to_datetime(data['YY'].astype(str) + "-" + \
                                        data['MM'].astype(str) + "-" + \
                                        data['DD'].astype(str) + " " + \
                                        data['HH'].astype(str) + ":00:00")

            data.index.name = "time"
            
            # Adding the 'Ct' column to the previously created empty list
            dfs.append(data[["Ct"]])
    
    # Creating a dataframe from the list containing all the 'Ct' values
    df = pd.concat(dfs)

    df.sort_index(inplace=True)

    # Replace all values in Ct column less than 10 or greater than 20 with 0
    # not baseline values
    df.loc[(df['Ct'] < 10) | (df['Ct'] >= 20), 'Ct'] = 0

    # Replace all values between 10 and 19 with 1
    # baseline values
    df.loc[(df['Ct'] >= 10) & (df['Ct'] < 20), 'Ct'] = 1

    # Rename Ct column to "baseline"
    df.rename(columns={'Ct': 'baseline'}, inplace=True)

    # Convert baseline column to int
    df['baseline'] = df['baseline'].astype(int)

    return df


def read_agage(site, species,
               start_year=None,
               end_year=None):
    """
    Read AGAGE data for a specific site and species.

    Args:
        site (str): Site code (e.g., MHD)
        species (str): Species code (e.g., cfc-11)
        start_year (int): Start year for the data extraction (inclusive)
        end_year (int): End year for the data extraction (inclusive)

    Returns:
        df (pandas.DataFrame): DataFrame with AGAGE data
    """

    agage_path = Path(cfg.obs_path)

    if agage_path.suffix == ".zip":
        # Assume there is a version number in the archive name
        version = agage_path.stem.split("-")[-1]

        with zipfile.ZipFile(agage_path, 'r') as zf:
            # Extract the file for a specific site/species
            nc_filename = f"{species}/agage_{site.lower()}_{species.lower()}_{version}.nc"
            nc_bytes = zf.read(nc_filename)
            with io.BytesIO(nc_bytes) as memfile:
                ds = xr.open_dataset(memfile)
    else:
        raise NotImplementedError("Only zip archive is supported for obs_path at the moment.")

    ds = ds.sel(time=slice(f"{start_year}-01-01", f"{end_year}-12-31"))

    df = ds[["mf", "mf_repeatability"]].to_dataframe()
    if "mf_variability" in ds:
        df["mf_variability"] = ds["mf_variability"].to_dataframe()
    else:
        df["mf_variability"] = 0.

    return df

