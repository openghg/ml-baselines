import xarray as xr
import numpy as np
import pandas as pd
from pathlib import Path
import zipfile

from ml_baselines.config import Config


cfg = Config()
site_coords_dict = cfg.site_coords_dict
data_path = Path(cfg.data_path)
package_path = cfg.package_dir
root_path = cfg.root_dir


def read_intem(site):
    """
    Extracting baseline flags for a given site

    Args:
    - site (str): Site code (e.g., MHD)

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

        for file in files:
            # Read the data, skipping metadata, putting into pandas dataframe
            data = pd.read_csv(file, skiprows=6, sep='\s+')

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
