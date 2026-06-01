'''
This script defines a configuration class used throughout the ml-baselines package.
The Config() class reads from a user-populated file (config.json) and site_info.json, providing access to file paths, site and model metadata.
'''

import json
import numpy as np
from pathlib import Path

root_dir = Path(__file__).parent.parent
package_dir = root_dir / "ml_baselines"


config_defaults = {
        "met_path": "",
        "obs_path": "",
        "model_type": "MLPClassifier",
        "models_path": str(root_dir / "models"),
        "met_type": "arco-era5"
    }

def setup():
    """
    Create a config file with default values.

    Default values are set, apart from met and obs paths, which the user must fill in. 
    The config file is created at ml_baselines/config.json.
    """

    config_path = package_dir / "config.json"

    with open(config_path, "w") as f:
        json.dump(config_defaults, f, indent=4)

    print(f"Config file created at {config_path}. Please fill in the necessary information.")


class Config():
    """
    Class to store configuration parameters.

    """

    def __init__(self):

        if not (package_dir / "config.json").exists():
            raise FileNotFoundError("Config file not found. Please run setup() to create a new config file, and then populate it.")

        # Paths
        self.root_dir = root_dir
        self.package_dir = package_dir

        # Read user config file
        with open(package_dir / "config.json") as f:
            config_user = json.load(f)
        
        for key, value in config_defaults.items():
            if key in config_user:
                setattr(self, key, config_user[key])
            else:
                setattr(self, key, value)

        # Read site info file
        with open(root_dir / "data/site_info.json") as f:
            site_info = json.load(f)

        self.site_dict = {site_code: site_info[site_code]["name"] for site_code in site_info}
        self.site_coords_dict = {site_code: site_info[site_code]["coords"] for site_code in site_info}
        self.training_period = {site_code: site_info[site_code]["training_period"] for site_code in site_info}
        self.validation_period = {site_code: site_info[site_code]["validation_period"] for site_code in site_info}
        self.testing_period = {site_code: site_info[site_code]["testing_period"] for site_code in site_info}
        self.full_period = {site_code: site_info[site_code]["full_period"] if "full_period" in site_info[site_code] else None for site_code in site_info}

        with open(root_dir / "data/met_info.json") as f:
            met_info = json.load(f)
        self.met_variables = met_info

        # Define the grid system (deviations in degrees from the site location)
        self.lats_grid = np.array([0, 5, 5, 0, -5, -5, -5, 0, 5, 10, 10, 0, -10, -10, -10, 0, 10])
        self.lons_grid = np.array([0, 0, 5, 5, 5, 0, -5, -5, -5, 0, 10, 10, 10, 0, -10, -10, -10])


if __name__ == "__main__":
    setup()
