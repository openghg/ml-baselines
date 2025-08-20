from pathlib import Path
import numpy as np
import json


# Path to the root directories of the project
root_dir = Path(__file__).parent.parent
package_dir = root_dir / "ml_baselines"


def setup():
    """Create a config file with default values.
    """

    # Create empty config file
    config_path = package_dir / "config.json"

    config_defaults = {
        "data_path": "",
        "model_type": "MLPClassifier",
        "models_path": str(root_dir / "models"),
        "met_type": "arco-era5"
    }

    with open(config_path, "w") as f:
        json.dump(config_defaults, f, indent=4)

    print(f"Config file created at {config_path}. Please fill in the necessary information.")


class Config():
    """Class to store configuration parameters.
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

        self.data_path = config_user["data_path"]
        self.model_type = config_user["model_type"]
        self.models_path = config_user["models_path"]
        self.met_type = config_user.get("met_type", "arco-era5")  # Default to "arco-era5" if not specified

        if "obs_path" in config_user:
            self.obs_path = config_user["obs_path"]
        else:
            self.obs_path = ""

        # Site codes and names
        self.site_dict = {
                "MHD":"Mace Head, Ireland", 
                "RPB":"Ragged Point, Barbados", 
                "CGO":"Cape Grim, Australia", 
                "GSN":"Gosan, South Korea",
                "JFJ":"Jungfraujoch, Switzerland", 
                "CMN":"Monte Cimone, Italy", 
                "THD":"Trinidad Head, USA", 
                "ZEP":"Zeppelin, Svalbard",
                "SMO": "Cape Matatula, American Samoa"
            }

        # Site coordinates
        self.site_coords_dict = {
                        "MHD":[53.3267, -9.9046], 
                        "RPB":[13.1651, -59.4321], 
                        "CGO":[-40.6833, 144.6894], 
                        "GSN":[33.2924, 126.1616],
                        "JFJ":[46.547767, 7.985883], 
                        "CMN":[44.1932, 10.7014], 
                        "THD":[41.0541, -124.151], 
                        "ZEP":[78.9072, 11.8867],
                        "SMO": [-14.2474, -170.5644]
                    }

        # Time periods
        self.training_period = {"MHD": [2014, 2018],
                                "RPB": [2014, 2018],
                                "CGO": [2014, 2018],
                                "GSN": [2009, 2013],
                                "JFJ": [2014, 2018],
                                "CMN": [2014, 2018],
                                "THD": [2014, 2018],
                                "ZEP": [2014, 2018],
                                "SMO": [2014, 2018]}
        self.validation_period = {"MHD": [2019, 2022],
                                "RPB": [2019, 2019],
                                "CGO": [2019, 2019],
                                "GSN": [2014, 2014],
                                "JFJ": [2019, 2019],
                                "CMN": [2019, 2019],
                                "THD": [2019, 2019],
                                "ZEP": [2019, 2019],
                                "SMO": [2019, 2019]}
        self.testing_period = {"MHD": [2020, 2023],
                                "RPB": [2020, 2023],
                                "CGO": [2020, 2023],
                                "GSN": [2015, 2017],
                                "JFJ": [2020, 2023],
                                "CMN": [2020, 2023],
                                "THD": [2020, 2023],
                                "ZEP": [2020, 2023],
                                "SMO": [2020, 2023]}

        self.confidence_threshold = 0.8

        # Met variables to be extracted (and their order)
        self.met_variables = {
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

        # Define the grid system (deviations in degrees from the site location)
        self.lats_grid = np.array([0, 5, 5, 0, -5, -5, -5, 0, 5, 10, 10, 0, -10, -10, -10, 0, 10])
        self.lons_grid = np.array([0, 0, 5, 5, 5, 0, -5, -5, -5, 0, 10, 10, 10, 0, -10, -10, -10])


if __name__ == "__main__":
    setup()