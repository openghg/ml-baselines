# ml-baselines

A machine learning library for the estimation of greenhouse gas baseline timeseries from high-frequency observations.

## Setup

### File structure

This repository is structured as follows:

```
data/ # data files required to run the code
   ├── intem_baselines.zip  # baseline flags from InTEM
   ├── site_info.json # Site information (location, validation/testing periods, etc.)
   └── met_info.json # Meteorological variable information (variable names, levels, etc.)
docker/ # dockerfiles for extracting meteorlogical fields from ERA-5 on GCP in a container
ml_baselines/   # main package
├── met_retrieve/
├── models/
├── data/
└── config.json # untracked config file for specifying paths
models/ # trained models and features
└── features/ # extracted features for training/testing
notebooks/ # Jupyter notebooks for experimentation and visualization
tests/ # unit and integration tests
```

In addition, you must specify the location of meteorological fields and mole fraction observations.

### Configuration

Some configuration parameters are required to run this code. These are stored in an untracked file ```ml_baselines/config.json```. To create a template of this file, run:

```
python ml_baselines/config.py
```

Example structure of ```config.json```:

```
{
    "met_path": <path to meteorological data files>,
    "obs_path": <path to mole fraction observation files>,
    "model_type": "MLPClassifier",
    "models_path": <defaults to location in repository>,
    "met_type": "arco-era5"
}
```

### Meteorological fields

Routines are provided for downloading and processing ECMWF ERA5 meteorological fields into the required format. 

We provide two possible routes to obtaining these data:
1. Slices of ERA5 variables can be retrieved using the ECMWF CDF API (https://cds.climate.copernicus.eu/how-to-api) using the functions in ```ml_baselines/met_retrieve/ecmwf_retrieve.py```. The input features can then be extracted for individual years using ```ml_baselines.features.preprocess_features```, or for all years using ```ml_baselines.features.preprocess_features_all_years```.
2. Alternatively, you can extract the relevant meteorological points from the ARCO-ERA5 dataset that has been archived into ```zarr``` format directly from e.g., the Google Cloud Storage bucket (https://console.cloud.google.com/storage/browser/arco-era5). A container for extracting the relevant points is provided in ```ml_baselines/met_retrieve/gcp_era5```. The extracted data can be processed into features using ```ml_baselines.features.preprocess_features_arco_era5``` and ```ml_baselines.features.preprocess_all_features_arco_era5```. You can also run the GCP retrieval locally, but this will take substantially longer than running on the cloud.

The advantage of the first approach is it is free to use, and you could explore the use of different grids, etc. The second approach is much faster as all of the processing can be done in parallel (hours versus weeks to download) and requires orders of magnitude less storage space, since only the required fields are extracted in the cloud. However, it uses GCP credits (~$100 to extract ~40 years of data at 9 sites). The zarr store could also be accessed from an external server, but processing would likely be substantially slower (not tested in earnest).


### Observational data

Currently, will only read in AGAGE data files. These can be provided as a zip archive of the type that can be downloaded from the AGAGE website.


## Developer notes

To install an editable version of this package in your environment, go to the root directory of this repo and type:

```
pip install --no-build-isolation --no-deps -e .
```