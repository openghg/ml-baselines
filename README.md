# ml-baselines

A machine learning library for the estimation of greenhouse gas baseline timeseries from high-frequency observations.

## Setup

Some configuration parameters are required to run this code. These are stored in an untracked file ```ml_baselines/config.json```. To create a template of this file, run:

```
python ml_baselines/config.py
```

Input the ```data_path``` and other parameters in the relevant fields.

### Meteorological fields

Routines are provided for downloading and processing ECMWF ERA5 meteorological fields into the required format. 

We provide two possible routes to obtaining these data:
1. Slices of ERA5 variables can be retrieved using the ECMWF CDF API (https://cds.climate.copernicus.eu/how-to-api) using the functions in ```ml_baselines/met_retrieve/ecmwf_retrieve.py```. The input features can then be extracted for individual years using ```ml_baselines.features.preprocess_features```, or for all years using ```ml_baselines.features.preprocess_features_all_years```.
2. Alternatively, you can extract the relevant meteorological points from the ARCO-ERA5 dataset that has been archived into ```zarr``` format directly from e.g., the Google Cloud Storage bucket (https://console.cloud.google.com/storage/browser/arco-era5). A container for extracting the relevant points is provided in ```ml_baselines/met_retrieve/gcp_era5```. The extracted data can be processed into features using ```ml_baselines.features.preprocess_features_arco_era5``` and ```ml_baselines.features.preprocess_all_features_arco_era5```. You can also run the GCP retrieval locally, but this will take substantially longer than running on the cloud.

The advantage of the first approach is it is free to use, and you could explore the use of different grids, etc. The second approach is much faster as all of the processing can be done in parallel (hours versus weeks to download) and requires orders of magnitude less storage space, since only the required fields are extracted in the cloud. However, it uses GCP credits (~$100 to extract ~40 years of data at 9 sites). The zarr store could also be accessed from an external server, but processing would likely be substantially slower (not tested in earnest).


## Developer notes

To install an editable version of this package in your environment, go to the root directory of this repo and type:

```
pip install --no-build-isolation --no-deps -e .
```