# ml-baselines

A machine learning library for the estimation of greenhouse gas baseline timeseries from high-frequency observations.

## Setup

Some configuration parameters are required to run this code. These are stored in an untracked file ```ml_baselines/config.json```. To create a template of this file, run:

```
python ml_baselines/config.py
```

Input the ```data_path``` and other parameters in the relevant fields.

## Developer notes

To install an editable version of this package in your environment, go to the root directory of this repo and type:

```
pip install --no-build-isolation --no-deps -e .
```