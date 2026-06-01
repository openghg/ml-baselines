'''
This script downloads meteorological data from the ECMWF (European Centre for Medium-Range Weather Forecasts) ARCO-ERA5 dataset on the Google Cloud Storage bucket.
The data being downloaded has been limited to the area surrounding each site.

'''

import argparse
import gcsfs
import numpy as np
import tempfile
import xarray as xr

from ml_baselines.config import Config
from ml_baselines.utils import longitude_to_360

cfg = Config()
site_coords_dict = cfg.site_coords_dict

lons_grid = cfg.lons_grid
lats_grid = cfg.lats_grid

arco_era5_location = 'gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3'

gcp_bucket_name = "ml-baselines-era5"

levels = [850, 500]

variables_3d = ["u_component_of_wind", "v_component_of_wind"]
variables_2d = ["10m_u_component_of_wind", "10m_v_component_of_wind",
                "surface_pressure", "boundary_layer_height"]


def retrieve_grid(arco_era5_location):
    """ 
    Retrieve the grid from the ARCO ERA5 dataset.

    Args:
        arco_era5_location (str): The location of the ARCO ERA5 dataset.
    Returns:
        tuple: A tuple containing the grid latitudes, longitudes, and levels.

    """

    with xr.open_zarr(
        arco_era5_location,
        chunks=None,
        storage_options=dict(token='anon'),
        ) as ds:
        grid_lats = ds["latitude"].values
        grid_lons = ds["longitude"].values
        grid_levels = ds["level"].values

    return grid_lats, grid_lons, grid_levels


def define_points(site):
    """ 
    Define points based on the site coordinates and the grid.

    Args:
        site (str): The site for which to define points.
    Returns:
        tuple: A tuple containing the latitude points, longitude points, and their indices.

    """

    points_lat = []
    points_lon = []
    
    site_lat, site_lon = site_coords_dict[site]
    
    # creating a grid system with +/- 5 and 10 degrees latitude and longitude from the site of interest
    points_lat.append(lats_grid + site_lat)
    points_lon.append(lons_grid + site_lon)

    points_lon = np.concatenate(points_lon)
    points_lat = np.concatenate(points_lat)
    points = np.arange(len(points_lat))

    return points_lat, longitude_to_360(points_lon), points


def find_closest_grid_points(grid_lats, grid_lons,
                            points_lat, points_lon, points):
    """
    Find the closest grid points for the given latitude, longitude, and level.

    Args:
        grid_lats (np.ndarray): The grid latitude values.
        grid_lons (np.ndarray): The grid longitude values.
        points_lat (np.ndarray): The latitude points to match.
        points_lon (np.ndarray): The longitude points to match.
        points (np.ndarray): The point indices.

    Returns:
        tuple: A tuple containing the closest latitude points, longitude points, and level points.

    """
    global levels

    # Find values in grid_lats and grid_lons that are closest to the points
    def find_closest_indices(grid, points):
        indices = []
        for point in points:
            # Calculate the absolute difference between the grid and the point
            diff = np.abs(grid - point)
            # Find the index of the minimum difference
            index = np.argmin(diff)
            indices.append(index)
        return indices

    closest_lat_indices = find_closest_indices(grid_lats, points_lat)
    closest_lon_indices = find_closest_indices(grid_lons, points_lon)

    points_lat_closest = grid_lats[closest_lat_indices]
    points_lon_closest = grid_lons[closest_lon_indices]

    lats = xr.DataArray(points_lat_closest, dims=["points"], coords={"points": points})
    lons = xr.DataArray(points_lon_closest, dims=["points"], coords={"points": points})
    levels = xr.DataArray(levels, dims=["levels"], coords={"levels": levels})

    return lats, lons, levels


def get(year, lats, lons, levels):
    """ 
    Retrieve the dataset for the specified latitudes, longitudes, and levels.
    Args:
        year (int): The year of interest.
        lats (xarray.DataArray): The latitude points.
        lons (xarray.DataArray): The longitude points.
        levels (xarray.DataArray): The level points.
    Returns:
        xarray.Dataset: The dataset containing the specified variables.

    """

    time_slice = slice(f"{year}-01-01T00:00:00",
                       f"{year+1}-01-01T00:00:00")

    variables_3d = ["u_component_of_wind", "v_component_of_wind"]
    variables_2d = ["10m_u_component_of_wind", "10m_v_component_of_wind",
                    "surface_pressure", "boundary_layer_height"]
    
    with xr.open_zarr(
            arco_era5_location,
            chunks=None,
            storage_options=dict(token='anon'),
            ) as ds:

        ds_points_3d = ds[variables_3d].sel(
            latitude=lats,
            longitude=lons,
            level=levels,
            time=time_slice).load()

        ds_points_2d = ds[variables_2d].sel(
            latitude=lats,
            longitude=lons,
            time=time_slice).load()

    ds_points = xr.merge([ds_points_3d, ds_points_2d])

    return ds_points


def save_to_bucket(ds_points, bucket_name, file_name):
    """ 
    Save the dataset to a Google Cloud Storage bucket.

    Args:
        ds_points (xarray.Dataset): The dataset to save.
        bucket_name (str): The name of the GCP bucket.
        file_name (str): The name of the file to save.

    """

    with tempfile.NamedTemporaryFile(suffix=".nc") as tmp:
        ds_points.to_netcdf(tmp.name, mode="w", engine="netcdf4", format="NETCDF4")
        fs = gcsfs.GCSFileSystem(token='google_default')
        fs.put(tmp.name, f"{bucket_name}/{file_name}")

    # Uncomment to save locally
    # ds_points.to_netcdf(f"/app/output/{file_name}", mode='w', format='NETCDF4')


def run(site, year):
    """ 
    Run the process to retrieve and save ERA5 data for a specific site.

    Args:
        site (str): The site for which to retrieve data.
        year (int): The year of interest.

    """
    global levels

    if site not in site_coords_dict:
        raise ValueError(f"Site {site} not found in site coordinates dictionary.")

    grid_lats, grid_lons, grid_levels = retrieve_grid(arco_era5_location)

    for level in levels:
        if level not in grid_levels:
            raise ValueError(f"Level {level} not found in grid levels.")

    points_lat, points_lon, points = define_points(site)

    lats, lons, levels = find_closest_grid_points(grid_lats, grid_lons,
                            points_lat, points_lon, points)

    ds_points = get(year, lats, lons, levels)

    save_to_bucket(ds_points, gcp_bucket_name, f"era5-{site}-{year}.nc")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Retrieve ERA5 data for a specific site.")
    parser.add_argument("site", type=lambda s: s.strip(), help="Site for which to retrieve data (e.g., 'MHD').")
    parser.add_argument("year", type=lambda s: int(s.strip()), help="Year of interest (e.g., 1978).")
    args = parser.parse_args()

    run(args.site, args.year)