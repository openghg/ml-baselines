'''
This script downloads meteorological data from the ECMWF (European Centre for Medium-Range Weather Forecasts) using the CDS (Climate Data Store) API. 
The data being downloaded has been limited to the area surrounding each site.
The data being downloaded is u and v wind at pressure levels of 500 and 850 hPa.
'''

import cdsapi
import sys
from pathlib import Path
import os
from multiprocessing import Pool
#from ml_baselines import config

c = cdsapi.Client()

#TODO: This stuff should be in a config file
site_coords_dict = {"MHD":[53.3267, -9.9046], 
                    "RPB":[13.1651, -59.4321], 
                    "CGO":[-40.6833, 144.6894], 
                    "GSN":[33.2924, 126.1616],
                    "JFJ":[46.547767, 7.985883], 
                    "CMN":[44.1932, 10.7014], 
                    "THD":[41.0541, -124.151], 
                    "ZEP":[78.9072, 11.8867],
                    "SMO": [-14.2474, -170.5644]}

root_path = Path("/home/chxmr/data/ml-baselines/meteorological_data/ECMWF")

months = [str(m).zfill(2) for m in range(1, 13)]


def retrieve_dict(level, month, year, domain):
    """ Returns a dictionary of parameters to be used in the CDS API request.

    Args:
        level (str): 'pressure' or 'single'
        month (int): month number (0-11)
        year (int): year
        domain (tuple): top-left and bottom-right coordinates of the area to be downloaded

    Returns:
        dict: dictionary of parameters to be used in the CDS API request
    """

    dict = {
        'product_type': 'reanalysis',
        'year': f'{year}',
        'month': months[month],
        'day': [str(d).zfill(2) for d in range(1, 32)],
        'time': [str(t).zfill(2) + ':00' for t in range(24)],
        'format': 'netcdf',
        'area': domain,
    }

    if level == "pressure":
        dict['variable'] = ['u_component_of_wind',
                            'v_component_of_wind']
        dict['pressure_level'] = ['500',
                                  '850']
    elif level == "single":
        dict['variable'] = ['10m_u_component_of_wind',
                            '10m_v_component_of_wind',
                            'surface_pressure',
                            'boundary_layer_height']
    else:
        raise ValueError("Invalid level. Must be 'pressure' or 'single'.")

    return dict


def retrieve_site_month(site, level, year, month,
                        domain_size = 11):
    """ Downloads meteorological data for a specific site and month.

    Args:
        site (str): site code
        level (str): 'pressure' or 'single'
        year (int): year
        month (int): month number (0-11)
        domain_size (int): size of the area to be downloaded (in degrees from centre)

    Returns:
        None
    """

    if site not in site_coords_dict.keys():
        raise ValueError("Invalid site. Must be one of {}".format(site_coords_dict.keys()))

    if level == "pressure":
        output_path = root_path / site / "pressure_levels"
        output_filename = output_path / f"{site}_3dwind_{year}_{months[month]}.nc"
        dataset = 'reanalysis-era5-pressure-levels'
    elif level == "single":
        output_path = root_path / site / "single_level"
        output_filename = output_path / f"{site}_2dmet_{year}_{months[month]}.nc"
        dataset = 'reanalysis-era5-single-levels'
    else:
        raise ValueError("Invalid level. Must be 'pressure' or 'single'.")

    #TODO: I think this causes an error when run in parallel. Move higher in the call stack?
    # If output path doesn't exist, create it
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Retrieve using CDS API
    if os.path.exists(output_filename):
        print(f"{site} {months[month]} {year} already downloaded. Skipping.")
    else:
        print(f'Downloading {site} {level}: {months[month]} {year}')
        
        # domain is top-left and bottom-right coordinates
        # max_lat, min_lon, min_lat, max_lon
        domain = (site_coords_dict[site][0]+domain_size,
                  site_coords_dict[site][1]-domain_size,
                  site_coords_dict[site][0]-domain_size,
                  site_coords_dict[site][1]+domain_size)


        try:
            c.retrieve(
                dataset,
                retrieve_dict(level, month, year, domain),
                output_filename)
        except Exception as e:
            print(f'Error downloading {site} {level}: {months[month]} {year}')
            print(e)
            
        print(f'{site} {level}: {months[month]} {year} downloaded')


def retrieve_site_year(level, site, year):
    """ Downloads meteorological data for a specific site and year.
    Just a wrapper for retrieve_site_month to run asynchronously.

    Args:
        level (str): 'pressure' or 'single'
        site (str): site code
        year (int): year

    Returns:
        None
    """

    # Run asynchronously. Seems to be a limit of 2 or 3 simultaneous requests?
    with Pool(2) as pool:
        pool.starmap(retrieve_site_month, [(site, level, year, month) for month in range(12)])


if __name__ == '__main__':

    error_log = root_path / "error_log.txt"

    # Redirect stdout and stderr to a log file
    sys.stdout = open(error_log, "w")

    for site in site_coords_dict.keys():
        for year in range(1978, 2024):
            retrieve_site_year("pressure", site, year)
            retrieve_site_year("single", site, year)

    sys.stdout.close()