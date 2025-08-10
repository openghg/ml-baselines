import numpy as np


def longitude_to_360(lons):
    """
    Convert longitudes to 0-360 range.
    """

    lons_out = np.where(lons < 0, lons + 360, lons)
    lons_out = np.where(lons_out >= 360, lons_out - 360, lons_out)

    return lons_out
