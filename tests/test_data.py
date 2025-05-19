import pandas as pd
import numpy as np

from ml_baselines.data import read_intem


def test_read_intem():
    """
    Test function for read_intem
    """
    site = "MHD"
    df = read_intem(site)
    assert isinstance(df, pd.DataFrame), "Output is not a DataFrame"
    assert 'baseline' in df.columns, "Column 'baseline' not found in DataFrame"
    assert df['baseline'].dtype == np.int64, "Column 'baseline' is not of type int64"

    # 1st Jan 1999 is not baseline
    assert df.loc['1999-01-01 01:00', 'baseline'] == 0, "Baseline value for 1st Jan 1999 is incorrect"
    # 5pm on the 2nd Jan 2000 is not baseline
    assert df.loc['2000-01-02 17:00:00', 'baseline'] == 0, "Baseline value for 2nd Jan 2000 is incorrect"
    # 21st September 1990 at 7am is baseline
    assert df.loc['1990-09-21 07:00:00', 'baseline'] == 1, "Baseline value for 21st September 1990 is incorrect"

    # Test timestamps are in order and continuous
    assert df.index.is_monotonic_increasing, "Timestamps are not in increasing order"
    assert df.index.is_unique, "Timestamps are not unique"


    # Check outputting only one year
    df = read_intem(site, start_year=1990, end_year=1990)
    assert df.index.min().year == 1990, "Start year is incorrect"
    assert df.index.max().year == 1990, "End year is incorrect"
