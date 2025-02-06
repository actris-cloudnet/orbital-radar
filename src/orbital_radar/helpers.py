"""
This module contains helper functions for the orbital radar simulator.
"""

import numpy as np
import xarray as xr


def db2li(x):
    """
    Conversion from dB to linear.

    Parameters
    ----------
    x : float
        Any value or array to be converted from dB to linear unit
    """
    return 10 ** (0.1 * x)


def li2db(x, epsilon=1e-15):
    """
    Conversion from linear to dB.

    Parameters
    ----------
    x : float
        Any value or array to be converted from linear to dB unit
    """
    return 10 * np.log10(x + epsilon)


def remove_duplicate_times(ds: xr.Dataset) -> xr.Dataset:
    _, index = np.unique(ds["time"], return_index=True)
    ds = ds.isel(along_track=index)
    return ds
