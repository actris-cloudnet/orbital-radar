"""
This module contains helper functions for the orbital radar simulator.
"""

from typing import overload

import numpy as np
import xarray as xr


@overload
def db2li(x: xr.DataArray) -> xr.DataArray: ...


@overload
def db2li(x: float) -> float: ...


@overload
def db2li(x: np.ndarray) -> np.ndarray: ...


def db2li(
    x: xr.DataArray | float | np.ndarray,
) -> xr.DataArray | float | np.ndarray:
    """
    Conversion from dB to linear.

    Parameters
    ----------
    x :
        Any value or array to be converted from dB to linear unit
    """
    return 10 ** (0.1 * x)


@overload
def li2db(x: xr.DataArray) -> xr.DataArray: ...


@overload
def li2db(x: float) -> float: ...


@overload
def li2db(x: np.ndarray) -> np.ndarray: ...


def li2db(
    x: xr.DataArray | float | np.ndarray,
) -> xr.DataArray | float | np.ndarray:
    """
    Conversion from linear to dB.

    Parameters
    ----------
    x :
        Any value or array to be converted from linear to dB unit
    """
    epsilon = 1e-15
    return 10 * np.log10(x + epsilon)


def remove_duplicate_times(ds: xr.Dataset) -> xr.Dataset:
    _, index = np.unique(ds["time"], return_index=True)
    ds = ds.isel(along_track=index)
    return ds
