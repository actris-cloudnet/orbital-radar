"""
This module contains helper functions for the orbital radar simulator.
"""

from pathlib import Path
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


def read_range_weighting_function(file: Path) -> xr.Dataset:
    """
    Reads EarthCARE CPR range weighting function. The pulse length factor
    is reversed to match the sign convention of the groundbased radar.

    Parameters
    ----------
    file : Path
        Path to file containing weighting function

    Returns
    -------
    wf : xarray.Dataset
        Weighting function
    """

    wf = np.loadtxt(file)

    ds_wf = xr.Dataset()
    ds_wf.coords["tau_factor"] = -wf[:, 0]
    ds_wf["response"] = ("tau_factor", wf[:, 1])

    return ds_wf
