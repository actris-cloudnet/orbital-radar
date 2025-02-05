"""
This script contains functions to read cloudnet data.
"""

import os
from glob import glob

import numpy as np
import pandas as pd
import xarray as xr

FILENAMES = {
    "cloudnet_ecmwf": "ecmwf",
    "cloudnet_categorize": "categorize",
}


def read_cloudnet_attenuation_file(filepath, date):
    """
    Reads Cloudnet data.

    The following file naming is expected (e.g. for 2022-02-14 at Mindelo):
    20220214_mindelo_ecmwf.nc
    20220214_mindelo_categorize.nc

    Parameters
    ----------
    filepath: str
        Cloudnet product to read. Either 'categorize' or 'ecmwf'.
    date: np.datetime64
        Date for which data is read.

    Returns
    -------
    ds: xarray.Dataset
        Cloudnet data.
    """

    print(f"Reading {filepath} Cloudnet data")

    # model_time unit for older cloudnetpy versions in bad format
    if filepath == "cloudnet_categorize":
        ds = xr.open_dataset(filepath, decode_times=False)

        if (
            ds["model_time"].units == "decimal hours since midnight"
            or ds["model_time"].units == f"hours since {str(date)} +00:00"
        ):
            # model time
            ds = convert_time(
                ds=ds,
                time_variable="model_time",
                base_time=np.datetime64(date),
                factor=60 * 60 * 1e9,
            )

            # radar time
            ds = convert_time(
                ds=ds,
                time_variable="time",
                base_time=np.datetime64(date),
                factor=60 * 60 * 1e9,
            )

        # make sure that difference between first and last time is more than 12 h
        if (
            ds["model_time"].values[-1] - ds["model_time"].values[0]
        ) < np.timedelta64(12, "h"):
            print(
                f"Warning: The time difference between the first and last time "
                f"step is less than 12 hours for {date}. "
                f"Check if time format is being read correctly."
            )

            return None

        if (ds["time"].values[-1] - ds["time"].values[0]) < np.timedelta64(
            12, "h"
        ):
            print(
                f"Warning: The time difference between the first and last time "
                f"step is less than 12 hours for {date}. "
                f"Check if time format is being read correctly."
            )

            return None

    # problem did not occur for ecmwf data
    else:
        ds = xr.open_dataset(filepath)

    return ds


def convert_time(ds, time_variable, base_time, factor=1):
    """
    Convert time in seconds since base_time to datetime64.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing the time variable.
    time_variable : str
        Name of the time variable.
    base_time : str
        Base time as string (e.g. "1970-01-01")
    factor : float, optional
        Factor to convert time to nanoseconds. Default is 1.
    """

    ds[time_variable] = (ds[time_variable] * factor).astype(
        "timedelta64[ns]"
    ) + np.datetime64(base_time)

    return ds
