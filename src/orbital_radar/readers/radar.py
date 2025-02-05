"""
This script contains all reader functions for the different radar formats.
These functions are wrapped by the main function, which picks the correct
reader depending on the radar site.

The final output is always an xarray.Dataset with the two variables radar
reflectivity "ze" in [mm6 m-3] and "vm" in [m s-1] as a function of range and
time.

The input Doppler velocity should be negative for downward motion and positive
for upward motion. This is changed to negative upward and positive downward to
match spaceborne convention.
"""

import os
import os.path
from glob import glob

import numpy as np
import xarray as xr


class Radar:
    """
    This class selects the reading function for the provided site and
    performs quality-checks of the imported data. The output contains "ze"
    and "vm" variables.
    """

    def __init__(
        self,
        radar_filepath: str,
        categorize_filepath: str | None = None,
    ) -> None:
        """
        Reads radar data with these standardized output variables:
        - Radar reflectivity (ze) in mm6 m-3
        - Mean Doppler velocity (vm) in m s-1
        - Range as height above NN (range) in m
        - Time (time)

        Parameters
        ----------
        radar_filepath : str
        categorize_filepath : str
        """

        self.radar_filepath = radar_filepath
        self.categorize_filepath = categorize_filepath
        self.ds_rad = xr.Dataset()

        self.read_cloudnet()

        print("Vm sign convention: negative=upward, " "positive=downward")

        self.ds_rad["vm"] = -self.ds_rad["vm"]

        print(f"Quality checks for radar data.")

        # ensure ze and vm variables exist
        assert "ze" in list(self.ds_rad)
        assert "vm" in list(self.ds_rad)

        # ensure same dimension order
        if "height" in list(self.ds_rad.dims):
            dim_order = ["time", "height"]
        else:
            dim_order = ["time", "range"]
        self.ds_rad["ze"] = self.ds_rad.ze.transpose(*dim_order)
        self.ds_rad["vm"] = self.ds_rad.vm.transpose(*dim_order)

        # ensure reasonable value ranges
        assert (
            self.ds_rad.ze.isnull().all() or self.ds_rad.ze.min() >= 0
        ), "Ze out of range."
        assert self.ds_rad.ze.isnull().all() or (
            10 * np.log10(self.ds_rad.ze.max()) < 100
        ), "Ze out of range."

        assert (
            self.ds_rad.vm.isnull().all() or self.ds_rad.vm.min() > -80
        ), "Vm values out of range."
        assert (
            self.ds_rad.vm.isnull().all() or self.ds_rad.vm.max() < 80
        ), "Vm values out of range."

        # make sure that alt is in the data
        assert "alt" in list(self.ds_rad), "Altitude not found."

    def read_cloudnet(self):
        """
        Reads radar reflectivity and Doppler velocity from Cloudnet categorize
        files.

        Note: Cloudnet height is already in height above mean sea level.
        """

        if self.categorize_filepath is None:
            raise FileNotFoundError("No categorize file found.")

        ds = xr.open_dataset(self.categorize_filepath)

        ds = ds.rename({"Z": "ze", "v": "vm"})

        self.ds_rad = ds[["ze", "vm"]]

        # extract instrument location and altitude
        self.ds_rad["lon"] = ds["longitude"]
        self.ds_rad["lat"] = ds["latitude"]
        self.ds_rad["alt"] = ds["altitude"]

        # convert from dB to linear units
        self.ds_rad["ze"] = 10 ** (0.1 * self.ds_rad["ze"])

        # set inf ze to nan
        self.ds_rad["ze"] = self.ds_rad["ze"].where(
            self.ds_rad["ze"] != np.inf
        )

        # set very low vm to nan
        self.ds_rad["vm"] = self.ds_rad["vm"].where(self.ds_rad["vm"] > -500)

        self.ds_rad["vm"] = self.ds_rad["vm"].where(self.ds_rad["vm"] < 500)
