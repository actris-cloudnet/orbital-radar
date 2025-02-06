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
        categorize_filepath: str,
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

        self.categorize_filepath = categorize_filepath
        self.ds_rad = self.read_cloudnet()

        # print("Vm sign convention: negative=upward, " "positive=downward")

        self.ds_rad["vm"] = -self.ds_rad["vm"]

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

    def read_cloudnet(self) -> xr.Dataset:
        """
        Reads radar reflectivity and Doppler velocity from Cloudnet categorize
        files.

        Note: Cloudnet height is already in height above mean sea level.
        """

        ds = xr.open_dataset(self.categorize_filepath)

        ds = ds.rename({"Z": "ze", "v": "vm"})
        # ds = self.remove_duplicate_times(ds)

        ds_radar = ds[["ze", "vm"]]

        # extract instrument location and altitude
        ds_radar["longitude"] = ds["longitude"]
        ds_radar["latitude"] = ds["latitude"]
        ds_radar["altitude"] = ds["altitude"]

        # convert from dB to linear units
        ds_radar["ze"] = 10 ** (0.1 * ds_radar["ze"])

        # set inf ze to nan
        ds_radar["ze"] = ds_radar["ze"].where(ds_radar["ze"] != np.inf)

        # set very low vm to nan
        ds_radar["vm"] = ds_radar["vm"].where(ds_radar["vm"] > -500)
        ds_radar["vm"] = ds_radar["vm"].where(ds_radar["vm"] < 500)

        return ds_radar
