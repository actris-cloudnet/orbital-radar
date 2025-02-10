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
        self.ds = self.read_cloudnet()

        # ensure that vm is negative for upward motion (satellite convention)
        self.ds["vm"] = -self.ds["vm"]

    def read_cloudnet(self) -> xr.Dataset:
        """
        Reads radar reflectivity and Doppler velocity from Cloudnet categorize
        files.

        Note: Cloudnet height is already in height above mean sea level.
        """

        ds_categorize = xr.open_dataset(self.categorize_filepath)

        ds_categorize = ds_categorize.rename({"Z": "ze", "v": "vm"})

        ds_radar = ds_categorize[["ze", "vm"]]

        # extract instrument location and altitude
        ds_radar["longitude"] = ds_categorize["longitude"]
        ds_radar["latitude"] = ds_categorize["latitude"]
        ds_radar["altitude"] = ds_categorize["altitude"]

        # convert from dB to linear units
        ds_radar["ze"] = 10 ** (0.1 * ds_radar["ze"])

        # set inf ze to nan
        ds_radar["ze"] = ds_radar["ze"].where(ds_radar["ze"] != np.inf)

        # set very low vm to nan
        ds_radar["vm"] = ds_radar["vm"].where(ds_radar["vm"] > -500)
        ds_radar["vm"] = ds_radar["vm"].where(ds_radar["vm"] < 500)

        return ds_radar
