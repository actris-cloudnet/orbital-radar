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

import logging

import numpy as np
import xarray as xr

from orbital_radar.helpers import db2li, li2db


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
        categorize_filepath : str
        """

        self.categorize_filepath = categorize_filepath
        self.ds = self._read_cloudnet()

        # ensure that vm is negative for upward motion (satellite convention)
        self.ds["vm"] = -self.ds["vm"]

    def convert_frequency(self) -> None:
        """
        Convert frequency from 35 to 94 GHz.

        The conversion is based on Kollias et al. (2019)
        (doi: https://doi.org/10.5194/amt-12-4949-2019)

        """

        logging.info("Converting frequency from 35 to 94 GHz")

        # keep only reflectivities below 30 dBZ
        self.ds["ze"] = self.ds["ze"].where(self.ds["ze"] < db2li(30))

        a = -16.8251
        b = 8.4923
        self.ds["ze"] = db2li(
            li2db(self.ds["ze"]) - 10**a * (li2db(self.ds["ze"]) + 100) ** b
        )

        # set negative ze to zero
        self.ds["ze"] = self.ds["ze"].where(self.ds["ze"] > 0.0, 0.0)

    def correct_dielectric_constant(
        self, satellite_k2: float, radar_k2: float
    ) -> None:
        r"""
        Apply correction for dielectric constant assumed in Ze calculation
        of suborbital radar to match the dielectric constant of the
        spaceborne radar.

        Correction equation with :math:`K_g` and :math:`K_s` as dielectric
        constants of the suborbital and spaceborne radar, respectively:

        .. math::
           Z_e = 10 \log_{10} \left( \frac{K_s}{K_g} \right) + Z_e

        Parameters
        ----------
        satellite_k2 : float
            Dielectric constant of the spaceborne radar.
        radar_k2 : float
            Dielectric constant of the suborbital radar.
        """

        msg = (
            f"Correcting dielectric constant {radar_k2} "
            f"to match spaceborne radar {satellite_k2}"
        )
        logging.debug(msg)

        correction = satellite_k2 / radar_k2

        self.ds["ze"] *= correction

    def create_along_track(self, mean_wind: float) -> None:
        """
        Creates along-track coordinates from time coordinates.

        Parameters
        ----------
        mean_wind : float
            Mean wind speed in m/s.

        """

        dt = self.ds.time.diff("time") / np.timedelta64(
            1, "s"
        )  # time difference [s]

        dt = xr.align(self.ds.time, dt, join="outer")[1].fillna(
            0
        )  # start is dt = 0

        arr_along_track = np.cumsum(mean_wind * dt)

        da_along_track = xr.DataArray(
            arr_along_track,
            dims="time",
            coords=[self.ds.time],
            name="along_track",
        )
        da_along_track.attrs = {
            "standard_name": "along_track_distance",
            "long_name": "Along track distance",
            "units": "m",
            "description": "Distance along track of the suborbital radar",
        }

        # swap from time to along track
        self.ds = self.ds.assign_coords(along_track=da_along_track)
        self.ds = self.ds.swap_dims({"time": "along_track"})

        # add time as variable
        self.ds = self.ds.reset_coords()

    def _read_cloudnet(self) -> xr.Dataset:
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

        ds_radar["ze"] = db2li(ds_radar["ze"])

        # set inf ze to nan
        ds_radar["ze"] = ds_radar["ze"].where(ds_radar["ze"] != np.inf)

        # set very low vm to nan
        ds_radar["vm"] = ds_radar["vm"].where(ds_radar["vm"] > -500)
        ds_radar["vm"] = ds_radar["vm"].where(ds_radar["vm"] < 500)

        return ds_radar
