"""
This module contains the OrbitalRadar class that runs the simulator for
suborbital radar data. It is a subclass of the Simulator class.
"""

import datetime
import logging
import uuid as uuidlib
from dataclasses import dataclass

import netCDF4
import numpy as np
import xarray as xr

from orbital_radar.helpers import db2li, li2db
from orbital_radar.radar import Radar
from orbital_radar.simulator import Simulator


@dataclass
class SatelliteOptions:
    height_min: float = -2500.0
    height_max: float = 17500.0
    height_res: float = 10.0
    ground_echo_ze_max: float = 52.0
    ground_echo_pulse_length: float = 100.0


class Suborbital(Simulator):
    """
    Run the simulator for suborbital radar data.
    """

    def __init__(
        self,
        satellite_options: SatelliteOptions = SatelliteOptions(),
    ):
        """
        Initialize the simulator for suborbital radar data.

        Parameters
        ----------
        satellite_options : SatelliteOptions
            Options for the satellite radar.
        """

        self.satellite_options = satellite_options
        super().__init__()

    def simulate_cloudnet(
        self,
        categorize_filepath: str,
        output_filepath: str,
        mean_wind: float,
        uuid: str | None = None,
    ) -> str:
        """
        Runs simulation for a single day.

        Parameters
        ----------
        categorize_filepath : str
            Path to Cloudnet categorize file.
        output_filepath : str
            Path to output file.
        mean_wind : float
            Mean wind speed in m/s.
        uuid : str, optional
            UUID of the file.
        """

        radar = Radar(categorize_filepath)

        with xr.open_dataset(categorize_filepath) as ds_categorize:
            frequency = ds_categorize.radar_frequency.item()

            if np.isclose(frequency, 35, atol=1):
                radar.convert_frequency()

            # TODO: Check this. Maybe you want to do
            # this every time after frequency conversion?
            elif np.isclose(frequency, 94, atol=1):
                radar.correct_dielectric_constant(
                    satellite_k2=0.75, radar_k2=0.86
                )
            else:
                raise ValueError(f"Frequency {frequency} not supported")

            radar.create_along_track(mean_wind)

            self._interpolate_to_regular_grid(radar.ds)

            # TODO: Check this. Cloudnet categorize
            # is already attenuation corrected?!
            self._apply_gas_attenuation(ds_categorize)

        self._add_ground_echo()

        # run simulator
        self.transform()

        self._remove_duplicate_times()

        self._add_ground_based_variables(mean_wind)

        self._to_netcdf(output_filepath)

        file_uuid = self._harmonize_for_cloudnet(
            output_filepath, categorize_filepath, uuid
        )

        return file_uuid

    @staticmethod
    def _harmonize_for_cloudnet(
        filepath: str, source_filepath: str, uuid: str | None
    ) -> str:
        with (
            netCDF4.Dataset(filepath, "r+") as nc,
            netCDF4.Dataset(source_filepath, "r") as nc_source,
        ):
            if uuid is None:
                uuid = str(uuidlib.uuid4())
            nc.file_uuid = uuid

            # Format time units
            time = nc.variables["time"]
            time.units = time.units[:22] + " 00:00:00 +00:00"

            # Add offset to along_track stuff to match Cloudnet timestamps
            # TODO: there must be a way fix this earlier in the code
            time = time[:]
            n_hours_of_data = time[-1] - time[0]
            for key in ("along_track", "along_track_sat"):
                data = nc.variables[key][:]
                distance_covered = data[-1] - data[0]
                distance_per_hour = distance_covered / n_hours_of_data
                offset = distance_per_hour * time[0]
                nc.variables[key][:] += offset

            # Convert linear units to dB
            for var in ("ze_sat", "ze_sat_noise", "ze"):
                nc.variables[var][:] = li2db(nc.variables[var][:])
                nc.variables[var].units = "dBZ"

            # Change velocity direction to match Cloudnet
            for var in ("vm_sat", "vm_sat_noise", "vm_sat_folded"):
                nc.variables[var][:] *= -1

            # Remove unnecessary global attributes
            for attr in (
                "cloudnetpy_version",
                "pid",
            ):
                if attr in nc.ncattrs():
                    nc.delncattr(attr)

            file_type = "cpr-simulation"
            nc.cloudnet_file_type = file_type
            nc.location = nc_source.location
            nc.title = f"Simulated CPR radar from {nc_source.location}"
            nc.source = nc_source.variables["Z"].source
            nc.source_file_uuids = nc_source.file_uuid
            now = f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S +00:00}"
            nc.history = f"{now} - {file_type} file created\n" + nc.history
            nc.references = "https://doi.org/10.5194/gmd-18-101-2025"

        return uuid

    def _interpolate_to_regular_grid(self, radar_ds: xr.Dataset) -> None:
        """
        Interpolates radar data to regular grid in along-track and height.

        Parameters
        ----------
        radar_ds : xarray.Dataset
            Data with "time" and "height" coordinates.

        """

        regular_height = self._create_regular_height()
        regular_along_track = self._create_regular_along_track(radar_ds)

        self.ds = radar_ds.interp(
            along_track=regular_along_track,
            height=regular_height,
            method="nearest",
        )
        # get nearest time for each regular along track grid point
        self.ds.coords["time"] = (
            ("along_track"),
            radar_ds["time"]
            .sel(along_track=self.ds.along_track, method="nearest")
            .values,
        )

    def _create_regular_height(self) -> xr.DataArray:
        """
        Creates regular height coordinate for suborbital radar.

        Returns
        -------
        xarray.DataArray
            Regular height coordinate for suborbital radar.
        """

        regular_height = np.arange(
            self.satellite_options.height_min,
            self.satellite_options.height_max,
            self.satellite_options.height_res,
        )

        return xr.DataArray(
            regular_height, dims="height", coords=[regular_height]
        )

    @staticmethod
    def _create_regular_along_track(ds: xr.Dataset) -> xr.DataArray:
        """
        Creates regular along-track coordinate for suborbital radar.

        Parameters
        ----------
        ds : xarray.Dataset
            Data with "along_track" coordinate.

        Returns
        -------
        xarray.DataArray
            Regular along-track coordinate for suborbital radar.
        """

        along_track_res = np.round(
            ds.along_track.diff("along_track").median().item()
        )
        along_track_max = ds.along_track.max().item()

        along_track_regular = np.arange(
            0,
            along_track_max,
            along_track_res,
        )

        return xr.DataArray(
            along_track_regular,
            dims="along_track",
            coords=[along_track_regular],
        )

    def _add_ground_echo(self) -> None:
        """
        Calculates artificial ground echo inside ground-based radar range
        grid. The values are chosen such that the final ground echo after
        the along-range convolution is equal to the ground echo of the
        satellite. The pulse length used here is not the same as the pulse
        length of the satellite.

        """

        assert len(np.unique(np.diff(self.ds.height))) == 1, (
            "Height grid is not equidistant. "
            "Range weighting function cannot be calculated."
        )

        # grid with size of two pulse lengths centered around zero
        height_bins = np.arange(
            -self.satellite_options.ground_echo_pulse_length,
            self.satellite_options.ground_echo_pulse_length
            + self.satellite_options.height_res,
            self.satellite_options.height_res,
        )

        # calculate DEFAULT range weighting function
        # should this be EarthCARE CPR weighting function?
        weights = self._normalized_range_weighting_function_default(
            pulse_length=self.satellite_options.ground_echo_pulse_length,
            range_bins=height_bins,
        )

        ground_echo = weights * db2li(
            self.satellite_options.ground_echo_ze_max
        )

        # add ground echo to dataset shifted by one height bin to have maximum
        # below zero
        # get closest height bin to ground
        idx = (np.abs(self.ds.height - self.ds.altitude.item())).argmin()
        base = self.ds.height[idx].item()

        # insert half of the calculated ground echo and shift maximum below the
        # surface
        ground_echo = ground_echo[int(len(ground_echo) / 2) :]
        height_bins = (
            base
            + height_bins[int(len(height_bins) / 2) :]
            - self.satellite_options.height_res
        )

        # add ground echo to dataset (first fill nan values with zero in this
        # height interval)
        self.ds["ze"].loc[{"height": height_bins}] = (
            self.ds["ze"].loc[{"height": height_bins}].fillna(0)
        )
        self.ds["ze"].loc[{"height": height_bins}] += ground_echo

    @staticmethod
    def _normalized_range_weighting_function_default(
        pulse_length: float, range_bins: np.ndarray
    ) -> np.ndarray:
        """
        Defines the range weighting function for the along-range averaging.
        """
        # calculate along-range weighting function
        w_const = -(np.pi**2) / (2.0 * np.log(2) * pulse_length**2)
        range_weights = np.exp(w_const * range_bins**2)
        return range_weights / np.sum(range_weights)

    def _add_ground_based_variables(self, mean_wind: float) -> None:
        self.ds["mean_wind"] = xr.DataArray(
            mean_wind,
            attrs={
                "long_name": "Mean horizontal wind",
                "units": "m s-1",
                "description": "Mean horizontal wind",
            },
        )

    def _to_netcdf(self, output_filepath: str) -> None:
        """
        Writes dataset to netcdf file. Note that not all variables are stored.

        Parameters
        ----------
        output_filepath : str
            Path to the output file.

        """

        date = self.ds.time[0].dt.strftime("%Y-%m-%d").item()

        TIME = {
            "dtype": "float64",
            "_FillValue": None,
            "units": f"hours since {date} 00:00:00 +00:00",
        }
        FLOAT = {
            "dtype": "float32",
            "_FillValue": netCDF4.default_fillvals["f4"],
        }
        INT = {"dtype": "int32", "_FillValue": netCDF4.default_fillvals["i4"]}
        FLOAT_SCALAR = {"dtype": "float32", "_FillValue": None}

        variable_map = {
            "time": TIME,
            "sat_ifov": FLOAT_SCALAR,
            "sat_range_resolution": FLOAT_SCALAR,
            "sat_along_track_resolution": FLOAT_SCALAR,
            "ze": FLOAT,
            "vm": FLOAT,
            "ze_sat": FLOAT,
            "vm_sat": FLOAT,
            "vm_sat_vel": FLOAT,
            "ze_sat_noise": FLOAT,
            "vm_sat_noise": FLOAT,
            "vm_sat_folded": FLOAT,
            "nubf_flag": INT,
            "ms_flag": INT,
            "folding_flag": INT,
            "mean_wind": FLOAT_SCALAR,
            "latitude": FLOAT_SCALAR,
            "longitude": FLOAT_SCALAR,
            "altitude": FLOAT_SCALAR,
            "along_track": FLOAT,
            "height_sat": FLOAT_SCALAR,
            "along_track_sat": FLOAT,
            "height": FLOAT,
        }

        self.ds[variable_map.keys()].to_netcdf(  # type: ignore
            output_filepath,
            mode="w",
            encoding=variable_map,
            format="NETCDF4_CLASSIC",
        )
        logging.debug(f"Written file: {output_filepath}")

    def _apply_gas_attenuation(self, ds_categorize: xr.Dataset) -> None:
        """
        Gas attenuation correction based on Cloudnet categorize file.

        Parameters
        ----------
        ds_categorize : xarray.Dataset
            Cloudnet data with "gas_atten" variable. Unit: dBZ
        """

        # interpolate to radar height grid
        gas_atten = ds_categorize.radar_gas_atten.interp(
            height=self.ds.height, method="linear"
        )

        # interpolate to radar time grid and extrapolate if needed
        gas_atten = gas_atten.interp(
            time=self.ds.time,
            method="linear",
            kwargs={"fill_value": "extrapolate"},
        )

        # apply attenuation correction
        self.ds["ze"] *= db2li(gas_atten)

    def _remove_duplicate_times(self) -> None:
        _, index = np.unique(self.ds["time"], return_index=True)
        self.ds = self.ds.isel(along_track=index)
