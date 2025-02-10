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

from orbital_radar.helpers import db2li, li2db, remove_duplicate_times
from orbital_radar.radarspec import RadarBeam
from orbital_radar.readers.radar import Radar
from orbital_radar.simulator import Simulator
from orbital_radar.version import __version__


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
        uuid: uuidlib.UUID | None = None,
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
        """

        radar = Radar(categorize_filepath)

        with xr.open_dataset(categorize_filepath) as ds_categorize:
            frequency = ds_categorize.radar_frequency.item()

            if np.isclose(frequency, 35, atol=1):
                radar.ds = self._convert_frequency(radar.ds)

            # TODO: Check this. Maybe you want to do this every time after frequency conversion?
            elif np.isclose(frequency, 94, atol=1):
                radar.ds = self._correct_dielectric_constant(
                    radar.ds, satellite_k2=0.75, radar_k2=0.86
                )

            else:
                raise ValueError(f"Frequency {frequency} not supported")

            radar.ds = self._create_along_track(radar.ds, mean_wind)

            ds = self._interpolate_to_regular_grid(radar.ds)

            # TODO: Check this. Cloudnet categorize is already attenuation corrected?!
            ds = self._apply_gas_attenuation(ds, ds_categorize)

        ds = self._add_ground_echo(ds)

        # run simulator
        self.transform(ds)

        self.ds = remove_duplicate_times(self.ds)

        self._add_ground_based_variables(mean_wind)

        self._to_netcdf(output_filepath)

        file_uuid = self._harmonize_for_cloudnet(
            output_filepath, categorize_filepath, uuid
        )

        return file_uuid

    @staticmethod
    def _harmonize_for_cloudnet(
        filepath: str, source_filepath: str, uuid: uuidlib.UUID | None
    ) -> str:
        with (
            netCDF4.Dataset(filepath, "r+") as nc,
            netCDF4.Dataset(source_filepath, "r") as nc_source,
        ):
            if uuid is None:
                uuid = uuidlib.uuid4()
            nc.file_uuid = str(uuid)

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
            for var in ("ze_sat",):
                nc.variables[var][:] = li2db(nc.variables[var][:])
                nc.variables[var].units = "dBZ"

            # Change velocity direction to match Cloudnet
            for var in ("vm_sat", "vm_sat_noise", "vm_sat_folded"):
                nc.variables[var][:] *= -1

            # Remove unnecessary global attributes
            for attr in (
                "cloudnetpy_version",
                "pid",
                "location",
                "title",
                "references",
                "source",
            ):
                if attr in nc.ncattrs():
                    nc.delncattr(attr)

            file_type = "earthcare-simulation"
            nc.cloudnet_file_type = file_type
            nc.location = nc_source.location
            nc.title = f"Simulated EarthCARE radar from {nc_source.location}"
            nc.source = nc_source.variables["Z"].source
            nc.source_file_uuids = nc_source.file_uuid
            nc.history = (
                f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S +00:00} - {file_type} file created\n"
                + nc.history
            )

        return str(uuid)

    def _convert_frequency(self, ds: xr.Dataset) -> xr.Dataset:
        """
        Convert frequency from 35 to 94 GHz.

        The conversion is based on Kollias et al. (2019)
        (doi: https://doi.org/10.5194/amt-12-4949-2019)

        Parameters
        ----------
        ds : xarray.Dataset
            Data with "ze" variable in mm6/mm3. Ze was measured at 35 GHz.

        Returns
        -------
        ds : xarray.Dataset
            Data with converted "ze" variable. Ze is now transformed to 94 GHz.
        """

        logging.info("Converting frequency from 35 to 94 GHz")

        # keep only reflectivities below 30 dBZ
        ds["ze"] = ds["ze"].where(ds["ze"] < db2li(30))

        a = -16.8251
        b = 8.4923
        ds["ze"] = db2li(
            li2db(ds["ze"]) - 10**a * (li2db(ds["ze"]) + 100) ** b
        )

        # set negative ze to zero
        ds["ze"] = ds["ze"].where(ds["ze"] > 0.0, 0.0)

        return ds

    def _correct_dielectric_constant(
        self, ds: xr.Dataset, satellite_k2: float, radar_k2: float
    ) -> xr.Dataset:
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
        ds : xarray.Dataset
            Data with "ze" variable.
        satellite_k2 : float
            Dielectric constant of the spaceborne radar.
        radar_k2 : float
            Dielectric constant of the suborbital radar.
        """

        logging.info(
            f"Correcting dielectric constant {radar_k2} to match spaceborne radar {satellite_k2}"
        )

        correction = satellite_k2 / radar_k2

        ds["ze"] = db2li(li2db(ds["ze"]) + 10 * np.log10(correction))

        return ds

    def _add_vmze_attrs(self, ds: xr.Dataset) -> xr.Dataset:
        """
        Adds attributes to Doppler velocity and radar reflectivity variables.

        Parameters
        ----------
        ds : xarray.Dataset
            Data with "ze" and "vm" variables.

        Returns
        -------
        ds : xarray.Dataset
            Data with added attributes.
        """

        ds["ze"].attrs = dict(
            units="mm6 m-3",
            standard_name="radar_reflectivity",
            long_name="Radar reflectivity",
            description="Radar reflectivity",
        )

        ds["vm"].attrs = dict(
            units="m s-1",
            standard_name="Doppler_velocity",
            long_name="Doppler velocity",
            description="Doppler velocity",
        )

        return ds

    def _create_along_track(
        self, ds: xr.Dataset, mean_wind: float
    ) -> xr.Dataset:
        """
        Creates along-track coordinates from time coordinates.

        Parameters
        ----------
        ds : xarray.Dataset
            Data with "time" and "height" coordinates.
        mean_wind : float
            Mean wind speed in m/s.

        Returns
        -------
        ds : xarray.Dataset
            Data with "along_track" coordinate.
        """

        dt = ds.time.diff("time") / np.timedelta64(
            1, "s"
        )  # time difference [s]

        dt = xr.align(ds.time, dt, join="outer")[1].fillna(
            0
        )  # start is dt = 0

        arr_along_track = np.cumsum(mean_wind * dt)

        da_along_track = xr.DataArray(
            arr_along_track, dims="time", coords=[ds.time], name="along_track"
        )
        da_along_track.attrs = dict(
            standard_name="along_track_distance",
            long_name="Along track distance",
            units="m",
            description="Distance along track of the suborbital radar",
        )

        # swap from time to along track
        ds = ds.assign_coords(along_track=da_along_track)
        ds = ds.swap_dims({"time": "along_track"})

        # add time as variable
        ds = ds.reset_coords()

        return ds

    def _create_regular_height(self) -> xr.DataArray:
        """
        Creates regular height coordinate for suborbital radar.

        Returns
        -------
        xarray.DataArray
            Regular height coordinate for suborbital radar.
        """

        height_regular = np.arange(
            self.satellite_options.height_min,
            self.satellite_options.height_max,
            self.satellite_options.height_res,
        )

        da_height_regular = xr.DataArray(
            height_regular, dims="height", coords=[height_regular]
        )
        da_height_regular.attrs = dict(
            units="m",
            standard_name="height",
            long_name="Height of radar bin above sea level",
            description="Height of radar bin above sea level",
        )

        return da_height_regular

    def _create_regular_along_track(self, ds: xr.Dataset) -> xr.DataArray:
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

        da_along_track_regular = xr.DataArray(
            along_track_regular,
            dims="along_track",
            coords=[along_track_regular],
        )
        da_along_track_regular.attrs = ds.along_track.attrs

        return da_along_track_regular

    def _interpolate_to_regular_grid(self, ds: xr.Dataset) -> xr.Dataset:
        """
        Interpolates radar data to regular grid in along-track and height.

        Parameters
        ----------
        ds : xarray.Dataset
            Data with "time" and "height" coordinates.

        Returns
        -------
        ds : xarray.Dataset
            Data with interpolated "along_track" and "height" coordinates.
        """

        da_height_regular = self._create_regular_height()
        da_along_track_regular = self._create_regular_along_track(ds=ds)

        # interpolation along-track and height
        # workaround for time: convert to seconds since start, then
        # interpolate, then convert back to datetime
        da_time = ds["time"]
        ds = ds.interp(
            along_track=da_along_track_regular,
            height=da_height_regular,
            method="nearest",
        )
        # get nearest time for each regular along track grid point
        ds.coords["time"] = (
            ("along_track"),
            da_time.sel(along_track=ds.along_track, method="nearest").values,
        )

        # add attributes
        ds = self._add_vmze_attrs(ds)

        return ds

    def _add_ground_echo(self, ds: xr.Dataset) -> xr.Dataset:
        """
        Calculates artificial ground echo inside ground-based radar range
        grid. The values are chosen such that the final ground echo after
        the along-range convolution is equal to the ground echo of the
        satellite. The pulse length used here is not the same as the pulse
        length of the satellite.

        Parameters
        ----------
        ds : xarray.Dataset
            Data with "ze" and "vm" variables.

        Returns
        -------
        ds : xarray.Dataset
            Data with added ground echo.
        """

        assert len(np.unique(np.diff(ds.height))) == 1, (
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

        # calculate range weighting function
        weights = RadarBeam._normalized_range_weighting_function_default(
            pulse_length=self.satellite_options.ground_echo_pulse_length,
            range_bins=height_bins,
        )

        ground_echo = weights * db2li(
            self.satellite_options.ground_echo_ze_max
        )

        # add ground echo to dataset shifted by one height bin to have maximum
        # below zero
        # get closest height bin to ground
        idx = (np.abs(ds.height - ds.altitude.item())).argmin()
        base = ds.height[idx].item()

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
        ds["ze"].loc[{"height": height_bins}] = (
            ds["ze"].loc[{"height": height_bins}].fillna(0)
        )
        ds["ze"].loc[{"height": height_bins}] += ground_echo

        return ds

    def _add_ground_based_variables(self, mean_wind: float) -> None:
        self.ds["mean_wind"] = xr.DataArray(
            mean_wind,
            attrs=dict(
                long_name="Mean horizontal wind",
                units="m s-1",
                description="Mean horizontal wind",
            ),
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
            output_filepath, mode="w", encoding=variable_map
        )
        logging.debug(f"Written file: {output_filepath}")

    def _apply_gas_attenuation(
        self, ds: xr.Dataset, ds_cloudnet: xr.Dataset
    ) -> xr.Dataset:
        """
        Gas attenuation correction based on Cloudnet categorize file.

        Parameters
        ----------
        ds : xarray.Dataset
            Data with "ze" variable. Unit: mm6 m-3
        ds_cloudnet : xarray.Dataset
            Cloudnet data with "gas_atten" variable. Unit: dBZ

        Returns
        -------
        ds : xarray.Dataset
            Data with added attenuation.
        """

        # interpolate to radar height grid
        gas_atten = ds_cloudnet.radar_gas_atten.interp(
            height=ds.height, method="linear"
        )

        # interpolate to radar time grid and extrapolate if needed
        gas_atten = gas_atten.interp(
            time=ds.time, method="linear", kwargs={"fill_value": "extrapolate"}
        )

        # ensures every time step contains attenuation in range column
        has_atten = ~gas_atten.isnull().all("height")
        assert has_atten.all()

        # fill nan values with zero
        gas_atten = gas_atten.fillna(0)

        # apply attenuation correction
        ds["ze"] = db2li(li2db(ds["ze"]) + gas_atten)

        return ds
