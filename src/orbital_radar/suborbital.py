"""
This module contains the OrbitalRadar class that runs the simulator for
suborbital radar data. It is a subclass of the Simulator class.

Difference between ground-based and airborne radar geometry:
- along-track coordinate from mean wind for ground based and mean flight vel.
from airborne radar
- no ground echo added to airborne radar
- range grid of airborne radar assumed to be height above mean sea level and
height above ground for groundbased
- no attenuation correction for airborne radar
- lat/lon coordinates included as input to airborne radar
"""

import logging
import pathlib

import numpy as np
import xarray as xr

from orbital_radar.helpers import db2li, li2db
from orbital_radar.radarspec import RadarBeam
from orbital_radar.readers.config import read_config
from orbital_radar.readers.radar import Radar
from orbital_radar.simulator import Simulator
from orbital_radar.version import __version__

# from orbital_radar.writers.spaceview import write_spaceview


class Suborbital(Simulator):
    """
    Run the simulator for suborbital radar data.
    """

    def __init__(
        self,
        geometry: str = "groundbased",
        input_radar_format: str = "cloudnet",
        config_file: str | None = None,
    ):
        """
        Initialize the simulator for suborbital radar data.

        Parameters
        ----------
        geometry : str
            Observation geometry of radar (groundbased or airborne).
        input_radar_format : str
            Format of the input radar data (e.g. cloudnet).
        config_file : str
            Path to the configuration file that contains the site-dependent
            parameters and directory paths.
        """

        # set class attributes
        self.geometry = geometry
        self.input_radar_format = input_radar_format

        if config_file:
            self.config = read_config(config_file)
        else:
            file_path = pathlib.Path(__file__).parent.absolute()
            self.config = read_config(file_path / "orbital_radar_config.toml")

        # preparation of input radar data
        self.prepare = self.config["prepare"]["general"]
        self.prepare.update(self.config["prepare"][self.geometry])

        file_path = pathlib.Path(__file__).parent.absolute()
        cpr_file = file_path / "data/CPR_PointTargetResponse.txt"

        super().__init__(file_earthcare=cpr_file)

    def convert_frequency(self, ds: xr.Dataset):
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

    def correct_dielectric_constant(
        self, ds: xr.Dataset, satellite_k2: float, radar_k2: float
    ):
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
        """

        logging.info(
            f"Correcting dielectric constant {radar_k2} to match spaceborne radar {satellite_k2}"
        )

        correction = satellite_k2 / radar_k2

        ds["ze"] = db2li(li2db(ds["ze"]) + 10 * np.log10(correction))

        return ds

    def add_vmze_attrs(self, ds):
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

    def range_to_height(self, ds):
        """
        Convert range coordinate to height coordinate by adding the station
        height above mean sea level to the range coordinate.

        The altitude is pre-defined for each station in the configuration file.

        Parameters
        ----------
        ds : xarray.Dataset
            Data with "range" coordinate.
        """

        ds["height"] = ds["range"] + ds.alt.item()

        # swap range with height
        ds = ds.swap_dims({"range": "height"})

        # drop range coordinate
        ds = ds.reset_coords(drop=True)

        return ds

    def create_along_track(self, ds):
        """
        Creates along-track coordinates from time coordinates.

        Parameters
        ----------
        ds : xarray.Dataset
            Data with "time" and "height" coordinates.

        Returns
        -------
        ds : xarray.Dataset
            Data with "along_track" coordinate.
        """

        if self.geometry == "groundbased":
            # print("Using mean wind for along-track coordinates")
            v = self.prepare["mean_wind"]

        elif self.geometry == "airborne" and "ac_speed" in list(ds):
            # print("Using flight velocity for along-track coordinates")
            v = ds.ac_speed.values

        elif self.geometry == "airborne" and "ac_speed" not in list(ds):
            # print("Using mean flight velocity for along-track coordinates")
            v = self.prepare["mean_flight_velocity"]

        else:
            raise ValueError

        # calculate the along-track distance
        dt = ds.time.diff("time") / np.timedelta64(1, "s")
        dt = xr.align(ds.time, dt, join="outer")[1].fillna(0)  # start is dt=0
        arr_along_track = np.cumsum(v * dt)

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

    def create_regular_height(self):
        """
        Creates regular height coordinate for suborbital radar.

        Returns
        -------
        xarray.DataArray
            Regular height coordinate for suborbital radar.
        """

        height_regular = np.arange(
            self.prepare["height_min"],
            self.prepare["height_max"],
            self.prepare["height_res"],
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

    def create_regular_along_track(self, ds):
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

    def interpolate_to_regular_grid(self, ds):
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

        da_height_regular = self.create_regular_height()
        da_along_track_regular = self.create_regular_along_track(ds=ds)

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
        ds = self.add_vmze_attrs(ds)

        return ds

    def add_ground_echo(self, ds):
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
            -self.prepare["ground_echo_pulse_length"],
            self.prepare["ground_echo_pulse_length"]
            + self.prepare["height_res"],
            self.prepare["height_res"],
        )

        # calculate range weighting function
        weights = RadarBeam.normalized_range_weighting_function_default(
            pulse_length=self.prepare["ground_echo_pulse_length"],
            range_bins=height_bins,
        )

        ground_echo = weights * db2li(self.prepare["ground_echo_ze_max"])

        # add ground echo to dataset shifted by one height bin to have maximum
        # below zero
        # get closest height bin to ground
        idx = (np.abs(ds.height - ds.alt.item())).argmin()
        base = ds.height[idx].item()

        # insert half of the calculated ground echo and shift maximum below the
        # surface
        ground_echo = ground_echo[int(len(ground_echo) / 2) :]
        height_bins = (
            base
            + height_bins[int(len(height_bins) / 2) :]
            - self.prepare["height_res"]
        )

        # add ground echo to dataset (first fill nan values with zero in this
        # height interval)
        ds["ze"].loc[{"height": height_bins}] = (
            ds["ze"].loc[{"height": height_bins}].fillna(0)
        )
        ds["ze"].loc[{"height": height_bins}] += ground_echo

        return ds

    def add_groundbased_variables(self):
        """
        Add variables specific to groundbased simulator to the dataset, i.e.,
        the mean horizontal wind.
        """

        self.ds["mean_wind"] = xr.DataArray(
            self.prepare["mean_wind"],
            attrs=dict(
                standard_name="v_hor",
                long_name="Mean horizontal wind",
                units="m s-1",
                description="Mean horizontal wind",
            ),
        )

    def to_netcdf(self, output_filepath: str):
        """
        Writes dataset to netcdf file. Note that not all variables are stored.

        Parameters
        ----------
        output_filepath : str
            Path to the output file.

        """

        output_variables = [
            "sat_ifov",
            "sat_range_resolution",
            "sat_along_track_resolution",
            "ze",
            "vm",
            "ze_sat",
            "vm_sat",
            "vm_sat_vel",
            "ze_sat_noise",
            "vm_sat_noise",
            "vm_sat_folded",
            "nubf_flag",
            "ms_flag",
            "folding_flag",
        ]

        if self.geometry == "groundbased":
            output_variables += ["mean_wind"]

        if self.geometry == "airborne":
            output_variables += ["mean_flight_velocity"]

        self.ds[output_variables].to_netcdf(output_filepath, mode="w")
        logging.info(f"Written file: {output_filepath}")

    @staticmethod
    def add_attenuation(ds: xr.Dataset, attenuation: xr.DataArray):
        """
        Add attenuation to dataset.

        Parameters
        ----------
        ds : xarray.Dataset
            Data with "ze" variable. Unit: mm6 m-3
        attenuation : xarray.DataArray
            Interpolated attenuation data on the same grid as ds.
            Unit: dBZ.

        Returns
        -------
        ds : xarray.Dataset
            Data with added attenuation.
        """

        # add attenuation in dB and convert back to
        ds["ze"] = db2li(li2db(ds["ze"]) + attenuation)

        return ds

    def attenuation_correction(self, ds: xr.Dataset, ds_cloudnet: xr.Dataset):
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
        ds = self.add_attenuation(ds=ds, attenuation=gas_atten)

        return ds

    def simulate_cloudnet(
        self,
        categorize_filepath: str,
        output_filepath: str,
    ):
        """
        Runs simulation for a single day.

        Parameters
        ----------
        date : np.datetime64
            Date to simulate.
        write_output : bool
            If True, write output to netcdf file.
        """

        radar = Radar(categorize_filepath)

        with xr.open_dataset(categorize_filepath) as ds_categorize:
            frequency = ds_categorize.radar_frequency.item()

            if np.isclose(frequency, 35, atol=1):
                radar.ds_rad = self.convert_frequency(radar.ds_rad)
            elif np.isclose(frequency, 94, atol=1):
                radar.ds_rad = self.correct_dielectric_constant(
                    radar.ds_rad, satellite_k2=0.75, radar_k2=0.86
                )
            else:
                raise NotImplementedError("Frequency not supported")

            radar.ds_rad = self.create_along_track(ds=radar.ds_rad)

            ds = self.interpolate_to_regular_grid(radar.ds_rad)

            # TODO: Check this. Cloudnet categorize is already attenuation corrected?
            ds = self.attenuation_correction(ds, ds_categorize)

        ds = self.add_ground_echo(ds)

        # run simulator
        self.transform(ds)

        self.add_groundbased_variables()

        self.to_netcdf(output_filepath)
