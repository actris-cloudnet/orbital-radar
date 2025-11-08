"""
Runs the orbital radar simulator.
"""

import numpy as np
import xarray as xr
from scipy import stats
from scipy.interpolate import interp1d

from orbital_radar.helpers import db2li, li2db
from orbital_radar.radarspec import RadarBeam


class Simulator:
    """
    Runs the orbital radar simulator.
    """

    def __init__(self) -> None:
        """
        Initialize the simulator class. The input dataset will be extended with
        intermediate simulation steps.

        To run the simulator:
        - initialize the class
        - run the transform method

        Requirement:
        - all nan values should be filled with zeros
        - along-track and height coordinates should be monotonic increasing,
          evenly spaced, and multiples of the satellite resolution to ensure
          that each satellite bin contains the same number of high-resolution
          bins (e.g. 0, 100, 200, 300... --> 0, 500, 1000)

        Parameters
        ----------
        """
        self.ds: xr.Dataset
        self.beam = RadarBeam()

    def transform(self) -> None:
        """
        Runs the entire simulator.
        """

        self._check_input_dataset()
        self._prepare_input_dataset()

        self.beam.calculate_weighting_functions(
            range_coords=self.ds["height"],
            along_track_coords=self.ds["along_track"],
        )

        self._apply_detection_limit(var_ze="ze", var_other=["ze", "vm"])

        self._convolve_along_track()
        self._integrate_along_track()
        self._convolve_height()
        self._apply_detection_limit(
            var_ze="ze_sat", var_other=["ze_sat", "vm_sat", "vm_sat_vel"]
        )
        self._calculate_ze_noise()
        self._calculate_vm_noise()
        self._fold_vm()
        self._calculate_nubf()
        self._calculate_nubf_flag()
        self._calculate_vm_bias()
        self._calculate_vm_bias_flag()
        self._calculate_ms_flag()
        self._calculate_signal_fraction()
        self._add_attributes()

    def _check_input_dataset(self) -> None:
        """
        Check user input for consistency.
        """

        # make sure that dimensions are named correctly
        assert self.ds["ze"].dims == ("along_track", "height")

        # make sure that radar reflectivity is in linear units
        assert self.ds["ze"].min() >= 0

        # check if satellite resolution is a multiple of the range resolution
        assert (
            self.beam.spec.range_resolution
            % self.ds["height"].diff("height")[0]
            == 0
        ), (
            f"Height resolution not a multiple of the satellite resolution: "
            f"{self.ds['height'].diff('height')[0]} m"
        )

        # check if range resolution is smaller or equal to satellite resolution
        assert (
            self.ds["height"].diff("height")[0]
            <= self.beam.spec.range_resolution
        ), (
            f"Range resolution is larger than the satellite resolution: "
            f"{self.ds['height'].diff('height')[0]} m"
        )

    def _prepare_input_dataset(self) -> None:
        """
        Prepares input dataset for computations. This only includes replacing
        nan values by zero in both ze and vm.
        """

        self.ds = self.ds.fillna(0)
        assert not self.ds["ze"].isnull().any()
        assert not self.ds["vm"].isnull().any()

    def calculate_along_track_sat_bin_edges(self) -> np.ndarray:
        """
        Calculate the bin edges of the along-track satellite grid. This way
        is equivalent to height.
        """

        return np.append(
            self.ds["along_track_sat"]
            - self.beam.spec.along_track_resolution / 2,
            self.ds["along_track_sat"][-1]
            + self.beam.spec.along_track_resolution / 2,
        )

    def calculate_height_sat_bin_edges(self) -> np.ndarray:
        """
        Calculate the bin edges of the height satellite grid. This way is
        equivalent to along-track.
        """

        return np.append(
            self.ds["height_sat"] - self.beam.spec.range_resolution / 2,
            self.ds["height_sat"][-1] + self.beam.spec.range_resolution / 2,
        )

    def _convolve_along_track(self) -> None:
        """
        Calculates the along-track convolution from the input suborbital data
        using the along-track weighting function of the spaceborne radar.
        Further, the function calculates the error due to satellite velocity.
        """

        ds = self.ds[["ze", "vm"]].copy()
        ds = ds.fillna(0)

        # create dask array by splitting height to reduce memory when expanding
        # window dimension
        ds = ds.chunk(chunks={"height": 50})

        # create new dataset with window dimension stacked as third dimension
        weight = xr.DataArray(self.beam.along_track_weights, dims=["window"])
        ds = ds.rolling(along_track=len(weight), center=True).construct(
            "window"
        )

        # add error due to satellite motion to each doppler velocity window
        da_vel_error = xr.DataArray(self.beam.velocity_error, dims=["window"])
        ds["vm_err"] = ds["vm"] + da_vel_error
        ds["vm_err"] = ds["vm_err"].where(ds["vm"] != 0, ds["vm"])

        # calculate along-track convolution and convert dask to xarray
        self.ds["ze_acon"] = ds["ze"].dot(weight).compute()
        self.ds["vm_acon"] = ds["vm"].dot(weight).compute()
        self.ds["vm_acon_err"] = ds["vm_err"].dot(weight).compute()

    def _integrate_along_track(self) -> None:
        """
        Integrates the along-track convoluted data to profiles, which represent
        the satellite's footprint. The along-track integration is given by the
        along track resolution satellite variable.

        Along track bins of satellite refer to center of field of view.
        """

        # create bin edges for along-track integration
        # the last radar bin is created only if it is included in the input
        # grid. the same convention is applied to the height grid
        along_track_sat_edges: np.ndarray = np.arange(
            self.ds["along_track"][0],
            self.ds["along_track"][-1],
            self.beam.spec.along_track_resolution,
        )

        # create bin centers for along-track integration
        along_track_bin_center = (
            along_track_sat_edges[:-1] + along_track_sat_edges[1:]
        ) / 2

        # along-track integration onto satellite along-track grid
        kwds = {
            "group": "along_track",
            "bins": along_track_sat_edges,
            "labels": along_track_bin_center,
        }

        self.ds["ze_aconint"] = self.ds["ze_acon"].groupby_bins(**kwds).mean()
        self.ds["vm_aconint"] = self.ds["vm_acon"].groupby_bins(**kwds).mean()
        self.ds["vm_aconint_err"] = (
            self.ds["vm_acon_err"].groupby_bins(**kwds).mean()
        )

        # rename along-track dimension
        self.ds = self.ds.rename({"along_track_bins": "along_track_sat"})

    def _convolve_height(self) -> None:
        """
        Convolution of the along-track integrated data with the range
        weighting function of the spaceborne radar.
        """

        # this defines the weights for the range gates that will be averaged
        da_range_weights = xr.DataArray(
            data=self.beam.range_weights,
            coords={"pulse_center_distance": self.beam.range_bins},
            dims=["pulse_center_distance"],
            name="range_weights",
        )

        # this defines the rolling window interval, i.e., the factor by which
        # the along-track resolution is reduced
        stride = int(
            self.beam.spec.range_resolution
            / self.ds["height"].diff("height")[0]
        )

        # create new dimension with all range gates that contribute to the
        # along-height convolution at each range gate
        ds = (
            self.ds[["ze_aconint", "vm_aconint", "vm_aconint_err"]]
            .rolling(height=len(da_range_weights), center=True)
            .construct("pulse_center_distance", stride=stride)
        )
        ds = ds.rename({"height": "height_sat"})

        # calculate along-range convolution for reflectivity
        self.ds["ze_sat"] = ds["ze_aconint"].dot(da_range_weights)

        # calculate reflectivity-weighted along-range
        # convolution for Doppler velocity
        # reflectivity threshold set to -20 dBZ (0.01 mm^6/m^3)
        reflectivity_threshold = 0.01
        ze_masked = ds["ze_aconint"].where(
            ds["ze_aconint"] >= reflectivity_threshold, 0
        )

        numerator_vm_sat = (
            ds["vm_aconint"] * ze_masked * da_range_weights
        ).sum("pulse_center_distance")
        denominator_vm_sat = (ze_masked * da_range_weights).sum(
            "pulse_center_distance"
        )
        self.ds["vm_sat"] = numerator_vm_sat / denominator_vm_sat
        self.ds["vm_sat"] = self.ds["vm_sat"].where(denominator_vm_sat != 0, 0)

        numerator_vm_sat_vel = (
            ds["vm_aconint_err"] * ze_masked * da_range_weights
        ).sum("pulse_center_distance")
        denominator_vm_sat_vel = (ze_masked * da_range_weights).sum(
            "pulse_center_distance"
        )
        self.ds["vm_sat_vel"] = numerator_vm_sat_vel / denominator_vm_sat_vel
        self.ds["vm_sat_vel"] = self.ds["vm_sat_vel"].where(
            denominator_vm_sat_vel != 0, 0
        )

    def _calculate_nubf(self) -> None:
        r"""
        Calculates the non-uniform beam filling from the standard
        deviation of Ze within the radar volume.

        Currently, the flag is expressed as standard deviation only and no
        threshold to indicate high standard deviation is applied. This may
        be added in the future to reduce the output file size.
        """

        # create labels for each satellite pixel (height_sat x along_track_sat)
        labels = np.arange(
            self.ds["height_sat"].size * self.ds["along_track_sat"].size
        ).reshape(self.ds["ze_sat"].shape)

        # calculate bin edges of satellite grid
        along_track_sat_bin_edges = self.calculate_along_track_sat_bin_edges()
        height_sat_bin_edges = self.calculate_height_sat_bin_edges()

        # assign satellite pixel label to each input pixel of suborbital radar
        ix_along_track = np.searchsorted(
            along_track_sat_bin_edges[:-1],
            self.ds["along_track"].values,
            side="left",
        )
        ix_height = np.searchsorted(
            height_sat_bin_edges[:-1],
            self.ds["height"].values,
            side="left",
        )

        # adjust index at first position
        ix_height[ix_height == 0] = 1
        ix_along_track[ix_along_track == 0] = 1

        ix_height = ix_height - 1
        ix_along_track = ix_along_track - 1

        ix_height, ix_along_track = np.meshgrid(
            ix_along_track, ix_height, indexing="ij"
        )
        labels_input_grid = labels[ix_height, ix_along_track]

        # calculate standard deviation of ze on input grid in linear units
        # this is done with pandas for faster performance
        df_ze = (
            self.ds["ze"]
            .stack({"x": ("along_track", "height")})
            .to_dataframe()
        )
        df_ze["labels"] = labels_input_grid.flatten()
        df_nubf = li2db(df_ze["ze"]).groupby(df_ze["labels"]).std()

        # convert to xarray
        self.ds["nubf"] = xr.DataArray(
            df_nubf.values.reshape(labels.shape),
            dims=["along_track_sat", "height_sat"],
            coords={
                "along_track_sat": self.ds["along_track_sat"],
                "height_sat": self.ds["height_sat"],
            },
        )

    def _calculate_nubf_flag(self, threshold: float = 1.0) -> None:
        """
        Calculate non-uniform beam filling flag. The flag is 1 if the
        non-uniform beam filling is higher than a certain threshold, and 0
        otherwise.

        Parameters
        ----------
        threshold : float
            Threshold for non-uniform beam filling. The default is 1 dB.
        """

        self.ds["nubf_flag"] = (self.ds["nubf"] > threshold).astype("int")

    def _calculate_vm_bias(self) -> None:
        """
        Calculate the satellite Doppler velocity bias between the estimate
        with and without satellite motion error.
        """

        self.ds["vm_bias"] = self.ds["vm_sat"] - self.ds["vm_sat_vel"]

    def _calculate_vm_bias_flag(self, threshold: float = 0.5) -> None:
        """
        Calculate the satellite Doppler velocity bias flag. The flag is 1 if
        the absolute satellite Doppler velocity bias is higher than 0.5 m s-1,
        and 0 otherwise.

        Parameters
        ----------
        threshold : float
            Threshold for satellite Doppler velocity bias. The default is 0.5
            m s-1.
        """

        self.ds["vm_bias_flag"] = (
            np.abs(self.ds["vm_bias"]) > threshold
        ).astype("int")

    def _calculate_signal_fraction(self) -> None:
        """
        Calculates the fraction of bins that contain a ze signal above the
        detection limit of the spaceborne radar. The fraction is 1 if all
        bins contain signal, and 0 if no bins contain signal.
        """

        # calculate bin edges of satellite grid
        along_track_sat_bin_edges = self.calculate_along_track_sat_bin_edges()
        height_sat_bin_edges = self.calculate_height_sat_bin_edges()

        # calculate fraction of bins that contain signal
        self.ds["signal_fraction"] = self.ds["ze"] > 0

        self.ds["signal_fraction"] = (
            self.ds["signal_fraction"]
            .groupby_bins(
                "along_track",
                bins=along_track_sat_bin_edges,
                labels=self.ds["along_track_sat"].values,
            )
            .mean()
        ).rename({"along_track_bins": "along_track_sat"})

        self.ds["signal_fraction"] = (
            self.ds["signal_fraction"]
            .groupby_bins(
                "height",
                bins=height_sat_bin_edges,
                labels=self.ds["height_sat"].values,
            )
            .mean()
        ).rename({"height_bins": "height_sat"})

    def _calculate_ms_flag(self) -> None:
        """
        Calculates the multiple scattering flag. The flag is 1 if multiple
        scattering occurs, and 0 if no multiple scattering occurs.

        The flag is calculated from the radar reflectivity of the spaceborne
        radar from these steps:
        - Calculate integral of radar reflectivity above a certain threshold
        from the top of the atmosphere (TOA) down to the surface.
        - Multiple scattering occurs if the integral reaches a critical value
        at a certain height.
        """

        ms_threshold = 12.0
        ms_threshold_integral = 41.0

        # get ze above multiple scattering threshold
        da_ze_above_threshold = self.ds["ze_sat"] > db2li(ms_threshold)

        # integrate from top to bottom (this requires sel)
        self.ds["ms_flag"] = (
            self.ds["ze_sat"]
            .where(da_ze_above_threshold)
            .sel(height_sat=self.ds["height_sat"][::-1])
            .cumsum("height_sat")
            .sel(height_sat=self.ds["height_sat"])
        ) * self.beam.spec.range_resolution

        # convert to dBZ and calculate flag
        self.ds["ms_flag"] = (
            li2db(self.ds["ms_flag"]) > ms_threshold_integral
        ).astype("int")

        # set flag to 0 below the surface
        subsurface = self.ds["height_sat"].where(
            self.ds["height_sat"] < 0, drop=True
        )
        self.ds["ms_flag"].loc[{"height_sat": subsurface}] = 0

    def _apply_detection_limit(self, var_ze: str, var_other: list) -> None:
        """
        Applies the detection limit of the spaceborne radar to the along-height
        convoluted data.

        Parameters
        ----------
        var_ze : xr.DataArray
            Radar reflectivity reflectivity variable name
        var_other : list
            List with other variables that should be masked with the radar
            reflectivity detection limit.
        """

        # apply radar reflectivity detection limit
        ix = self.ds[var_ze] > db2li(self.beam.spec.detection_limit)

        for var in var_other:
            self.ds[var] = self.ds[var].where(ix)

    @staticmethod
    def add_noise(
        x: xr.DataArray, x_std: np.ndarray, noise: np.ndarray
    ) -> xr.DataArray:
        """
        Equation to calculate the noise from values without noise, the
        uncertainty of the values, and random noise.

        Parameters
        ----------
        x :
            Radar reflectivity [dB] or doppler velocity [m s-1]
        x_std :
            Radar reflectivity uncertainty [dB] or doppler velocity uncertainty
            [m s-1]
        noise :
            Random noise with shape equal to x.

        Returns
        -------
        x_noise :
            Radar reflectivity with added noise [dB]
        """

        return x + x_std * noise

    def calculate_vm_std_nubf(self) -> np.ndarray:
        """
        Calculate outstanding error in correcting Mean Doppler Velocity biases
        caused by non-uniform beam filling

        The calculation is based on the horizontal radar reflectivity gradient
        at the input resolution. The gradient is calculated along the along-
        track direction. The gradient is then averaged onto the satellite grid
        and the absolute value is taken. The error is then calculated as 0.15
        times the gradient divided by 3 dBZ/km. Bins without reflectivity are
        set to 0 before averaging onto satellite resolution.
        """

        # calculate bin edges of satellite grid
        along_track_sat_bin_edges = self.calculate_along_track_sat_bin_edges()
        height_sat_bin_edges = self.calculate_height_sat_bin_edges()

        # calculate horizontal ze gradient on input grid in dBZ/km
        ze_gradient = li2db(self.ds["ze"]).diff("along_track") / (
            self.ds["along_track"].diff("along_track").mean() / 1000
        )

        # fill nan values with zero
        ze_gradient = ze_gradient.fillna(0)
        ze_gradient = (
            ze_gradient.groupby_bins(
                "along_track",
                bins=along_track_sat_bin_edges,
                labels=self.ds["along_track_sat"].values,
            )
            .mean()
            .groupby_bins(
                "height",
                bins=height_sat_bin_edges,
                labels=self.ds["height_sat"].values,
            )
            .mean()
        )
        ze_gradient = ze_gradient.rename(
            {
                "along_track_bins": "along_track_sat",
                "height_bins": "height_sat",
            }
        )

        return 0.15 * np.abs(ze_gradient) / 3

    def vm_uncertainty_equation(
        self,
        vm_std_broad: np.ndarray,
        vm_std_nubf: np.ndarray,
    ) -> np.ndarray:
        """
        Calculates the total Doppler velocity uncertainty based on the
        broadening Doppler velocity uncertainty and the non-uniform beam
        filling Doppler velocity uncertainty.

        Based on Equation (4) in

        Parameters
        ----------
        vm_std_broad : float, np.array
            Doppler velocity uncertainty due to broadening [m s-1]
        vm_std_nubf : float, np.array
            Doppler velocity uncertainty due to non-uniform beam filling
            [m s-1]
        """

        return np.sqrt(vm_std_broad**2 + vm_std_nubf**2)

    def _calculate_ze_noise(self) -> None:
        """
        Adds noise to satellite radar reflectivity based on the pre-defined
        lookup table with noise values for different radar reflectivity bins.
        Empty bins are filled with noise according to the noise level.
        """

        # generate noise
        lower = -4.5
        upper = 4.5
        mu = 0
        sigma = 2
        n = np.prod(self.ds["ze_sat"].shape)
        noise = np.array(
            stats.truncnorm.rvs(
                a=(lower - mu) / sigma,
                b=(upper - mu) / sigma,
                loc=mu,
                scale=sigma,
                size=n,
            )
        ).reshape(self.ds["ze_sat"].shape)

        # interpolates discrete standard deviations
        f = interp1d(
            self.beam.spec.ze_bins,
            self.beam.spec.ze_std,
            kind="linear",
            fill_value="extrapolate",
        )

        # apply noise
        self.ds["ze_sat_noise"] = db2li(
            self.add_noise(
                x=li2db(self.ds["ze_sat"]),
                x_std=f(li2db(self.ds["ze_sat"])),
                noise=noise,
            )
        )

    def _calculate_vm_noise(self) -> None:
        """
        Adds noise to satellite Doppler velocity based on the pre-defined
        lookup table with noise values for different radar reflectivity bins.

        Note:
        The noise is added to the satellite Doppler velocity with the satellite
        motion error.
        """

        lower = -self.beam.spec.nyquist_velocity
        upper = self.beam.spec.nyquist_velocity
        mu = 0
        sigma = 1
        n = np.prod(self.ds["vm_sat_vel"].shape)
        noise = np.array(
            stats.truncnorm.rvs(
                a=(lower - mu) / sigma,
                b=(upper - mu) / sigma,
                loc=mu,
                scale=sigma,
                size=n,
            )
        ).reshape(self.ds["vm_sat_vel"].shape)

        # interpolates discrete standard deviations
        f = interp1d(
            self.beam.spec.vm_bins_broad,
            self.beam.spec.vm_std_broad,
            kind="linear",
            fill_value="extrapolate",
        )

        # calculate uncertainty due to broadening
        vm_std_broad = f(li2db(self.ds["ze_sat"]))

        # calculate uncertainty due to non-uniform beam filling
        vm_std_nubf = self.calculate_vm_std_nubf()

        # calculate total Doppler velocity uncertainty
        vm_std = self.vm_uncertainty_equation(
            vm_std_broad=vm_std_broad,
            vm_std_nubf=vm_std_nubf,
        )

        # add Doppler velocity error
        self.ds["vm_sat_noise"] = self.add_noise(
            x=self.ds["vm_sat_vel"], x_std=vm_std, noise=noise
        )

    def _fold_vm(self) -> None:
        """
        Doppler velocity folding correction.
        """

        # keys: nyquist velocity offset added for folding
        # values: velocity bin edges as multiple of the nyquist velocity
        folding_dct = {
            -2: [1.0, 3.0],
            -4: [3.0, 5.0],
            -6: [5.0, 7.0],
            -8: [7.0, 9.0],
            2: [-3.0, -1.0],
            4: [-5.0, -3.0],
            6: [-7.0, -5.0],
            8: [-9.0, -7.0],
        }

        # data array with folded velocity
        self.ds["vm_sat_folded"] = self.ds["vm_sat_noise"].copy()

        # folding flag
        self.ds["folding_flag"] = xr.zeros_like(self.ds["vm_sat_noise"])

        for offset, (v0, v1) in folding_dct.items():
            # convert factors to doppler velocity
            v0 = v0 * self.beam.spec.nyquist_velocity
            v1 = v1 * self.beam.spec.nyquist_velocity
            vm_offset = offset * self.beam.spec.nyquist_velocity

            # this is true if folding is applied
            in_interval = (self.ds["vm_sat_folded"] >= v0) & (
                self.ds["vm_sat_folded"] < v1
            )

            # assign folding factor to flag
            self.ds["folding_flag"] = xr.where(
                in_interval,
                1,
                self.ds["folding_flag"],
            )

            # fold velocity within the given interval
            self.ds["vm_sat_folded"] = xr.where(
                in_interval,
                self.ds["vm_sat_folded"] + vm_offset,
                self.ds["vm_sat_folded"],
            )

        # ensure that doppler velocity is within the nyquist velocity
        assert (
            self.ds["vm_sat_folded"].min() >= -self.beam.spec.nyquist_velocity
        ), (
            f"Velocity values below the nyquist velocity: "
            f"{self.ds['vm_sat_folded'].min()}"
        )

        assert (
            self.ds["vm_sat_folded"].max() <= self.beam.spec.nyquist_velocity
        ), (
            f"Velocity values above the nyquist velocity: "
            f"{self.ds['vm_sat_folded'].max()}"
        )

    def _add_attributes(self) -> None:
        """
        Adds attributes to the variables of the dataset
        """

        # overwrite attributes of ze and vm inputs
        self.ds["ze"].attrs = {
            "long_name": "Radar reflectivity factor of input",
            "units": "mm6 m-3",
        }

        self.ds["vm"].attrs = {
            "long_name": "Mean Doppler velocity of input",
            "units": "m s-1",
        }

        # add attributes to dimensions
        self.ds["along_track"].attrs = {
            "long_name": "Along-track distance",
            "units": "m",
        }

        self.ds["height"].attrs = {
            "standard_name": "height_above_mean_sea_level",
            "long_name": "Height above mean sea level",
            "units": "m",
        }

        self.ds["along_track_sat"].attrs = {
            "long_name": "Along-track distance at satellite resolution",
            "units": "m",
        }

        self.ds["height_sat"].attrs = {
            "long_name": "Height above mean sea level at satellite resolution",
            "units": "m",
        }

        # add attributes to variables
        self.ds["nubf"].attrs = {
            "long_name": "Non-uniform beam filling",
            "units": "dBZ",
            "description": "Non-uniform beam filling calculated as the "
            "standard deviation of radar reflectivity in linear units "
            "of the input data.",
        }

        self.ds["nubf_flag"].attrs = {
            "long_name": "Non-uniform beam filling flag",
            "units": "1",
            "description": "1 means non-uniform beam filling is higher "
            "than 1 dB, 0 means non-uniform beam filling is lower than 1 dB.",
        }

        self.ds["signal_fraction"].attrs = {
            "long_name": "Fraction of bins that contain signal",
            "description": "1 means all bins contain signal, "
            "0 means no bins contain signal.",
        }

        self.ds["ms_flag"].attrs = {
            "long_name": "Multiple scattering flag",
            "units": "1",
            "description": "1 means multiple "
            "scattering occurs, 0 means no multiple scattering occurs. "
            "This flag only makes sense for airborne observations. "
            "Groundbased observations likely underestimate the occurrence of "
            "multiple scattering due to rain attenuation.",
        }

        self.ds["folding_flag"].attrs = {
            "long_name": "Folding flag",
            "units": "1",
            "description": "1 means velocity is folded, 0 means "
            "velocity is not folded.",
        }

        self.ds["ze_acon"].attrs = {
            "long_name": "Convolved radar reflectivity factor",
            "units": "mm6 m-3",
        }

        self.ds["vm_acon"].attrs = {
            "long_name": "Convolved mean Doppler velocity",
            "units": "m s-1",
        }

        self.ds["vm_acon_err"].attrs = {
            "long_name": "Convolved mean Doppler velocity with satellite "
            "motion error",
            "units": "m s-1",
        }

        self.ds["ze_aconint"].attrs = {
            "long_name": "Convolved and integrated radar reflectivity factor",
            "units": "mm6 m-3",
        }

        self.ds["vm_aconint"].attrs = {
            "long_name": "Convolved and integrated mean Doppler velocity",
            "units": "m s-1",
        }

        self.ds["vm_aconint_err"].attrs = {
            "long_name": "Convolved and integrated mean Doppler velocity with "
            "satellite motion error",
            "units": "m s-1",
        }

        self.ds["ze_sat"].attrs = {
            "long_name": "Convolved and integrated radar reflectivity factor",
            "units": "mm6 m-3",
            "description": "Convolved and integrated radar reflectivity factor "
            "along height and track",
        }

        self.ds["vm_sat"].attrs = {
            "long_name": "Convolved and integrated mean Doppler velocity",
            "units": "m s-1",
            "description": "Convolved and integrated mean Doppler velocity "
            "along height and track",
        }

        self.ds["vm_sat_vel"].attrs = {
            "long_name": "Convolved and integrated mean Doppler velocity with "
            "satellite motion error",
            "units": "m s-1",
            "description": "Convolved and integrated mean Doppler velocity "
            "with satellite motion error along height and track",
        }

        self.ds["vm_bias"].attrs = {
            "long_name": "Doppler velocity bias",
            "units": "m s-1",
            "description": "Doppler velocity bias between the estimate "
            "with and without satellite motion error. Higher biases "
            "occur under higher non-uniform beam filling.",
        }

        self.ds["ze_sat_noise"].attrs = {
            "long_name": "Convolved and integrated radar reflectivity factor "
            "with noise",
            "units": "mm6 m-3",
        }

        self.ds["vm_sat_noise"].attrs = {
            "long_name": "Convolved and integrated mean Doppler velocity "
            "with noise and satellite motion error",
            "units": "m s-1",
        }

        self.ds["vm_sat_folded"].attrs = {
            "long_name": "Doppler velocity with noise, satellite motion "
            "error, and folding",
            "units": "m s-1",
        }

        # time encoding
        self.ds["time"].encoding = {
            "units": "seconds since 1970-01-01 00:00:00",
            "calendar": "gregorian",
        }
        self.ds["time"].attrs = {
            "standard_name": "time",
            "long_name": "Time UTC",
        }

        # add variables about satellite
        self.ds["sat_ifov"] = xr.DataArray(
            self.beam.ifov,
            attrs={
                "long_name": "Satellite instantaneous field of view",
                "units": "m",
            },
        )

        self.ds["sat_range_resolution"] = xr.DataArray(
            self.beam.spec.range_resolution,
            attrs={
                "long_name": "Satellite range resolution",
                "units": "m",
            },
        )

        self.ds["sat_along_track_resolution"] = xr.DataArray(
            self.beam.spec.along_track_resolution,
            attrs={
                "long_name": "Satellite along-track resolution",
                "units": "m",
            },
        )
