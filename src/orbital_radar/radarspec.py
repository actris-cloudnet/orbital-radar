import logging

import xarray as xr

"""
This module contains the satellite class and functions to calculate the
along-track and along-range averaging parameters. Main method and
definitions based on Lamer et al (2020) and Schirmacher et al. (2023).

Two pre-defined satellites are available: EarthCARE and CloudSat.

**EarthCARE**

- frequency:                        Kollias et al. (2014), Table 1
- velocity:                         Kollias et al. (2014), Eq 4
- antenna diameter:                 Kollias et al. (2014), Table 1
- altitude:                         Kollias et al. (2014), Table 1
- pulse length:                     Kollias et al. (2014), Table 1
- along track resolution:           Kollias et al. (2014), Table 1
- range resolution:                 Kollias et al. (2014), Table 1
- ifov_factor:                      Kollias et al. (2022), Table 1
- ifov_scale:                       based on Tanelli et al. (2008) and as long nothing is reported it is 1 s
- detection limit:                  Kollias et al. (2014), Table 1
- pules repetition frequency (PRF): Kollias et al. (2022), Table 1
- noise_ze:                         Kollias et al. (2014), Table 1
- ze_bins:                          Hogan et al. (2005),
- ze_std:                           Hogan et al. (2005),
- ze_std_background:
- vm_bins_broad:                    Kollias et al. (2022), Figure 7
- vm_std_broad:                     Kollias et al. (2022), Figure 7
- vm_std_broad_background:          Kollias et al. (2022)

**CloudSat**

- frequency:                        Kollias et al. (2014), Table 1
- velocity:                         Kollias et al. (2014), Eq 4
- antenna diameter:                 Kollias et al. (2014), Table 1
- altitude:                         Kollias et al. (2014), Table 1
- pulse length:                     Kollias et al. (2014), Table 1
- along track resolution:           Kollias et al. (2014), Table 1
- range resolution:                 Kollias et al. (2014), Table 1
- ifov_factor:                      Kollias et al. (2022), Table 1 for the Arctic Schirmacher et al. (2023)
- ifov_scale:                       Tanelli et al. (2008), "integration accurecy of the pulse is 0.968 s"
- detection limit:                  Kollias et al. (2014), Table 1
- pules repetition frequency (PRF): Kollias et al. (2014, 2022), Table 1
- noise_ze:
- ze_bins:                          Hogan et al. (2005),
- ze_std:                           Hogan et al. (2005),
- ze_std_background:
- vm_bins_broad:                    not used
- vm_std_broad:                     not used
- vm_std_broad_background:          not used

References
----------
Hogan et al. (2005)       : https://doi.org/10.1175/JTECH1768.1
Kollias et al. (2014)     : https://doi.org/10.1175/JTECH-D-11-00202.1
Kollias et al. (2022)     : https://doi.org/10.3389/frsen.2022.860284
Lamer et al. (2020)       : https://doi.org/10.5194/amt-13-2363-2020
Schirmacher et al. (2023) : https://doi.org/10.5194/egusphere-2023-636
Tanelli et al. (2008)     : https://doi.org/10.1109/TGRS.2008.2002030
"""

import pathlib
from dataclasses import dataclass
from typing import List, Union

import numpy as np
from scipy.constants import c as SPEED_OF_LIGHT

from orbital_radar.helpers import db2li, read_range_weighting_function

RADARS_PREDEFINED = {
    "earthcare": {
        "name": "EarthCARE",
        "frequency": 94.05e9,
        "velocity": 7200,
        "antenna_diameter": 2.5,
        "altitude": 400000,
        "pulse_length": 500,
        "along_track_resolution": 500,
        "range_resolution": 100,
        "ifov_factor": 74.5,
        "ifov_scale": 1,
        "detection_limit": -35,
        "nyquist_velocity": 5.7,
        "pulse_repetition_frequency": 6000,
        "noise_ze": -21.5,
        "ze_bins": [-37, -25, -13],
        "ze_std": [0.5, 0.3, 0.2],
        "ze_std_background": 0.4252,  # 0.2176,
        "vm_bins_broad": [
            -37,
            -34,
            -31,
            -28,
            -25,
            -22,
            -19,
            -16,
            -13,
            -10,
            -7,
            -4,
        ],
        "vm_std_broad": [
            3.27,
            3.12,
            2.83,
            2.35,
            1.63,
            1.09,
            0.76,
            0.59,
            0.52,
            0.49,
            0.48,
            0.47,
        ],
        "vm_std_broad_background": 1.09,
    },
    "cloudsat": {
        "name": "CloudSat",
        "frequency": 94.05e9,
        "velocity": 7000,
        "antenna_diameter": 1.85,
        "altitude": 720000,
        "pulse_length": 480,
        "along_track_resolution": 1100,
        "range_resolution": 240,
        "ifov_factor": 67,
        "ifov_scale": 1,
        "detection_limit": -30,
        "nyquist_velocity": 5.7,
        "pulse_repetition_frequency": 4000,
        "noise_ze": -15.0,
        "ze_bins": [
            -37,
            -34,
            -31,
            -28,
            -25,
            -22,
            -19,
            -16,
            -13,
            -10,
            -7,
            -4,
        ],
        "ze_std": [
            9.24,
            4.77,
            2.54,
            1.41,
            0.85,
            0.56,
            0.42,
            0.35,
            0.32,
            0.3,
            0.29,
            0.28,
        ],
        "ze_std_background": 0.2176,
        "vm_bins_broad": [],
        "vm_std_broad": [],
        "vm_std_broad_background": np.nan,
    },
    "cloudsat_schirmacher_et_al_2023": {
        "name": "CloudSat",
        "frequency": 94.05e9,
        "velocity": 6800,
        "antenna_diameter": 1.85,
        "altitude": 705000,
        "pulse_length": 480,
        "along_track_resolution": 1093,
        "range_resolution": 240,
        "ifov_factor": 67,
        "ifov_scale": 0.968,
        "detection_limit": -27,
        "nyquist_velocity": 5.7,
        "pulse_repetition_frequency": 7150,
        "noise_ze": -27.0,
        "ze_bins": [
            -37,
            -34,
            -31,
            -28,
            -25,
            -22,
            -19,
            -16,
            -13,
            -10,
            -7,
            -4,
        ],
        "ze_std": [
            9.24,
            4.77,
            2.54,
            1.41,
            0.85,
            0.56,
            0.42,
            0.35,
            0.32,
            0.3,
            0.29,
            0.28,
        ],
        "ze_std_background": 0.2176,
        "vm_bins_broad": [],
        "vm_std_broad": [],
        "vm_std_broad_background": np.nan,
    },
}


@dataclass
class RadarSpec:
    """
    This class contains the satellite parameters.

    Units of radar specification
    ----------------------------
    - frequency: radar frequency [Hz]
    - velocity: satellite velocity [m s-1]
    - antenna diameter: radar antenna diameter [m]
    - altitude: satellite altitude [m]
    - pulse length: radar pulse length [m]
    - along track resolution: radar along track resolution [m]
    - range resolution: radar range resolution [m]
    - detection limit: radar detection limit [dBZ]
    - noise_ze: radar noise floor [dBZ]
    - ze_bins: radar Ze lookup table [dBZ]
    - ze_std: radar standard deviation lookup table [dBZ]
    - ze_std_background: radar standard deviation background [dBZ]
    - vm_bins_broad: radar reflectivity bin of vm_std_broad [dBZ]
    - vm_std_broad: Doppler velocity broadening due to platform motion [m s-1]
    - vm_std_broad_background: radar standard deviation background [m s-1]
    - nyquist velocity: radar nyquist velocity [m s-1]
    - pulse repetition frequency: radar pulse repetition frequency [Hz]
    """

    name: str
    frequency: float
    velocity: int
    antenna_diameter: float
    altitude: int
    pulse_length: int
    along_track_resolution: int
    range_resolution: int
    ifov_factor: float
    ifov_scale: float
    detection_limit: float
    noise_ze: float
    ze_bins: Union[List[float], np.ndarray]
    ze_std: Union[List[float], np.ndarray]
    ze_std_background: float
    vm_bins_broad: Union[List[float], np.ndarray]
    vm_std_broad: Union[List[float], np.ndarray]
    vm_std_broad_background: float
    nyquist_velocity: float = np.nan
    pulse_repetition_frequency: float = np.nan


class RadarBeam:
    """
    This class manages the satellite specifications from pre-defined or user-
    specified space-borne radars. It also contains transformation functions
    for along-track and along-range averaging.
    """

    def __init__(  # type: ignore
        self,
        sat_name: str | None = None,
        nyquist_from_prf: bool = False,
        **sat_params,
    ):
        """
        Initializes the satellite parameters and calculates along-track and
        along-range weighting functions, and the velocity error due to
        satellite velocity.

        The function requires along-track and along-range bins.

        The following parameters will be derived for later use in the simulator

        - instantaneous field of view
        - normalized along-track weighting function
        - along track resolution
        - normalized along-range weighting function
        - range resolution
        - satellite velocity error

        Parameters
        ----------
        file_earthcare : str
            path to file containing EarthCARE CPR weighting function. This
            file is used if the satellite name is 'earthcare'.
        satellite_name : str
            name of the satellite, e.g. 'earthcare' or 'cloudsat'
        nqv_from_prf : bool
            if True, the Nyquist velocity is calculated from the pulse
            repetition frequency. If False, the Nyquist velocity must be given
            as a parameter. Default is False.
        **sat_params: keyword arguments to overwrite the predefined satellite
        """

        # set satellite parameters from pre-defined satellites
        if sat_name is not None:
            # update pre-defined satellite parameters with sat_params
            radar_predefined = RADARS_PREDEFINED[sat_name].copy()
            radar_predefined.update(sat_params)
            self.spec = RadarSpec(**radar_predefined)  # type: ignore

        # set satellite parameters from user-specified satellite
        else:
            self.spec = RadarSpec(**sat_params)

        # convert lookup tables to numpy arrays
        self.spec.ze_bins = np.array(self.spec.ze_bins)
        self.spec.ze_std = np.array(self.spec.ze_std)
        self.spec.vm_bins_broad = np.array(self.spec.vm_bins_broad)
        self.spec.vm_std_broad = np.array(self.spec.vm_std_broad)

        self.sat_name = sat_name

        # initialize along-track and along-range averaging parameters
        self.atrack_bins = np.array([])
        self.atrack_weights = np.array([])
        self.range_weights = np.array([])
        self.range_bins = np.array([])

        # initialize derived parameters
        self.wavelength = np.nan
        self.ifov = np.nan
        self.theta_along = np.nan
        self.velocity_error = np.nan

        # calculate derived parameters
        self._calculate_wavelength()

        # calculate Nyquist velocity from pulse repetition frequency
        if nyquist_from_prf:
            logging.info(
                "Nyquist velocity is calculated from pulse repetition frequency."
            )
            self._calculate_nyquist_velocity()

        else:
            logging.info(
                "Nyquist velocity parameter is used instead of pulse "
                "repetition frequency."
            )

        # show summary of satellite parameters
        # self.params

    def calculate_weighting_functions(
        self, along_track_coords: xr.DataArray, range_coords: xr.DataArray
    ) -> None:
        """
        Calculates the along-track and along-range weighting functions.

        Parameters
        ----------
        along_track_coords : array
            along-track coordinates of the ground-based radar [m]
        range_coords : array
            range coordinates of the ground-based radar [m]
        """

        # calculate along-track averaging parameters
        self._calculate_along_track(along_track_coords=along_track_coords)

        # calculate velocity error due to satellite velocity
        self._calculate_velocity_error()

        # calculate along-range averaging parameters
        self._calculate_along_range(range_coords=range_coords)

    @property
    def params(self) -> None:
        """Prints a summary of the satellite parameters"""

        print(
            f"Satellite: {self.spec.name}\n"
            f"Frequency: {self.spec.frequency*1e-9} GHz\n"
            f"Velocity: {self.spec.velocity} m s-1\n"
            f"Antenna diameter: {self.spec.antenna_diameter} m\n"
            f"Altitude: {self.spec.altitude} m\n"
            f"Pulse length: {self.spec.pulse_length} m\n"
            f"Horizontal resolution: {self.spec.along_track_resolution} m\n"
            f"Vertical resolution: {self.spec.range_resolution} m\n"
            f"Nyquist velocity: {np.round(self.spec.nyquist_velocity, 2)} m s-1\n"
            f"Pulse repetition frequency: {np.round(self.spec.pulse_repetition_frequency, 0)} Hz\n"
        )

    def _calculate_wavelength(self) -> None:
        """
        Calculates the radar wavelength from the radar frequency.

        Units
        -----
        - frequency: radar frequency [Hz]
        - wavelength: radar wavelength [m]
        - speed of light: speed of light [m s-1]
        """

        self.wavelength = SPEED_OF_LIGHT / self.spec.frequency

    def _calculate_nyquist_velocity(self) -> None:
        """
        Calculates the Nyquist velocity from the pulse repetition frequency
        and the radar wavelength.
        """

        self.spec.nyquist_velocity = (
            self.wavelength * self.spec.pulse_repetition_frequency / 4
        )

    def _calculate_ifov(self) -> None:
        """
        Calculates the instantaneous field of view (IFOV) from the along-track
        averaging parameters.
        """

        # constant for ifov calculation
        self.theta_along = (
            self.spec.ifov_factor * self.wavelength
        ) / self.spec.antenna_diameter

        # instantaneous field of view
        self.ifov = (
            self.spec.altitude
            * np.tan(np.pi * self.theta_along / 180)
            * self.spec.ifov_scale
        )

    def _create_along_track_grid(
        self, along_track_coords: xr.DataArray
    ) -> None:
        """
        Creates the along-track grid.

        The along-track grid is defined from -ifov/2 to ifov/2. The spacing
        is defined by the along-track resolution. The outermost along-track
        bins relative to the line of size always lie within the IFOV.

        If the along-track grid is not equidistant, the along-track weighting
        function cannot be calculated.

        Parameters
        ----------
        along_track_coords : array
            along-track coordinates of the ground-based radar [m]
        """

        assert len(np.unique(np.diff(along_track_coords))) == 1, (
            "Along-track grid is not equidistant. "
            "Along-track weighting function cannot be calculated."
        )

        # grid with size of ifov centered around zero
        step = np.diff(along_track_coords)[0]
        self.atrack_bins = np.append(
            np.arange(-step, -self.ifov / 2, -step)[::-1],
            np.arange(0, self.ifov / 2, step),
        )

    def _create_along_range_grid(self, range_coords: xr.DataArray) -> None:
        """
        Creates range grid at which range weighting function is evaluated.

        The range grid is defined from -pulse_length to pulse_length. The
        spacing is defined by the range resolution of the ground-based radar.

        If the range grid is not equidistant, the range weighting function
        cannot be calculated.

        Parameters
        ----------
        range_coords : array
            range coordinates of the ground-based radar [m]
        """

        assert len(np.unique(np.diff(range_coords))) == 1, (
            "Range grid is not equidistant. "
            "Range weighting function cannot be calculated."
        )

        # grid with size of two pulse lengths centered around zero
        step = np.diff(range_coords)[0]
        self.range_bins = np.arange(
            -self.spec.pulse_length,
            self.spec.pulse_length + step,
            step,
        )

    def _calculate_along_track(self, along_track_coords: xr.DataArray) -> None:
        """
        Calculates along-track averaging parameters.

        Parameters
        ----------
        along_track_coords : array
            along-track coordinates of the ground-based radar [m]
        """

        # instantaneous field of view
        self._calculate_ifov()

        # calculate along-track grid
        self._create_along_track_grid(along_track_coords=along_track_coords)

        # along-track weighting function
        w_at = np.exp(
            -2 * np.log(2) * (self.atrack_bins / (self.ifov / 2)) ** 2
        )
        self.atrack_weights = w_at / np.sum(w_at)  # normalization

        assert (
            np.sum(self.atrack_weights) - 1 < 1e-10
        ), "Along-track weighting function is not normalized"

    def _calculate_velocity_error(self) -> None:
        """
        Calculates the velocity error due to satellite velocity.
        """

        # velocity error due to satellite velocity
        self.velocity_error = (
            self.spec.velocity / self.spec.altitude
        ) * self.atrack_bins  # type: ignore

    def _calculate_along_range(self, range_coords: xr.DataArray) -> None:
        """
        Calculates along-range averaging parameters.

        Parameters
        ----------
        range_coords : array
            range coordinates of the ground-based radar [m]
        """

        self._create_along_range_grid(range_coords=range_coords)

        # range weighting function
        if self.sat_name == "earthcare":
            self.range_weights = (
                self._normalized_range_weighting_function_earthcare()
            )

        else:
            self.range_weights = (
                self._normalized_range_weighting_function_default(
                    pulse_length=self.spec.pulse_length,
                    range_bins=self.range_bins,
                )
            )

    def _normalized_range_weighting_function_earthcare(self) -> np.ndarray:
        """
        Prepares EarthCARE range weighting function for along-range averaging.

        The high-resolution weighting function is interpolated to the
        range resolution of the ground-based radar.

        Returns
        -------
        range_weights : array
            normalized range weighting function
        """

        file_path = pathlib.Path(__file__).parent.absolute()
        cpr_file = file_path / "data/CPR_PointTargetResponse.txt"

        ds_wf = read_range_weighting_function(cpr_file)

        # linearize the weighting function
        da_wf = db2li(ds_wf["response"])

        # convert from tau factor to range and set height as dimension
        da_wf["height"] = da_wf["tau_factor"] * self.spec.pulse_length
        da_wf = da_wf.swap_dims({"tau_factor": "height"})

        # interpolate to range grid of ground-based radar
        da_wf = da_wf.interp(height=self.range_bins, method="linear")
        da_wf = da_wf.fillna(0)

        # normalize the linear weighting function
        da_wf /= da_wf.sum()

        range_weights = da_wf.values

        return range_weights

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
        range_weights = range_weights / np.sum(range_weights)  # normalization

        return range_weights
