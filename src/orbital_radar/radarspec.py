"""
This module contains the satellite class and functions to calculate the
along-track and along-range averaging parameters. Main method and
definitions based on Lamer et al (2020) and Schirmacher et al. (2023).

**EarthCARE**

- frequency:                        Kollias et al. (2014), Table 1
- velocity:                         Kollias et al. (2014), Eq 4
- antenna diameter:                 Kollias et al. (2014), Table 1
- altitude:                         Kollias et al. (2014), Table 1
- pulse length:                     Kollias et al. (2014), Table 1
- along track resolution:           Kollias et al. (2014), Table 1
- range resolution:                 Kollias et al. (2014), Table 1
- ifov_factor:                      Kollias et al. (2022), Table 1
- ifov_scale:                       based on Tanelli et al. (2008)
                                    and as long nothing is reported it is 1 s
- detection limit:                  Kollias et al. (2014), Table 1
- pules repetition frequency (PRF): Kollias et al. (2022), Table 1
- noise_ze:                         Kollias et al. (2014), Table 1
- ze_bins:                          Hogan et al. (2005),
- ze_std:                           Hogan et al. (2005),
- ze_std_background:
- vm_bins_broad:                    Kollias et al. (2022), Figure 7
- vm_std_broad:                     Kollias et al. (2022), Figure 7
- vm_std_broad_background:          Kollias et al. (2022)


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
from dataclasses import dataclass, field

import numpy as np
import xarray as xr
from scipy.constants import c as SPEED_OF_LIGHT

from orbital_radar import helpers


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

    name: str = "EarthCARE"
    frequency: float = 94.05e9
    velocity: int = 7200
    antenna_diameter: float = 2.5
    altitude: int = 400000
    pulse_length: int = 500
    along_track_resolution: int = 500
    range_resolution: int = 100
    ifov_factor: float = 74.5
    ifov_scale: float = 1.0
    detection_limit: float = -35.0
    nyquist_velocity: float = 5.7
    pulse_repetition_frequency: int = 6000
    noise_ze: float = -21.5
    ze_std_background: float = 0.4252
    ze_bins: np.ndarray = field(
        default_factory=lambda: np.array([-37, -25, -13])
    )
    ze_std: np.ndarray = field(
        default_factory=lambda: np.array([0.5, 0.3, 0.2])
    )
    vm_bins_broad: np.ndarray = field(
        default_factory=lambda: np.array(
            [
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
            ]
        )
    )
    vm_std_broad: np.ndarray = field(
        default_factory=lambda: np.array(
            [
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
            ]
        )
    )
    vm_std_broad_background: float = 1.09


class RadarBeam:
    """
    This class manages the satellite specifications from pre-defined or user-
    specified space-borne radars. It also contains transformation functions
    for along-track and along-range averaging.
    """

    def __init__(self) -> None:
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

        """

        self.spec = RadarSpec()

        self.along_track_bins: np.ndarray
        self.along_track_weights: np.ndarray
        self.range_weights: np.ndarray
        self.range_bins: np.ndarray

        # initialize derived parameters
        self.wavelength = SPEED_OF_LIGHT / self.spec.frequency
        self.ifov: float
        self.velocity_error: np.ndarray

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
        self._calculate_along_track(along_track_coords)

        # calculate velocity error due to satellite velocity
        self._calculate_velocity_error()

        # calculate along-range averaging parameters
        self._calculate_along_range(range_coords)

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
        self._create_along_track_grid(along_track_coords)

        # along-track weighting function
        w_at = np.exp(
            -2 * np.log(2) * (self.along_track_bins / (self.ifov / 2)) ** 2
        )
        self.along_track_weights = w_at / np.sum(w_at)  # normalization

        assert np.sum(self.along_track_weights) - 1 < 1e-10, (
            "Along-track weighting function is not normalized"
        )

    def _calculate_ifov(self) -> None:
        """
        Calculates the instantaneous field of view (IFOV) from the along-track
        averaging parameters.
        """

        # constant for ifov calculation
        theta_along = (
            self.spec.ifov_factor * self.wavelength
        ) / self.spec.antenna_diameter

        # instantaneous field of view
        self.ifov = (
            self.spec.altitude
            * np.tan(np.pi * theta_along / 180)
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
        self.along_track_bins = np.append(
            np.arange(-step, -self.ifov / 2, -step)[::-1],
            np.arange(0, self.ifov / 2, step),
        )

    def _calculate_velocity_error(self) -> None:
        """
        Calculates the velocity error due to satellite velocity.
        """
        # velocity error due to satellite velocity
        self.velocity_error = (
            self.spec.velocity / self.spec.altitude
        ) * self.along_track_bins

    def _calculate_along_range(self, range_coords: xr.DataArray) -> None:
        """
        Calculates along-range averaging parameters.

        Parameters
        ----------
        range_coords : array
            range coordinates of the ground-based radar [m]
        """

        self._create_along_range_grid(range_coords)

        # range weighting function
        self.range_weights = (
            self._normalized_range_weighting_function_earthcare()
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

        ds_wf = helpers.read_range_weighting_function(cpr_file)

        # linearize the weighting function
        da_wf = helpers.db2li(ds_wf["response"])

        # convert from tau factor to range and set height as dimension
        da_wf["height"] = da_wf["tau_factor"] * self.spec.pulse_length
        da_wf = da_wf.swap_dims({"tau_factor": "height"})

        # interpolate to range grid of ground-based radar
        da_wf = da_wf.interp(height=self.range_bins, method="linear")
        da_wf = da_wf.fillna(0)

        # normalize the linear weighting function
        da_wf /= da_wf.sum()

        return da_wf.values
