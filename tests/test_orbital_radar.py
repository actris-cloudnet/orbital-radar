from pathlib import Path

import netCDF4
import pytest

from orbital_radar import Suborbital

FILE_PATH = Path(__file__).parent.absolute()
MEAN_WIND = 6


@pytest.fixture(scope="class")
def simulated_cloudnet(tmp_path_factory: pytest.TempPathFactory) -> Path:
    tmp_dir = tmp_path_factory.mktemp("data")
    output_file = tmp_dir / "output.nc"

    suborbital = Suborbital()
    suborbital.simulate_cloudnet(
        categorize_filepath=str(FILE_PATH / "data/truncated-categorize.nc"),
        output_filepath=str(output_file),
        mean_wind=MEAN_WIND,
    )

    return output_file


class TestSimulateCloudnet:
    def test_uuid_is_not_none(self, simulated_cloudnet: Path) -> None:
        with netCDF4.Dataset(simulated_cloudnet) as nc:
            assert nc.file_uuid is not None

    def test_dimensions(self, simulated_cloudnet: Path) -> None:
        with netCDF4.Dataset(simulated_cloudnet) as nc:
            assert nc.dimensions["along_track"].size == 99
            assert nc.dimensions["along_track_sat"].size == 35
