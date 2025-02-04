"""
Reads TOML configuration
"""

import os

# use tomllib (only for Python >= 3.11) if available, otherwise use toml
try:
    import tomllib as toml

    MODE = "rb"
except ImportError:
    import toml  # type: ignore

    MODE = "r"


def read_config(filename):
    """
    Reads user configuration from TOML file.

    Parameters
    ----------
    filename: str
        Name of the TOML file

    Returns
    -------
    config: dict
        Configuration dictionary
    """

    if not os.path.isfile(filename):
        raise FileNotFoundError(f"Config file {filename} not found")

    with open(filename, MODE) as f:
        config = toml.load(f)

    check_config(config)

    return config


def check_config(config):
    """
    Check config file for consistency
    """

    # validity checks
    # make sure that cloudnet product is either caterogize or ecmwf
    if config["prepare"]["general"]["attenuation_correction_input"] not in [
        "cloudnet_categorize",
        "cloudnet_ecmwf",
    ]:
        raise ValueError(
            "attenuation_correction_input must be either "
            "'cloudnet_categorize' or 'cloudnet_ecmwf'"
        )

    # type checks
    # make sure that attenuation correction is boolean
    if not isinstance(
        config["prepare"]["general"]["attenuation_correction"], bool
    ):
        raise ValueError("attenuation_correction must be boolean")
