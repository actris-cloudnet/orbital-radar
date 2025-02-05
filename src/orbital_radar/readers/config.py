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

    return config
