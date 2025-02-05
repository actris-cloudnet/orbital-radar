"""
Reads TOML configuration
"""

import os
import tomllib as toml


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

    with open(filename, "rb") as f:
        config = toml.load(f)

    return config
