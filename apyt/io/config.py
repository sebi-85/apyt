"""
The APyT configuration module
=============================

This module provides functions to load and access user-specific configuration
settings for the APyT software package. It automatically manages the location
and structure of the configuration file using platform-specific directories, and
creates a default configuration file if none is found.

The configuration file is written in TOML format and includes user-defined
settings such as database URLs, instrument parameters (e.g. flight length and
detector radius), and other metadata needed for data processing.

Configuration files are cached for performance and can be reloaded on demand.
Nested configuration settings can be accessed using dot notation.


Configuration file location
---------------------------

The configuration file is stored in a platform-specific user directory
(e.g. ``~/.config/apyt/config.toml`` on Linux). This location is determined
using the ``platformdirs`` package.


Default configuration structure
-------------------------------

The default configuration file (TOML format) has the following structure:

.. code-block:: toml

    [database]
    url = "https://apt-upload.mp.imw.uni-stuttgart.de"

    [devices.metap]
    flight_length   = 144.0
    detector_radius = 60.0

    [devices.micro]
    flight_length   = 106.0
    detector_radius = 37.5

    [devices.tap]
    flight_length   = 305.0
    detector_radius = 60.0


List of functions
-----------------

* :func:`get_setting`: Retrieve a nested setting from the configuration.
* :func:`load_config`: Load configuration from file (or cache if already
  loaded).


.. sectionauthor:: Sebastian M. Eich <Sebastian.Eich@imw.uni-stuttgart.de>
.. codeauthor::    Sebastian M. Eich <Sebastian.Eich@imw.uni-stuttgart.de>
"""
#
#
#
#
__version__ = "0.1.0"
__all__ = [
    "get_setting",
    "load_config"
]
#
#
#
#
# import modules
import logging
import tomllib
#
# import special functions
from pathlib import Path
from platformdirs import user_config_dir
#
#
#
#
# set up logger
logger = logging.getLogger(__name__)
#
#
#
#
################################################################################
#
# internal configuration
#
################################################################################
# configuration file name and path
_APP_NAME = "apyt"
_CONFIG_FILENAME = "config.toml"
_CONFIG_DIR = Path(user_config_dir(_APP_NAME))
_CONFIG_PATH = _CONFIG_DIR / _CONFIG_FILENAME
#
#
# default configuration content
_DEFAULT_CONFIG_TEXT = """
[database]
url = "https://apt-upload.mp.imw.uni-stuttgart.de"

# flight length and detector radius in mm
[devices.metap]
flight_length   = 144.0
detector_radius =  60.0

[devices.micro]
flight_length   = 106.0
detector_radius =  37.5

[devices.tap]
flight_length   = 305.0
detector_radius =  60.0
"""
#
#
#
#
################################################################################
#
# private module-level variables
#
################################################################################
# cached configuration dictionary
_config_cache = None
#
#
#
#
################################################################################
#
# public functions
#
################################################################################
def get_setting(key_path):
    """
    Retrieve a nested setting from the configuration.

    Parameters
    ----------
    key_path : str
        Dot-separated path to the config setting, e.g. ``"database.url"``.

    Returns
    -------
    Any
        The requested config value.

    Raises
    ------
    KeyError
        If the setting does not exist.
    """
    #
    #
    # load configuration
    config = load_config()
    #
    #
    # traverse along configuration dictionary
    for key in key_path.split("."):
        try:
            config = config[key]
        except(KeyError):
            raise KeyError(
                f"Setting \"{key_path}\" not found in configuration."
            )
    #
    #
    # return (nested) configuration setting
    return config
#
#
#
#
def load_config(force_reload = False):
    """
    Load configuration from file (or cache if already loaded).

    Parameters
    ----------
    force_reload : bool
        Whether to reload the configuration file from disk.

    Returns
    -------
    dict
        The parsed configuration dictionary.
    """
    #
    #
    # use global configuration cache
    global _config_cache
    #
    #
    # create configuration directory if not present
    if not _CONFIG_DIR.exists():
        logger.info(f"Creating configuration directory at \"{_CONFIG_DIR}\".")
        _CONFIG_DIR.mkdir(parents = True, exist_ok = True)
    #
    #
    # create default configuration file if not present
    if not _CONFIG_PATH.exists():
        logger.info(
            f"Creating default configuration file \"{_CONFIG_FILENAME}\"."
        )
        _CONFIG_PATH.write_text(_DEFAULT_CONFIG_TEXT, encoding = "utf-8")
    #
    #
    # load configuration from file
    if _config_cache is None or force_reload:
        logger.info(f"Loading configuration from \"{_CONFIG_PATH}\".")
        with _CONFIG_PATH.open("rb") as f:
            _config_cache = tomllib.load(f)
    else:
        logger.debug("Using cached configuration settings.")
    #
    #
    # return (cached) configuration
    return _config_cache
