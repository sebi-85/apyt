"""
The APyT local database module
==============================

This module provides a lightweight Python interface to a local database stored
in YAML format. It implements convenience functions for **downloading**,
**querying**, and **updating** measurement records.

The primary design goal is to provide a consistent API for accessing
experimental data in Python (see also :ref:`apyt.io.sql:The APyT SQL module`).


Configuration
-------------

General database settings are configured in the ``[localdb]`` section of the
global TOML configuration file
(see :ref:`apyt.io.config:Default configuration structure`).


Typical use cases
-----------------

- Fetching structured measurement data from the database for analysis.
- Updating metadata keys (e.g., experiment or evaluation parameters) of
  existing records.
- Executing queries for specific records and keys.


Example local database structure
--------------------------------

The local database is stored in YAML format. Each entry corresponds to a
measurement record and is indexed by a numeric key (the measurement ID).

.. code-block:: yaml

    1:
      file: W_calibration_tap_01.raw
      device: tap
      custom_id: W_calibration_tap_01
      parameters: {}

    2:
      file: my_measurement_01.raw
      device: metap
      custom_id: my_id_01
      parameters: {}


Explanation
^^^^^^^^^^^

- **Top-level keys** (``1``, ``2``, ...): Numeric identifiers representing
  measurement records. Each key corresponds to one entry in the local database.

- ``file``: Name of the raw measurement data file (relative to the configured
  ``localdb.data`` directory).

- ``device`` Identifier of the atom probe device. Must match one of the devices
  defined in the global configuration file (see
  :ref:`apyt.io.config:Default configuration structure`).

- ``custom_id``: A user-defined identifier for the measurement. This may be any
  string and is useful for referencing records by name rather than by numeric
  ID.
- ``parameters``: An initially empty dictionary that **must** be present for
  compatibility and defined as ``{}``.

The keys ``file``, ``device``, ``custom_id``, and ``parameters`` are
**mandatory**. Additional keys may be added as needed and will be populated
automatically by the APyT modules (e.g., analysis parameters or metadata).


List of functions
-----------------

* :func:`download`: Download measurement data from a local file.
* :func:`query`: Query one or more keys from a database record.
* :func:`update`: Update a specific key of a database record.


Implementation notes
--------------------

- Binary measurement datasets are read and converted directly into structured
  NumPy arrays.
- Error handling and logging are integrated throughout the code.
- The API is designed to be fully compatible with
  :ref:`apyt.io.sql:The APyT SQL module`.


.. sectionauthor:: Sebastian M. Eich <Sebastian.Eich@imw.uni-stuttgart.de>
.. codeauthor::    Sebastian M. Eich <Sebastian.Eich@imw.uni-stuttgart.de>
"""
#
#
__version__ = "0.1.0"
__all__ = ["download", "query", "update"]
#
#
# import modules
import logging
import numpy as np
import shutil
import yaml
#
# import individual functions
from apyt.io.config import _RAW_FILE_DTYPE, get_setting
from datetime import datetime
from os.path import expanduser, isfile
from pathlib import Path
from requests import codes
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
# public functions
#
################################################################################
def download(id):
    """
    Load measurement data from a local file.

    Retrieves measurement data stored in a local file associated with a database
    record. The data are returned as a structured NumPy array with predefined
    fields (e.g., detector positions, voltage signals, and timing information).


    Parameters
    ----------
    id : int
        The measurement ID of the record in the local database.

    Returns
    -------
    status : int
        Status code (for compatibility with
        :ref:`apyt.io.sql:The APyT SQL module`).

        - ``200`` indicates success.

    data : numpy.ndarray
        A structured NumPy array containing the measurement events.


    Raises
    ------

    FileNotFoundError
        If the local measurement file cannot be found.


    Notes
    -----

    - Errors are logged with the module-level logger.
    - The status code is returned only for compatibility with
      :ref:`apyt.io.sql:The APyT SQL module`.
    """
    #
    #
    # get record from local database
    record = _get_record(id)
    #
    #
    # check measurement data file
    data_file = \
        Path(expanduser(get_setting("localdb.data"))) / record['file']
    if not data_file.is_file():
        raise FileNotFoundError(
            f"Could not find local measurement file \"{data_file}\"."
        )
    #
    #
    # load data from file
    logger.info(f"Loading measurement data from local file \"{data_file}\".")
    data = np.fromfile(data_file, dtype = _RAW_FILE_DTYPE)
    #
    #
    # return data
    logger.info(f"Data file contains {len(data)} events.")
    return codes.ok, data
#
#
#
#
def query(id, keys):
    """
    Retrieve one or more keys from a local database record.

    This function reads a measurement entry from a local YAML database file and
    returns the requested keys as a dictionary mapping keys to values.


    Parameters
    ----------

    id : int
        The measurement ID of the record in the local database.
    keys : str or iterable of str
        The key(s) to retrieve from the database entry. If a single string is
        provided, it is automatically converted to a tuple.


    Returns
    -------

    status : int
        Status code for compatibility with
        :ref:`apyt.io.sql:The APyT SQL module`.

        - ``200`` indicates success.

    result : dict
        Dictionary containing the requested keys.


    Raises
    ------

    FileNotFoundError
        If the local database file does not exist.
    KeyError
        If the requested record or any requested key is missing.


    Notes
    -----

    - If the key ``'custom_id'`` is present, it is converted to ``str`` (in
      case of a numeric-only custom ID).
    - Errors are logged with the module-level logger.
    - The status code is returned only for compatibility with
      :ref:`apyt.io.sql:The APyT SQL module`.
    """
    #
    #
    # normalize keys
    if isinstance(keys, str):
        keys = (keys,)
    #
    #
    # get record from local database
    record = _get_record(id)
    #
    #
    # get requested keys
    try:
        record = {key: record[key] for key in keys}
    except KeyError as e:
        raise KeyError(f"Missing key \"{e.args[0]}\" in record {id}")
    #
    #
    # ensure custom_id is always string
    if 'custom_id' in record:
        record['custom_id'] = str(record['custom_id'])
    #
    #
    # return record
    return codes.ok, record
#
#
#
#
def update(id, key, value):
    """
    Update a specific key of a database record.

    This function modifies an existing entry in the local YAML database by
    updating a single key–value pair. Before updating, a timestamped backup of
    the database file is created.


    Parameters
    ----------

    id : int
        The measurement ID of the record in the local database.
    key : str
        The key name to update in the database entry.
    value : str
        The new value for the specified key.


    Returns
    -------

    status : int
        Status code for compatibility with
        :ref:`apyt.io.sql:The APyT SQL module`.

        - ``200`` indicates success.

    response : str
        ``"OK"`` when the update is successful.


    Notes
    -----
    - The status code is returned only for compatibility with
      :ref:`apyt.io.sql:The APyT SQL module`.
    """
    #
    #
    # get records from local database file
    db_file = expanduser(get_setting("localdb.file"))
    if not isfile(db_file):
        raise FileNotFoundError(
            f"Could not find local database file \"{db_file}\"."
        )
    logger.info(f"Reading records from local database file \"{db_file}\".")
    with open(db_file) as f:
        records = yaml.safe_load(f)
    #
    #
    # check whether record exists
    if id not in records:
        raise KeyError(f"Record {id} does not exist in local database.")
    #
    #
    # create backup file with current timestamp
    backup_file = f"{db_file}.{datetime.now().strftime('%Y%m%d_%H%M%S')}.bak"
    shutil.copy2(db_file, backup_file)
    logger.info(f"Created backup of database file at \"{backup_file}\".")
    #
    #
    # update key
    records[id][key] = value
    #
    # update complete database file
    with open(db_file, 'w') as f:
        yaml.safe_dump(
            records, f, default_flow_style = False, allow_unicode = True
        )
    logger.info(f"Updated key \"{key}\" for record {id}.")
    #
    #
    # return status
    return codes.ok, "OK"
#
#
#
#
################################################################################
#
# private module-level functions
#
################################################################################
def _get_record(id):
    """
    Simple function to obtain a single record from local database.
    """
    #
    #
    # get records from local database file
    db_file = expanduser(get_setting("localdb.file"))
    if not isfile(db_file):
        raise FileNotFoundError(
            f"Could not find local database file \"{db_file}\"."
        )
    with open(db_file) as f:
        records = yaml.safe_load(f)
    #
    #
    # check whether record exists
    logger.info(f"Reading record {id} from local database file \"{db_file}\".")
    if id not in records:
        raise KeyError(f"Record {id} does not exist in local database.")
    record = records[id]
    #
    #
    # check mandatory keys
    mandatory_keys = ('file', 'device', 'custom_id', 'parameters')
    for key in mandatory_keys:
        if key not in record:
            raise KeyError(f"Missing mandatory key '{key}' in record {id}")
    #
    #
    # return record
    return record
