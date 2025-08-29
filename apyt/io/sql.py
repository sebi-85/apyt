"""
The APyT SQL module
===================

This module provides a lightweight Python interface to the APyT SQL database. It
implements convenience functions for **downloading**, **querying**, and
**updating** measurement data stored in the central database.

The design goal of this module is to abstract away low-level HTTP/SQL details
and provide a consistent API for accessing experimental data in Python. In
particular:

* All functions can use explicit authorization credentials (username, password)
  if required for database access.
* Each function returns both the HTTP status code of the request and the
  retrieved or updated content (if applicable).
* Downloaded measurement data can optionally be cached on disk in NumPy `.npy`
  files to reduce repeated network requests.


Typical use cases
-----------------

- Fetching structured measurement data from the database for analysis.
- Updating metadata fields (e.g., experiment or evaluation parameters) of
  existing records.
- Executing SQL queries for specific columns and records.


List of functions
-----------------

* :func:`download`: Download and (optionally) cache measurement data from the
  database.
* :func:`dump_record`: Dump a single SQL database record to a JSON file.
* :func:`load_record`: Load a JSON record from file and upload it to the SQL
  database.
* :func:`query`: Query one or more fields from a SQL database record.
* :func:`update`: Update a specific field of a database record.


Implementation notes
--------------------

- Database access is performed via HTTP requests to PHP scripts
  (``download.php``, ``update.php``, ``query.php``) hosted at the configured
  database URL (see :func:`apyt.io.config.get_setting`).
- Binary measurement datasets are streamed and converted directly into NumPy
  arrays.
- Error handling and logging are integrated throughout; failed requests return
  the corresponding HTTP status code.


.. sectionauthor:: Sebastian M. Eich <Sebastian.Eich@imw.uni-stuttgart.de>
.. codeauthor::    Sebastian M. Eich <Sebastian.Eich@imw.uni-stuttgart.de>
"""
#
#
__version__ = "0.1.0"
__all__ = ["download", "dump_record", "load_record", "query", "update"]
#
#
# import modules
import fujson
import json
import logging
import numpy as np
import requests
import warnings
#
# import individual functions
from apyt.io.config import _RAW_FILE_DTYPE, get_setting
from datetime import datetime
from html2text import HTML2Text
from os.path import isfile
from pathlib import Path
from time import sleep
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
def download(id, use_cache = False, auth = None):
    """
    Download and (optionally) cache measurement data from the database.

    This function retrieves measurement data stored in the APyT SQL database.
    Data are returned as a structured NumPy array with predefined fields such as
    detector position, voltage signals, and timing information.

    To reduce repeated network requests, measurement data can be cached on disk
    in a local ``.npy`` file, identified by the database record's ``custom_id``.
    If caching is enabled, subsequent calls will load data directly from the
    cache instead of re-downloading.


    Parameters
    ----------
    id : int
        The measurement ID of the record in the SQL database.
    use_cache : bool, optional
        If ``True``, attempt to load data from a local cache file. If the cache
        does not exist, data will be downloaded and then written to disk. If
        ``False``, data are always fetched from the database. The latter is the
        default behavior.
    auth : tuple of (str, str), optional
        A tuple ``(username, password)`` providing authorization credentials. If
        ``None``, access to the database may fail, depending on its
        access/security configuration.


    Returns
    -------
    status : int
        HTTP status code returned by the database request.

        - ``200`` indicates success.
        - Other codes indicate failure.

    data : numpy.ndarray or None
        A structured NumPy array containing the measurement events with the
        following fields:

        - ``U_base`` : float32 — Base voltage
        - ``U_pulse`` : float32 — Pulse voltage
        - ``U_reflectron`` : float32 — Reflectron voltage
        - ``x_det`` : float32 — Detector *x*-position
        - ``y_det`` : float32 — Detector *y*-position
        - ``tof`` : float32 — Time-of-flight
        - ``epoch`` : int32 — Epoch time
        - ``pulse_num`` : uint32 — Pulse number

        Returns ``None`` if the download or query fails.


    Notes
    -----

    - Creates or reads a cache file ``<custom_id>.npy`` in the current working
      directory when ``use_cache = True``.
    - Requires network access to the SQL database unless loading from cache.
    - Logs progress and errors using the module-level logger.
    """
    #
    #
    # get custom ID from database
    status, record = query(id, "custom_id", auth = auth)
    if status != requests.codes.ok:
        logger.error(f"Failed to retrieve custom ID for record {id}.")
        return status, None
    custom_id = record['custom_id']
    #
    #
    # set cache file name
    cache_file = f"{custom_id}.npy"
    #
    #
    #
    #
    # download data from database
    if use_cache == False or isfile(cache_file) == False:
        # download file from database
        logger.info(
            f"Downloading data for custom ID \"{custom_id}\" from database…"
        )
        r = _request(
            get_setting("database.url") + "/download.php", {'id' : id}, auth
        )
        if r.status_code != requests.codes.ok:
            logger.error(
                f"Download failed for record {id} (HTTP {r.status_code})."
            )
            return r.status_code, None
        #
        #
        # copy buffer content to numpy array
        data = np.frombuffer(r.content, dtype = _RAW_FILE_DTYPE).copy()
        del r
    #
    #
    # get data from cache file
    if use_cache == True:
        # write cache file if it does not exist
        if isfile(cache_file) == False:
            logger.info(f"Writing data to cache file \"{cache_file}\".")
            with open(cache_file, 'wb') as f:
                np.save(f, data)
        # read cache file
        else:
            logger.info(f"Reading data from cache file \"{cache_file}\".")
            with open(cache_file, 'rb') as f:
                data = np.load(f)
    #
    #
    #
    #
    # return data
    logger.info(f"Data file contains {len(data)} events.")
    return requests.codes.ok, data
#
#
#
#
def dump_record(id, file_name = None):
    """
    Dump a single SQL database record to a JSON file.

    This function retrieves a measurement record from the SQL database and
    writes its content to a JSON file. If no output filename is provided, one
    will be generated automatically based on the record's ``custom_id`` and the
    current timestamp.


    Parameters
    ----------

    id : int
        The measurement ID of the record to retrieve.
    file_name : str or Path, optional
        The name of the output file. If ``None``, the filename is constructed as
        ``<custom_id>_<YYYYMMDD_HHMMSS>.rec``.


    Returns
    -------

    str or None
        The path to the created JSON file, or ``None`` if the query failed.

    Warns
    -----

    UserWarning
        If the SQL query does not succeed.
    """
    #
    #
    # retrieve database record
    logger.info(f"Retrieving record {id} from SQL database.")
    status, record = query(id, "*")
    if status != requests.codes.ok:
        warnings.warn(
            f"Failed to retrieve record {id} (status={status}).", UserWarning
        )
        return None
    #
    #
    # generate output filename if not provided
    if file_name is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        file_name = f"{record['custom_id']}_{timestamp}.rec"
    #
    #
    # write JSON data
    file_path = Path(file_name)
    logger.info(f"Writing record {id} to \"{file_path}\".")
    with open(file_path, "w", encoding = "utf-8") as f:
        fujson.dump(
            record, f, ensure_ascii = False, float_format = ".9e", indent = 4
        )
    #
    #
    # return path to the created JSON file
    return str(file_path)
#
#
#
#
def load_record(id, file_name, auth):
    """
    Load a JSON record from file and upload it to the SQL database.

    This function reads a JSON file containing record data and updates the
    corresponding entry in the SQL database. Each key--value pair is uploaded
    individually via the :func:`update` function, except for read-only fields.

    Parameters
    ----------

    id : int
        The measurement ID of the record in the SQL database.
    file_name : str or Path
        Path to the JSON file containing the record data.
    auth : tuple of (str, str)
        Authentication credentials (username, password) for the SQL database.


    Returns
    -------

    bool
        ``True`` if all fields were uploaded successfully, ``False`` if the
        input file is missing or if any upload fails.


    Warns
    -----

    UserWarning
        - If the input file does not exist.
        - If uploading any field fails.
    """
    #
    #
    # check file existence
    file_path = Path(file_name)
    if not file_path.is_file():
        warnings.warn(f"File \"{file_name}\" does not exist.", UserWarning)
        return False
    #
    #
    # read JSON data from file
    logger.info(f"Reading record from file \"{file_name}\".")
    with open(file_path, "r", encoding = "utf-8") as f:
        record = json.load(f)
    #
    #
    # upload parameters to SQL database
    ro_keys = ('id', 'last_modified', 'file', 'checksum', 'user')
    for key, value in record.items():
        # skip read-only and empty fields
        if key in ro_keys or value == "":
            continue
        #
        #
        # update field
        status, response = update(id, key, value, auth = auth)
        if status != requests.codes.ok or response != "OK":
            warnings.warn(
                f"Failed to upload \"{key}\" for record {id} "
                f"(status={status}, response={response}).",
                UserWarning
            )
            return False
        #
        #
        # throttle queries
        sleep(0.1)
    #
    #
    # return True on success
    return True
#
#
#
#
def query(id, keys, auth = None):
    """
    Query one or more fields from a SQL database record.

    This function retrieves specific fields of a measurement entry from the APyT
    SQL database. Results are returned as a dictionary mapping field names to
    values.


    Parameters
    ----------

    id : int
        The measurement ID of the record in the SQL database.
    keys : str or iterable of str
        The field(s) to retrieve from the database entry. If a single string is
        provided, it is automatically converted to a tuple. ``"*"`` retrieves
        all fields from the database entry.
    auth : tuple of (str, str), optional
        A tuple ``(username, password)`` providing authorization credentials. If
        ``None``, access to the database may fail, depending on its
        access/security configuration.


    Returns
    -------

    status : int
        HTTP status code returned by the database request.

        - ``200`` indicates success.
        - Other codes indicate failure (see ``requests.codes``).

    result : dict or None
        Dictionary containing the requested fields. Returns ``None`` if the
        request fails or the record is missing.


    Notes
    -----

    - If the field ``'custom_id'`` is present, it is converted to ``str`` (in
      case of a numeric-only custom ID).
    - Errors are logged with the module-level logger.
    """
    #
    #
    # build SQL query
    if keys == "*":
        sql = f"SELECT * FROM data WHERE id = {id}"
    elif isinstance(keys, str):
        sql = f"SELECT id, {keys} FROM data WHERE id = {id}"
    else:
        sql = f"SELECT id, {', '.join(keys)} FROM data WHERE id = {id}"
    payload = {"format": "json", "sql": sql}
    #
    #
    # get request response
    r = _request(get_setting("database.url") + "/query.php", payload, auth)
    if r.status_code != requests.codes.ok:
        logger.error(f"Query failed for record {id} (HTTP {r.status_code}).")
        return r.status_code, None
    #
    #
    # convert query results to dictionary
    try:
        record = r.json()[str(id)]
    except ValueError as e:
        logger.error(f"Failed to parse JSON response for record {id}: {e}")
        return r.status_code, None
    #
    #
    # ensure custom_id is always string
    if 'custom_id' in record:
        record['custom_id'] = str(record['custom_id'])
    #
    #
    # return query results dictionary
    return requests.codes.ok, record
#
#
#
#
def update(id, key, value, auth = None, method = 'GET'):
    """
    Update a specific field of a database record.

    This function modifies an existing entry in the APyT SQL database by
    updating a single key–value pair. The update request is sent via HTTP
    (either GET or POST) to the configured database endpoint.


    Parameters
    ----------

    id : int
        The measurement ID of the record in the SQL database.
    key : str
        The field name to update in the database entry.
    value : str
        The new value for the specified field.
    auth : tuple of (str, str), optional
        A tuple ``(username, password)`` providing authorization credentials. If
        ``None``, access to the database may fail, depending on its
        access/security configuration.
    method : {'GET', 'POST'}, default 'GET'
        The HTTP request method to use when submitting the update.


    Returns
    -------

    status : int
        HTTP status code returned by the database request.

        - ``200`` indicates success.
        - Other codes indicate failure.

    response : str or None
        The raw text returned by the database endpoint. Returns ``None`` if the
        request fails.


    Notes
    -----

    - The database endpoint returns ``"OK"`` when the update is successful.
    - Errors returned by the database are converted to readable text and logged.
    """
    #
    #
    # check for valid method
    if method not in ('GET', 'POST'):
        logger.error(f"Invalid method '{method}'. Must be 'GET' or 'POST'.")
        return requests.codes.bad_request, None
    #
    #
    # floats in "parameters" field shall be in scientific notation
    if key == "parameters":
        value = fujson.dumps(value, ensure_ascii = False, float_format = ".9e")
    #
    #
    # update database record (value needs to be quoted)
    logger.info(f"Updating \"{key}\" for record {id} in database… ")
    r = _request(
        get_setting("database.url") + "/update.php",
        {'id': id, 'key': key, 'value': f"'{value}'"},
        auth, method
    )
    if r.status_code != requests.codes.ok:
        logger.error(
            f"Update request failed for record {id} (HTTP {r.status_code})."
        )
        return r.status_code, None
    #
    #
    # check server response
    response = r.text
    if response == "OK":
        logger.info(f"SQL update successful for record {id}.")
    else:
        logger.error(
            "SQL update failed. Server returned:\n\n" +
            HTML2Text().handle(response)
        )
    #
    #
    # return response
    return requests.codes.ok, response
#
#
#
#
################################################################################
#
# private module-level functions
#
################################################################################
def _request(request_url, payload, auth = None, method = "GET"):
    """
    Internal helper: send an HTTP request to the database.


    Parameters
    ----------

    request_url : str
        Full URL of the database endpoint.
    payload : dict
        Dictionary of request parameters or data.
    auth : tuple of (str, str), optional
        Authorization credentials.
    method : {'GET', 'POST'}, default 'GET'
        HTTP request method.


    Returns
    -------

    requests.Response
        The HTTP response object.
    """
    #
    #
    # get request response
    try:
        if method == "GET":
            r = requests.get(
                request_url, params = payload, auth = auth, timeout = 30
            )
        elif method == "POST":
            r = requests.post(
                request_url + "?method=POST", data = payload,
                auth = auth, timeout = 30
            )
        else:
            raise ValueError(
                f"Unsupported request method \"{method}\". "
                "Expected \"GET\" or \"POST\"."
            )
    except requests.RequestException as e:
        logger.error(f"Request {method} {request_url} failed: {e}")
        raise
    #
    #
    # check request status code
    if r.status_code == requests.codes.ok:
        logger.debug(f"Request {method} {request_url} succeeded.")
    else:
        logger.error(
           f"Request {method} {request_url} failed (HTTP {r.status_code})."
    )
    #
    #
    # return response
    return r
