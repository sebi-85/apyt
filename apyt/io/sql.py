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
__all__ = ["download", "query", "update"]
#
#
# import modules
import logging
import numpy as np
import requests
#
# import individual functions
from apyt.io.config import get_setting
from html2text import HTML2Text
from os.path import isfile
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
        logger.info(
            f"Downloading data for custom ID \"{custom_id}\" from database…"
        )
        # define structured dtype
        dt = np.dtype([
            ('U_base', np.float32), ('U_pulse', np.float32),
            ('U_reflectron', np.float32),
            ('x_det', np.float32), ('y_det', np.float32), ('tof', np.float32),
            ('epoch', np.int32),   ('pulse_num', np.uint32)
        ])
        #
        #
        # download file from database
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
        data = np.frombuffer(r.content, dtype = dt).copy()
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
        provided, it is automatically converted to a tuple.
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

    - The returned dictionary always includes the key ``'id'``.
    - If the field ``'custom_id'`` is present, it is converted to ``str`` (in
      case of a numeric-only custom ID).
    - Errors are logged with the module-level logger.
    """
    #
    #
    # normalize keys
    if isinstance(keys, str):
        keys = (keys,)
    #
    #
    # build SQL query
    sql = "SELECT id, " + ", ".join(keys) + " FROM data WHERE id = " + str(id)
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
        record = r.json()[id]
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
    # update database record
    logger.info(f"Updating \"{key}\" for record {id} in database… ")
    r = _request(
        get_setting("database.url") + "/update.php",
        {'id': id, 'key': key, 'value': value},
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
