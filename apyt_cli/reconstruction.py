#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#
desc = """
Reconstruct three-dimensional atomic tip coordinates from raw measurement data.

This script performs a spatial reconstruction of atom probe data, using the
chemical identification obtained from a previously fitted mass spectrum (via the
``apyt_spectrum_fit`` script).

Two reconstruction modes are available:
  1. **Classic reconstruction scheme**
  2. **Taper geometry reconstruction**

This script serves as a command line wrapper for the APyT Python reconstruction
module. For more details and theoretical background, see:
https://apyt.mp.imw.uni-stuttgart.de
"""
#
#
#
#
# import modules
import apyt.io.localdb as localdb
import apyt.io.sql as sql
import apyt.spectrum.align as ms
import apyt.spectrum.fit as mf
import apyt.reconstruction.basic as rec
import argparse
import logging
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
#
# import individual functions/modules
from apyt.gui.forms import login
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import Button, CheckButtons, TextBox
from timeit import default_timer as timer
from tkinter import messagebox
from ttkthemes import ThemedTk
#
#
#
#
# default reconstruction parameters
_REC_DEFAULTS = {
    'background'       : False,
    'beta'             : 3.3,
    'efficiency'       : 0.5,
    'export_volumes'   : False,
    'image_compression': 1.43,
    'module'           : "taper",
    'split_charge'     : False,
    'split_molecules'  : [False, False],
    'classic'          : {
        'E_0': 30.0
    },
    'taper'            : {
        'alpha': {
            'value': 10.0,
            'fixed': False
        },
        'beta': {
            'fixed': True
        },
        'field_align': {
            'E_0': 30.0,
            'min':  0.0,
            'max':  1.0
        },
        'r_0': {
            'value': 60.0,
            'fixed': False
        }
    }
}
#
#
#
#
# set up logger
logger = logging.getLogger(__name__)
logging.basicConfig(level = logging.INFO)
#
#
#
#
# program's arguments
parser = argparse.ArgumentParser(
    description = desc, add_help = False,
    formatter_class = argparse.RawTextHelpFormatter
)
parser.add_argument(
    "id", type = int,
    help = """\
The record ID in the APT measurement database.
"""
)
parser.add_argument(
    "-c", "--cache", action = 'store_true',
    help = """\
Read data from a cached NumPy binary file instead of
downloading it from the database (recommended if the
script is run multiple times on the same dataset). The
cache file is named <custom_id>.npy, where <custom_id>
is retrieved from the SQL database for the specified
record ID. If the file does not exist, it is downloaded
once during the first run and reused in subsequent runs
to reduce network traffic.
"""
)
parser.add_argument(
    "-d", "--debug", action = 'store_true',
    help = """\
Whether to print debug messages.
"""
)
parser.add_argument(
    "-h", "--help", action = 'help',
    default = argparse.SUPPRESS,
    help = """\
Show this help message and exit.
"""
)
parser.add_argument(
    "-m", "--module", metavar = "<classic|taper>", type = str, default = None,
    help = """\
The reconstruction module to use. Defaults to 'taper'.
"""
)
parser.add_argument(
    "--no-sql", action = 'store_true',
    help = """\
Whether to read records from a local database instead
of the SQL database. Measurement data will also be read
from a local file.
"""
)
#
#
# parse and print arguments
args = parser.parse_args()
print("\nSettings used:")
for key, value in sorted(vars(args).items()):
    print("{0:8s}{1}".format(key, value))
print("")
#
#
#
#
# initialize authorization credentials
auth = None
#
#
# pass debug flag to reconstruction module
rec.enable_debug(args.debug)
#
#
# initialize random number generator
rng = np.random.default_rng(seed = 0)
#
#
#
#
def apply_filters(arr_list, id_xyz_ref):
    """
    Apply filters to arrays associated with reconstructed data.

    *arr_list* is either a list of arrays or a single array to be filtered.
    Various filter options should be implemented here to be used globally.
    *id_xyz_ref* is a tuple containing the reference IDs and reconstructed xyz
    tip positions.
    """
    #
    #
    # convert single input array to list if required
    if not (is_list := isinstance(arr_list, list)):
        arr_list = [arr_list]
    #
    #
    # unpack reference data
    ids_f, xyz_tip_f = id_xyz_ref
    #
    #
    #
    #
    # split molecules if requested; needs to be run first to preserve background
    # (id = 0)
    if cb_molecule_split.get_status()[0]:
        ids_f, xyz_tip_f = mf.split_molecules(
            ids_f, xyz_tip_f, peaks_list,
            group_charge = not cb_charge_split.get_status()[0],
            shuffle = cb_molecule_split.get_status()[1],
            verbose = True
        )
        #
        #
        # two points must be considered when molecules are split in connection
        # with additional filtering:
        # (1) if the input array list contains the IDs and/or xyz tip positions,
        #     the reference must be updated *after* molecule splitting, and
        # (2) the input array must not contain any array other than the IDs or
        #     xyz tip positions
        for i in range(len(arr_list)):
            is_in_ref = False
            # loop through reference data
            for j in range(len(id_xyz_ref)):
                # update reference if found
                if arr_list[i] is id_xyz_ref[j]:
                    arr_list[i] = (ids_f, xyz_tip_f)[j]
                    is_in_ref   = True
                    break
            # raise exception if not found in reference data
            if is_in_ref == False:
                raise Exception(
                    "Internal error: Input array list to be filtered must not "
                    "contain other arrays than the IDs and reconstructed "
                    "positions when used in connection with molecule splitting."
                )
    #
    #
    #
    #
    # filter background atoms
    if not cb_background.get_status()[0]:
        logger.info("Filtering background atoms…")
        mask     = (ids_f != 0)
        arr_list = [arr[mask] for arr in arr_list]
        logger.info(f"{len(arr_list[0])} atoms remaining.")
    #
    #
    # return filtered arrays
    return arr_list if is_list else arr_list[0]
#
#
#
#
def db_download(id, use_cache):
    """
    Download measurement data from the database.

    This is a simple wrapper around the `apyt.io.localdb.download` and
    `apyt.io.sql.download` functions, respectively.
    """
    #
    #
    # load file from local database if requested
    if args.no_sql:
        status, data = localdb.download(id)
        return data
    #
    #
    #
    #
    # first try with stored credentials (may be None)
    logger.info(f"Downloading measurement data for database record {id}…")
    global auth
    status, data = sql.download(id, use_cache = use_cache, auth = auth)
    if status == 200:
        logger.info(f"Download for record {id} succeeded.")
        return data
    #
    #
    #
    #
    # create dummy root window to load custom Tk theme
    root = ThemedTk(theme = "breeze")
    root.withdraw()
    #
    #
    # retry loop with interactive login
    while True:
        # get login credentials
        auth = login()
        #
        # download measurement data
        status, data = sql.download(id, use_cache = use_cache, auth = auth)
        if status == 200:
            root.destroy()
            logger.info(f"Download for record {id} succeeded.")
            return data
        #
        #
        # ask for retry
        retry = messagebox.askyesno(
            "Download failed.", "Do you want to retry?", parent = root
        )
        if not retry:
            root.destroy()
            logging.error(f"Download for record {id} failed.")
            raise Exception(f"Download for record {id} failed.")
#
#
#
#
def db_query(id, fields):
    """
    Query one or more fields from a database record.

    This is a simple wrapper around the `apyt.io.localdb.query` and
    `apyt.io.sql.query` functions, respectively.
    """
    #
    #
    # load record from local database if requested
    if args.no_sql:
        status, record = localdb.query(id, fields)
        return record
    #
    #
    #
    #
    # first try with stored credentials (may be None)
    logger.info(f"Querying database record {id} for fields {fields}…")
    global auth
    status, record = sql.query(id, fields, auth = auth)
    if status == 200:
        logger.info(f"SQL query for record {id} succeeded.")
        return record
    #
    #
    #
    #
    # create dummy root window to load custom Tk theme
    root = ThemedTk(theme = "breeze")
    root.withdraw()
    #
    #
    # retry loop with interactive login
    while True:
        # get login credentials
        auth = login()
        #
        # query database
        status, record = sql.query(id, fields, auth = auth)
        if status == 200:
            root.destroy()
            logger.info(f"SQL query for record {id} succeeded.")
            return record
        #
        #
        # ask for retry
        retry = messagebox.askyesno(
            "SQL query failed.", "Do you want to retry?", parent = root
        )
        if not retry:
            root.destroy()
            logging.error(f"SQL query for record {id} failed.")
            raise Exception(f"SQL query for record {id} failed.")
#
#
#
#
def db_upload():
    """
    Upload fit parameters to the database.

    This is a simple wrapper around the `apyt.io.localdb.update` and
    `apyt.io.sql.update` functions, respectively.
    """
    #
    #
    # set numeric values and check box states
    rec_params['background']        = cb_background.get_status()[0]
    rec_params['beta']              = float(tb_beta.text)
    rec_params['efficiency']        = float(tb_efficiency.text)
    rec_params['export_volumes']    = cb_export_volumes.get_status()[0]
    rec_params['image_compression'] = float(tb_image_compression.text)
    rec_params['split_charge']      = cb_charge_split.get_status()[0]
    rec_params['split_molecules']   = cb_molecule_split.get_status()
    #
    # only update parameters for currently *active* reconstruction module
    if rec_params['module'] == "classic":
        rec_params['classic'] = {
            'E_0': float(tb_field_target.text)
        }
    elif rec_params['module'] == "taper":
        rec_params['taper'] = {
            'r_0' : {
                'value': float(tb_radius.text),
                'fixed': cb_radius.get_status()[0]
            },
            'alpha': {
                'value': float(tb_alpha.text),
                'fixed': cb_alpha.get_status()[0]
            },
            'beta': {
                'fixed': cb_beta.get_status()[0]
            },
            'field_align': {
                'E_0': float(tb_field_target.text),
                'min': float(tb_field_align_min.text),
                'max': float(tb_field_align_max.text)
            }
        }
    #
    #
    #
    #
    # create dummy root window to load custom Tk theme
    root = ThemedTk(theme = "breeze")
    root.withdraw()
    #
    #
    # only upload data if requested
    if not messagebox.askyesno(
        "Database upload",
        "Do you want to upload your reconstruction parameters to the database?",
        parent = root
    ):
        root.destroy()
        return
    #
    #
    #
    #
    # update record in local database if requested
    if args.no_sql:
        localdb.update(args.id, "parameters", record['parameters'])
        root.destroy()
        return
    #
    #
    #
    #
    # upload always requires credentials
    global auth
    if auth == None:
        auth = login()
    #
    #
    # update SQL record
    while True:
        status, response = sql.update(
            args.id, "parameters", record['parameters'],
            auth = auth, method = 'POST'
        )
        #
        # check for success
        if status == 200 and response == "OK":
            root.destroy()
            logger.info("Upload of parameters to SQL database succeeded.")
            return
        #
        #
        retry = messagebox.askyesno(
            "Upload failed", "Do you want to retry?", parent = root
        )
        if not retry:
            root.destroy()
            logging.error("Upload of parameters to SQL database failed.")
            return
        #
        #
        # retry with new authorization credentials
        auth = login()
#
#
#
#
def export(_):
    """
    Export reconstructed data.
    """
    #
    #
    if 'xyz_tip' in globals():
        logger.info(f"Exporting reconstructed tip to \"{custom_id}_xyz.txt\"…")
        #
        # initialize export data and export format
        export_data = [ids, xyz_tip]
        fmt = "%d\t%+.6e\t%+.6e\t%+.6e"
        #
        # add atomic volumes to export data if requested
        if cb_export_volumes.get_status()[0]:
            export_data.append(Ω)
            fmt += "\t%.6e"
        #
        #
        # filter export data
        export_data = np.column_stack(
            apply_filters(export_data, (ids, xyz_tip))
        )
        np.savetxt(
            custom_id + "_xyz.txt", export_data,
            fmt = fmt, comments = "",
            header = "{0:d}\n".format(len(export_data))
        )
#
#
#
#
def get_geometry_classic(_):
    """
    Calculate tip geometry through classical scheme.
    """
    #
    #
    # make calculated z-positions, radii of curvature, and evaporation fields
    # available globally
    global z, r, E
    #
    #
    #
    #
    # get voltage from input data
    U = data[:, 0]
    #
    #
    # set evaporation fields (constant for all events in classic scheme)
    E = np.full_like(U, float(tb_field_target.text))
    #
    #
    # calculate radii of curvature (convert to float64 for internal consistency)
    r = (
        U / (float(tb_beta.text) * float(tb_field_target.text))
    ).astype(np.float64)
    #
    #
    # calculate tip geometry
    start = timer()
    z, r = rec.get_geometry_classic(
        Ω / float(tb_efficiency.text), r, get_aperture_angle()
    )
    logger.info(f"Calculation of tip geometry took {timer() - start:.3f}s.")
    #
    #
    #
    #
    # show resulting plots
    plot(z, r, U, E)
#
#
#
#
def get_geometry_taper(_):
    """
    Calculate taper geometry with optional automatic evaporation field
    alignment.
    """
    #
    #
    # make calculated z-positions, radii of curvature, and evaporation fields
    # available globally
    global z, r, E
    #
    #
    #
    #
    # parse all text box values
    #
    # taper geometry parameters
    r_0 = float(tb_radius.text)
    θ   = np.deg2rad(90.0 - float(tb_alpha.text) / 2.0)
    β   = float(tb_beta.text)
    #
    # field alignment parameters
    field_id_min = float(tb_field_align_min.text)
    field_id_max = float(tb_field_align_max.text)
    E_0          = float(tb_field_target.text)
    #
    #
    #
    #
    # get voltage from input data
    U = data[:, 0]
    #
    # calculate cumulated reconstructed volume (account for detection
    # efficiency)
    V = np.cumsum(Ω) / float(tb_efficiency.text)
    #
    #
    # dictionary with parameter names as keys and tuples containing the
    # parameter value and its variation flag as values
    field_params = {
        'r_0': (r_0, not cb_radius.get_status()[0]),
        'θ'  : (θ,   not cb_alpha.get_status()[0]),
        'β'  : (β,   not cb_beta.get_status()[0])
    }
    #
    #
    #
    #
    # only align field automatically if at least one variation flag is set
    for val in field_params.values():
        if val[1] == True:
            # start timer
            start = timer()
            #
            # limit range for evaporation field alignment
            sl = np.s_[int(field_id_min * len(V)) : int(field_id_max * len(V))]
            #
            # get optimized field alignment parameters
            r_0, θ, β = rec.align_evaporation_field(
                V[sl], U[sl], field_params, E_0, get_aperture_angle()
            )
            logger.info(
                f"Automatic field alignment took {timer() - start:.3f}s."
            )
            #
            # update text boxes with optimized parameters (disconnect observers
            # to avoid triggering of infinite loop if values change)
            global cid_tb_radius, cid_tb_alpha, cid_tb_beta
            tb_radius.disconnect(cid_tb_radius)
            tb_alpha.disconnect(cid_tb_alpha)
            tb_beta.disconnect(cid_tb_beta)
            #
            # set text box values
            tb_radius.set_val("{0:.4f}".format(r_0))
            tb_alpha.set_val("{0:.4f}".format(np.rad2deg(np.pi - 2.0 * θ)))
            tb_beta.set_val("{0:.4f}".format(β))
            #
            # reconnect observers
            cid_tb_radius = tb_radius.on_submit(get_geometry)
            cid_tb_alpha  = tb_alpha.on_submit(get_geometry)
            cid_tb_beta   = tb_beta.on_submit(get_geometry)
            #
            #
            # exit loop
            break
    #
    #
    #
    #
    # calculate taper geometry
    start = timer()
    z, r = rec.get_geometry_taper(V, r_0, θ, get_aperture_angle())
    logger.info(f"Calculation of taper geometry took {timer() - start:.3f}s.")
    #
    #
    #
    #
    # calculate evaporation fields
    start = timer()
    E = rec.get_evaporation_field(U, r, β)
    logger.info(
        f"Calculation of evaporation fields took {timer() - start:.3f}s."
    )
    #
    #
    #
    #
    # show resulting plots
    plot(z, r, U, E)
#
#
#
#
def get_aperture_angle():
    """
    Calculate (half) aperture angle based on device geometry and image
    compression.
    """
    #
    #
    θ_m = float(tb_image_compression.text) * data_filter['detector_radius'] / L
    logger.info(f"Full aperture angle is {2.0 * np.rad2deg(θ_m):.2f}°.")
    return θ_m
#
#
#
#
def map_ids(_):
    """
    Perform chemical mapping and assignment of atomic volumes.

    Note that the chemical IDs change according to the setting of the charge
    splitting option. However, atomic volumes are unaffected by the charge
    splitting option and need to be calculated in principle only once.
    """
    #
    #
    global ids, Ω
    ids, Ω = mf.map_ids(
        mc_ratio, rng_id_mapping, bin_centers, peaks_list,
        spectrum_fit['info']['function'], spectrum_fit['values'],
        group_charge = not cb_charge_split.get_status()[0], verbose = True
    )
#
#
#
#
def plot(z, r, U, E, num_points = 1000):
    """
    Show various plots for evaporation field alignment.
    """
    #
    #
    # define slice object to reduce data points in plots
    sl = np.s_[::len(z) // num_points]
    #
    #
    # calculate relative event id
    i = np.arange(len(z)) / len(z)
    #
    #
    # define mapping functions for conversion between z-position and event id
    # (required for second x-axis)
    id2z = np.poly1d(np.polyfit(i[sl], z[sl], 3))
    z2id = np.poly1d(np.polyfit(z[sl], i[sl], 3))
    #
    #
    #
    #
    # clear complete plot
    ax_z.cla()
    ax_radius.cla()
    ax_voltage.cla()
    ax_field.cla()
    #
    #
    # plot z-position vs. event id
    ax_z.set_xlabel("Rel. event id")
    ax_z.set_ylabel("$z$ (nm)")
    ax_z.plot(i[sl], z[sl])
    #
    #
    # plot radius vs. z-position
    ax_radius.set_xlabel("$z$ (nm)")
    ax_radius.set_ylabel("$r$ (nm)")
    ax_radius_top = ax_radius.secondary_xaxis('top', functions = (z2id, id2z))
    ax_radius_top.set_xlabel("Rel. event id")
    ax_radius.plot(z[sl], r[sl])
    #
    #
    # plot voltage vs. z-position
    ax_voltage.set_xlabel("$z$ (nm)")
    ax_voltage.set_ylabel("$U$ (kV)")
    ax_voltage_top = ax_voltage.secondary_xaxis('top', functions = (z2id, id2z))
    ax_voltage_top.set_xlabel("Rel. event id")
    ax_voltage.plot(z[sl], U[sl] / 1e3)
    #
    #
    # plot evaporation field vs. z-position
    ax_field.set_xlabel("$z$ (nm)")
    ax_field.set_ylabel("$E$ (V/nm)")
    ax_field_top = ax_field.secondary_xaxis('top', functions = (z2id, id2z))
    ax_field_top.set_xlabel("Rel. event id")
    ax_field.plot(z[sl], E[sl])
    #
    # add reference lines for automatic field alignment in taper geometry
    if rec_params['module'] == 'taper':
        E_mean = np.average(
            E[int(float(tb_field_align_min.text) * len(E)):
            int(float(tb_field_align_max.text) * len(E))]
        )
        ax_field.plot(
            (id2z(float(tb_field_align_min.text)),
            id2z(float(tb_field_align_max.text))),
            (E_mean, E_mean),
            linestyle = '--', color = 'black'
        )
        ax_field.axvline(
            x = id2z(float(tb_field_align_min.text)),
            linestyle = '--', color = 'black'
        )
        ax_field.axvline(
            x = id2z(float(tb_field_align_max.text)),
            linestyle = '--', color = 'black'
        )
    #
    #
    # update plot
    fig.canvas.draw_idle()
#
#
#
#
def reconstruct(_):
    """
    Reconstruct and render three-dimensional tip data.
    """
    #
    #
    # make reconstructed tip data available globally
    global xyz_tip
    #
    #
    # start timer
    start = timer()
    #
    # reconstruct tip data
    xyz_tip = rec.reconstruct(
        data[:, 1:3], z, r, L, float(tb_image_compression.text)
    )
    logger.info(
        f"Reconstruction of {len(xyz_tip) / 1e6:.1f}M events took "
        f"{timer() - start:.3f}s."
    )
    #
    #
    # render three-dimensional tip
    pv.PolyData(apply_filters(xyz_tip, (ids, xyz_tip))).plot(
        render_points_as_spheres = True, point_size = 0.5
    )
#
#
#
#
def set_defaults(config, defaults):
    """
    Recursively update config with missing keys from defaults.
    """
    #
    #
    # loop through all default values
    for key, default_value in defaults.items():
        if key not in config:
            # if the key is missing, take the default
            config[key] = default_value
        else:
            # if both values are dicts, recurse
            if isinstance(default_value, dict) and \
               isinstance(config[key], dict):
                set_defaults(config[key], default_value)
    #
    #
    return config
#
#
#
#
################################################################################
###                                                                          ###
###     DATA PROCESSING / FILTERING                                          ###
###                                                                          ###
################################################################################
# get custom ID and spectrum parameters
record = db_query(args.id, ("custom_id", "parameters"))
#
#
# do some simple error and compatibility checks
if 'spectrum_fit' not in record['parameters']:
    raise Exception(
        "Could not find fitting parameters for mass spectrum in database entry."
    )
mf.check_compatibility(
    record['parameters']['spectrum_fit']['info']['version']
)
if args.module is not None and args.module not in ("classic", "taper"):
    raise Exception(f"Unknown reconstruction module \"{args.module}\".")
#
#
# parse database entries
custom_id       = record['custom_id']
data_filter     = record['parameters']['data_filter']
spectrum_fit    = record['parameters']['spectrum_fit']
spectrum_params = record['parameters']['spectrum_params']
#
#
# get reconstruction parameters from database or initialize with default values
if 'reconstruction' not in record['parameters']:
    record['parameters']['reconstruction'] = _REC_DEFAULTS
else:
    set_defaults(record['parameters']['reconstruction'], _REC_DEFAULTS)
#
# set shorthand for reconstruction parameters
rec_params = record['parameters']['reconstruction']
#
#
# set reconstruction module if provided explicitly
if args.module is not None:
    rec_params['module'] = args.module
get_geometry = globals()["get_geometry_" + rec_params['module']]
#
#
#
#
# download data from database
data = db_download(args.id, args.cache)
# convert structured to regular array (this interprets epoch and pulse number
# incorrect as float32, but we drop these anyway)
data = data.view((data.dtype[0], len(data.dtype.names)))
#
#
# sum voltages and pick relevant columns
data[:, 0] = \
    data[:, 0] + spectrum_params['pulse_coupling'] * data[:, 1] + \
    spectrum_params['detector_voltage']
data = data[:, [0, 3, 4, 5]]
#
#
# store initial number of events
len_init = len(data)
#
#
#
#
# filter measurement interval
if 'interval' in data_filter:
    logger.info(f"Using measurement interval {data_filter['interval']}.")
    data = data[int(data_filter['interval'][0] * len(data)) :
                int(data_filter['interval'][1] * len(data))]
#
#
# filter detector range
if 'detector_radius' in data_filter:
    # get radius
    r = data_filter['detector_radius']
    logger.info(f"Using detector radius {r:.1f} mm.")
    #
    # filter data
    data = data[(data[:, 1]**2 + data[:, 2]**2 <= r**2)]
#
#
# filter voltage range
if 'voltage_range' in data_filter:
    logger.info(f"Using voltage range {data_filter['voltage_range']} kV.")
    data = data[(data_filter['voltage_range'][0] * 1000 <= data[:, 0]) &
                (data[:, 0] <= data_filter['voltage_range'][1] * 1000)]
#
#
# print number of remaining/filtered events
logger.info(
    f"{len(data)} events remaining ({len_init - len(data)} events filtered)."
)
#
#
#
#
# calculate distance between tip and detector (mm) (we assume that the voltage
# is exact and that alpha only corrects the distance between tip and detector;
# see also documentation of mass spectrum module)
L = spectrum_params['L_0'] / np.sqrt(spectrum_params['alpha'])
logger.info(
    "Corrected distance between tip and detector from mass spectrum "
    f"calibration is {L:.3f} mm."
)
#
#
#
#
# calculate mass spectrum
logger.info("Calculating mass spectrum…")
_, bin_centers, mc_ratio = ms.get_mass_spectrum(
    data, (
        np.float32(spectrum_params['t_0']),
        np.float32(spectrum_params['L_0']), (
            np.asarray(spectrum_params['voltage_coeffs'], dtype = np.float32),
            np.asarray(spectrum_params['flight_coeffs'],  dtype = np.float32)),
        np.float32(spectrum_params['alpha'])),
    hist = {
        'range': data_filter['mass_charge_range'],
        'width': spectrum_params['bin_width']
    }
)
#
#
# filter data based on mass-to-charge ratio range (infer range from returned
# histogram bin centers)
logger.info("Filtering data based on mass-to-charge ratio range…")
len_init = len(data)
mask     = \
    (bin_centers[0] - spectrum_params['bin_width'] / 2 <= mc_ratio) & \
    (mc_ratio <= bin_centers[-1] + spectrum_params['bin_width'] / 2)
mc_ratio = mc_ratio[mask]
data     = data[mask]
logger.info(
    f"{len(data)} events remaining ({len_init - len(data)} events filtered)."
)
#
#
#
#
# set random numbers for chemical mapping (should be set only once to keep
# determinism if mapping is applied multiple times)
rng_id_mapping = rng.random(len(mc_ratio))
#
#
#
#
# loop through elements/molecules to get list of all peak dictionaries
peaks_list = []
for element in spectrum_fit['peaks'].values():
    # loop through all charge states
    for charge_state in element.values():
        # charge state contains list of all isotopes
        peaks_list.extend(charge_state)
#
#
#
#
# print counts and fractions (for reference only)
mf.counts(
    peaks_list, spectrum_fit['info']['function'], spectrum_fit['values'],
    tuple(bin_centers[[0, -1]]), spectrum_params['bin_width'],
    verbose = True
)
#
#
#
#
################################################################################
###                                                                          ###
###     PLOT SETUP                                                           ###
###                                                                          ###
################################################################################
# create figure
fig = plt.figure(figsize = (6.4 * 2, 4.8 * 2))
#
# set figure title
fig.suptitle("{0:s} ({1:.1f} M events)".format(custom_id, len(data) / 1e6))
#
#
#
#
# set relative widths and heights of subplot elements
h_fig     = 4 # relative height of figure
h_spacing = 1 # relative height of spacing between figures
#
w_fig     = 3 # relative width of figure
w_spacing = 1 # relative width of spacing between figures
w_tb      = 2 # relative width of text boxes
#
#
# create GridSpec for subplots
gs = GridSpec(
    3, 6, hspace = 0.0, wspace = 0.0,
    left = 0.07, top = 0.90, bottom = 0.11, right = 0.92,
    height_ratios = [h_fig / 3.0, h_spacing,  h_fig],
    width_ratios  = [w_fig, w_spacing, w_fig, w_spacing, w_fig, w_tb]
)
#
#
# create axes for z-position, radius, voltage, and evaporation field
ax_z       = plt.subplot(gs[0, 0])
ax_radius  = plt.subplot(gs[0, 2])
ax_voltage = plt.subplot(gs[0, 4])
ax_field   = plt.subplot(gs[2, 0:6])
#
#
# get bounds of evaporation field plot to align text boxes
ll, bb, ww, hh = ax_field.get_position().bounds
#
# set width and height of text boxes
w_tb = 0.060
h_tb = 0.025
#
#
# create text box for detection efficiency
tb_efficiency = TextBox(
    plt.axes([ll + ww - w_tb, bb + hh + 14 * h_tb, w_tb, h_tb]),
    label = "$\\zeta$", label_pad = 0.05,
    initial = "{0:.3f}".format(rec_params['efficiency'])
)
# create text box for image compression factor
tb_image_compression = TextBox(
    plt.axes([ll + ww - w_tb, bb + hh + 13 * h_tb, w_tb, h_tb]),
    label = "$\\xi$", label_pad = 0.05,
    initial = "{0:.3f}".format(rec_params['image_compression'])
)
# create text box for field factor
tb_beta = TextBox(
    plt.axes([ll + ww - w_tb, bb + hh + 9.5 * h_tb, w_tb, h_tb]),
    label = "$\\beta$", label_pad = 0.05,
    initial = "{0:.3f}".format(rec_params['beta'])
)
#
#
# create text boxes and check buttons for taper geometry
if rec_params['module'] == 'taper':
    tb_radius = TextBox(
        plt.axes([ll + ww - w_tb, bb + hh + 11.5 * h_tb, w_tb, h_tb]),
        label = "$r_0$ (nm)", label_pad = 0.05,
        initial = "{0:.3f}".format(
            rec_params['taper']['r_0']['value']
        )
    )
    tb_alpha = TextBox(
        plt.axes([ll + ww - w_tb, bb + hh + 10.5 * h_tb, w_tb, h_tb]),
        label = "$\\alpha$ (°)", label_pad = 0.05,
        initial = "{0:.3f}".format(
            rec_params['taper']['alpha']['value']
        )
    )
    #
    tb_field_align_min = TextBox(
        plt.axes([ll, bb - 0.09, w_tb, h_tb]),
        label = "min", label_pad = 0.05,
        initial = "{0:.2f}".format(
            rec_params['taper']['field_align']['min']
        )
    )
    tb_field_align_max = TextBox(
        plt.axes([ll + ww - w_tb, bb - 0.09, w_tb, h_tb]),
        label = "max", label_pad = 0.05,
        initial = "{0:.2f}".format(
            rec_params['taper']['field_align']['max']
        )
    )
    #
    cb_radius = CheckButtons(
        plt.axes([ll + ww, bb + hh + 11.5 * h_tb, w_tb, h_tb]),
        labels = ["fixed"],
        actives = [rec_params['taper']['r_0']['fixed']]
    )
    cb_alpha = CheckButtons(
        plt.axes([ll + ww, bb + hh + 10.5 * h_tb, w_tb, h_tb]),
        labels = ["fixed"],
        actives = [rec_params['taper']['alpha']['fixed']]
    )
    cb_beta = CheckButtons(
        plt.axes([ll + ww, bb + hh + 9.5 * h_tb, w_tb, h_tb]),
        labels = ["fixed"],
        actives = [rec_params['taper']['beta']['fixed']]
    )
    #
    tb_field_target = TextBox(
        plt.axes([ll + (ww - w_tb) / 2.0, bb - 0.09, w_tb, h_tb]),
        label = "$E_0$", label_pad = 0.05,
        initial = "{0:.2f}".format(
            rec_params['taper']['field_align']['E_0']
        )
    )
#
# create text boxes and check buttons for classic module
elif rec_params['module'] == 'classic':
    tb_field_target = TextBox(
        plt.axes([ll + (ww - w_tb) / 2.0, bb - 0.09, w_tb, h_tb]),
        label = "$E_0$", label_pad = 0.05,
        initial = "{0:.2f}".format(rec_params['classic']['E_0'])
    )
#
#
# create check button for volume export
cb_export_volumes = CheckButtons(
    plt.axes([ll + ww - w_tb, bb + hh + 8 * h_tb, 2 * w_tb, h_tb]),
    labels = ["Export volumes"],
    actives = [rec_params['export_volumes']]
)
# create check buttons for molecule splitting
cb_molecule_split = CheckButtons(
    plt.axes([ll + ww - w_tb, bb + hh + 6 * h_tb, 2 * w_tb, 2 * h_tb]),
    labels = ["Split molecules", "Shuffle"],
    actives = rec_params['split_molecules']
)
# create check button for charge splitting
cb_charge_split = CheckButtons(
    plt.axes([ll + ww - w_tb, bb + hh + 5 * h_tb, 2 * w_tb, h_tb]),
    labels = ["Split charge"],
    actives = [rec_params['split_charge']]
)
# create check button for background atoms
cb_background = CheckButtons(
    plt.axes([ll + ww - w_tb, bb + hh + 4 * h_tb, 2 * w_tb, h_tb]),
    labels = ["Background"],
    actives = [rec_params['background']]
)
# create reconstruction button
bt_rec = Button(
    plt.axes([ll + ww - w_tb, bb + hh + 3 * h_tb, 2 * w_tb, h_tb]),
    "Reconstruct"
)
# create export button
bt_exp = Button(
    plt.axes([ll + ww - w_tb, bb + hh + 2 * h_tb, 2 * w_tb, h_tb]),
    "Export"
)
#
#
#
#
# register callback functions
#
# detection efficiency
cid_tb_efficiency = tb_efficiency.on_submit(get_geometry)
#
# image compression factor
cid_tb_image_compression = tb_image_compression.on_submit(get_geometry)
#
# field factor
cid_tb_beta = tb_beta.on_submit(get_geometry)
#
#
# taper geometry
if rec_params['module'] == 'taper':
    cid_tb_radius = tb_radius.on_submit(get_geometry)
    cid_tb_alpha  = tb_alpha.on_submit(get_geometry)
    #
    cid_cb_radius = cb_radius.on_clicked(get_geometry)
    cid_cb_alpha  = cb_alpha.on_clicked(get_geometry)
    cid_cb_beta   = cb_beta.on_clicked(get_geometry)
    #
    # field alignment
    cid_tb_field_align_min = tb_field_align_min.on_submit(get_geometry)
    cid_tb_field_align_max = tb_field_align_max.on_submit(get_geometry)
#
#
# charge splitting
cid_cb_charge_split = cb_charge_split.on_clicked(map_ids)
#
# reconstruction
cid_bt_rec = bt_rec.on_clicked(reconstruct)
#
# export
cid_bt_exp = bt_exp.on_clicked(export)
#
#
# field alignment
cid_tb_field_target = tb_field_target.on_submit(get_geometry)
#
#
#
#
# map chemical IDs and atomic volumes
map_ids(None)
#
#
# trigger initial calculation of tip geometry
get_geometry(None)
#
#
#
#
# show plot
plt.show()
#
#
#
#
################################################################################
###                                                                          ###
###     DB UPLOAD / EXPORT                                                   ###
###                                                                          ###
################################################################################
# upload reconstruction parameters to database
db_upload()
#
#
#
#
# export tip data
export(None)
