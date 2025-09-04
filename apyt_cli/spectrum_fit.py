#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#
desc = """
Fit a calibrated mass spectrum for peak deconvolution and background correction.

This script applies a fitting routine to a calibrated, high-quality mass
spectrum that has typically been pre-processed with the ``apyt_spectrum_align``
script. The fitting makes use of isotopic reference data to deconvolve
overlapping peaks, estimate peak intensities, model the background signal, and
also handle **molecular peaks** in addition to atomic ones.

The script serves as a convenient command line wrapper around the APyT Python
mass spectrum fitting module, enabling users to run the fitting procedure
without writing custom code. For more technical details, please refer to the
online documentation at: https://apyt.mp.imw.uni-stuttgart.de
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
import argparse
import ast
import logging
import matplotlib.pyplot as plt
import numpy as np
#
# import individual functions/modules
from apyt.gui.forms import login
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import RadioButtons, Slider, TextBox
from tkinter import messagebox
from ttkthemes import ThemedTk
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
    "species_properties", type = str,
    help = """\
Mapping of elemental or molecular species to their
charge states and associated reconstruction volumes (in
nm³). The argument must be given as a string
representation of a Python dictionary, for example:
"{'Cu': ((1, 2), 0.012), 'Ni': ((1, 2), 0.012)}".
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
    "--debug", action = 'store_true',
    help = """\
Whether to print debug messages.
"""
)
parser.add_argument(
    "--export-isotopes", action = 'store_true',
    help = """\
Whether to export the individual spectra of *all*
isotopes. Defaults to 'False', i.e. export the combined
spectrum for all isotopes of a specific element.
"""
)
parser.add_argument(
    "-h", "--help", action = "help",
    default = argparse.SUPPRESS,
    help = """\
Show this help message and exit.
"""
)
parser.add_argument(
    "--ignore", metavar = "<molecule>", nargs = '+', type = str,
    default = [],
    help = """\
The list of elements/molecules to ignore when
compositions are calculated. Use '--' to indicate
subsequent positional arguments. Defaults to empty
list.\
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
#
#
def add_peak_labels(abundance_thres):
    """
    Add peak labels to plot.
    """
    #
    #
    def add_label(mc_ratio, label):
        """
        Simple helper function to add label to peak.
        """
        #
        #
        # do nothing if label empty
        if label == "":
            return
        #
        # add label to peak
        ax.text(
            mc_ratio,
            # use maximum of fit and raw spectrum data for y-position
            max(
                fit_result.eval(x = mc_ratio),
                hist_data[np.abs(hist_data[:, 0] - mc_ratio).argmin(), 1]
            ),
            "  " + label,
            horizontalalignment = 'center', verticalalignment = 'bottom',
            rotation = 'vertical'
        )
    #
    #
    #
    #
    # sort peaks by mass-to-charge ratio
    peaks_list_sorted = sorted(
        peaks_list, key = lambda peak: peak['mass_charge_ratio']
    )
    #
    #
    # loop through peaks
    label         = ""
    mc_ratio_last = np.nan
    for peak in peaks_list_sorted:
        # check whether current peak position differs from previous peak
        # position
        if np.abs(
            peak['mass_charge_ratio'] - mc_ratio_last
        ) > np.finfo(np.float32).eps:
            # add label for previous peak
            add_label(mc_ratio_last, label)
            #
            # reset label
            label = ""
        #
        #
        # append label for current peak
        if peak['is_max'] == True or peak['abundance'] >= abundance_thres:
            if label != "":
                label += "/"
            label += "$\\mathregular{{{0:s}^{{{1:d}\\!\\!+}}}}$". \
                     format(peak['element'], peak['charge'])
        #
        #
        # update current peak position
        mc_ratio_last = peak['mass_charge_ratio']
    #
    #
    # add label for last peak
    add_label(mc_ratio_last, label)
#
#
#
#
def callback_refit(_):
    """
    Simple callback function to trigger re-fit of spectrum.
    """
    #
    # fit mass spectrum
    fit_spectrum()
    #
    # plot mass spectrum
    plot_spectrum(ax, sl_abundance_thres.val, rb_log_scale.value_selected)
#
#
#
#
def callback_replot(_):
    """
    Simple callback function to re-plot spectrum.
    """
    #
    # plot mass spectrum
    plot_spectrum(ax, sl_abundance_thres.val, rb_log_scale.value_selected)
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
    # construct results dictionary for upload to database
    errors_dict = {}
    for name, param in fit_result.params.items():
        errors_dict[name] = param.stderr
    result_dict = {
        'info': {
            'version':  mf.version(),
            'function': rb_function.value_selected
            },
        'values': fit_result.params.valuesdict(),
        'errors': errors_dict,
        'stats':  {
            'chi-square': float(fit_result.chisqr),
            'redchi':     float(fit_result.redchi),
            'R-squared':  float(fit_result.rsquared)
            },
        'peaks': peaks_dict
    }
    #
    # add fit results to parameters
    record['parameters']['spectrum_fit'] = result_dict
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
        "Do you want to upload your spectrum fit to the database?",
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
def fit_spectrum():
    """
    Fit mass spectrum with given peak shape function.
    """
    #
    #
    # we need to make the fitting results available globally
    global fit_result, x_model, y_model
    #
    #
    # perform fit
    fit_result = mf.fit(
        hist_data, peaks_list, rb_function.value_selected, verbose = True,
        peak_shift  = tb_peak_shift.text.split(),
        scale_width = (rb_width_scaling.value_selected == 'width scaling on')
    )
    #
    #
    # calculate model prediction (with denser grid)
    x_model = np.linspace(
        hist_data[0, 0], hist_data[-1, 0],
        np.rint((hist_data[-1, 0] - hist_data[0, 0]) / 1.0e-3 + 1).astype(int)
    )
    y_model = fit_result.eval(x = x_model)
    #
    #
    # get counts for each element
    bin_width = hist_data[1, 0] - hist_data[0, 0]
    counts, total_counts, background = mf.counts(
        peaks_list, rb_function.value_selected,
        fit_result.best_values, tuple(hist_data[[0, -1], 0]),
        bin_width, ignore_list = args.ignore, verbose = True
    )
    logger.info(
        "Total number of counts (with background) from raw spectrum data is:" +
        f"\t{np.sum(hist_data[:, 1]):.0f}."
    )
    logger.info(
        "Total number of counts (with background) from fit is:" +
        f"\t\t{total_counts + background:.0f}."
    )
#
#
#
#
def plot_spectrum(axis, abundance_thres, scale):
    """
    Plot mass spectrum including fit.
    """
    #
    #
    # clear complete plot
    axis.cla()
    #
    #
    # plot mass spectrum including fit
    axis.set_xlabel("mass-to-charge ratio (amu/e)")
    axis.set_ylabel("Counts")
    axis.plot(hist_data[:, 0], hist_data[:, 1], label = "data", linewidth = 3)
    axis.plot(x_model, y_model, label = "fit")
    axis.legend()
    #
    #
    # set y-axis scale
    axis.set_yscale(scale)
    #
    # add some additional margin for peak index labels
    axis.autoscale()
    if scale == "linear":
        axis.set_ylim(top = axis.get_ylim()[1] * 1.08)
    elif scale == "log":
        axis.set_ylim(top = axis.get_ylim()[1] * 3)
    #
    #
    # add peak labels
    add_peak_labels(abundance_thres)
    #
    #
    # add R^2 value from fit
    ax.text(
        1.02, 0.02, "$R^2 = {0:.4f}$".format(fit_result.rsquared),
        transform = ax.transAxes
    )
    #
    #
    # update plot
    fig.canvas.draw_idle()
#
#
#
#
# parse and print arguments
args = parser.parse_args()
print("\nSettings used:")
for key, value in sorted(vars(args).items()):
    print("{0:20s}{1}".format(key, value))
print("")
#
#
#
#
# initialize authorization credentials
auth = None
#
#
# pass debug flag to massfit module
mf.enable_debug(args.debug)
#
#
#
#
################################################################################
###                                                                          ###
###     DATA PROCESSING / FILTERING                                          ###
###                                                                          ###
################################################################################
# get custom ID and spectrum parameters and authorization details
record = db_query(args.id, ("custom_id", "parameters"))
#
custom_id       = record['custom_id']
data_filter     = record['parameters']['data_filter']
spectrum_params = record['parameters']['spectrum_params']
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
################################################################################
###                                                                          ###
###     PLOT SETUP                                                           ###
###                                                                          ###
################################################################################
# create figure
fig = plt.figure(figsize = (6.4 * 2, 4.8 * 2))
#
#
# set relative widths and heights of subplots
h_fig     = 30 # relative height of figure
h_spacing =  3 # relative height of spacing between figures
h_sl      =  1 # relative height of slider
#
w_fig = 4 # relative width of figure
w_rb  = 1 # relative width of radio buttons
#
#
# create GridSpec for subplots
gs = GridSpec(
    7, 2, hspace = 0.0, wspace = 0.0,
    left = 0.10, top = 0.95, bottom = 0.02, right = 0.98,
    height_ratios = [
        0.07 * h_fig, 0.07 * h_fig, 0.07 * h_fig, 0.03 * h_fig, 0.76 * h_fig,
        h_spacing, h_sl
    ],
    width_ratios  = [w_fig, w_rb]
)
#
# initialize current row and column for GridSpec
row = 0
col = 0
#
#
# create plot axis for mass spectrum
ax = plt.subplot(gs[row:row+5, col])
#
#
# create radio buttons
rb_log_scale = RadioButtons(
    ax = plt.subplot(gs[row+0, col+1]), labels = ('linear', 'log'),
    active = 0
)
rb_width_scaling = RadioButtons(
    ax = plt.subplot(gs[row+1, col+1]),
    labels = ('width scaling off', 'width scaling on'),
    active = 0
)
rb_function = RadioButtons(
    ax = plt.subplot(gs[row+2, col+1]),
    labels = ('error-expDecay', 'error-doubleExpDecay'),
    active = 0
)
# text box for peak shift
tb_peak_shift = TextBox(
    ax = plt.subplot(gs[row+3, col+1]),
    label = "RegEx's for peak shifts (space-\nseparated)\n"
            "Example: O[0-9]*_[0-9]+"
)
label = tb_peak_shift.ax.get_children()[0]
label.set_position([0.05, -0.2])
label.set_horizontalalignment('left')
label.set_verticalalignment('top')
row += 6
#
#
# create sliders
sl_abundance_thres = Slider(
    ax = plt.subplot(gs[row, col]), label = "Abundance\nthres.",
    valmin  = 0.00, valmax  = 0.50,
    valinit = 0.50, valstep = 0.01, valfmt = "%0.2f", dragging = False
)
#
#
# register callback functions
cid_rb_log_scale     = rb_log_scale.on_clicked(callback_replot)
cid_rb_width_scaling = rb_width_scaling.on_clicked(callback_refit)
cid_rb_function      = rb_function.on_clicked(callback_refit)
cid_tb_peak_shift    = tb_peak_shift.on_submit(callback_refit)
sl_abundance_thres.on_changed(callback_replot)
#
#
#
#
################################################################################
###                                                                          ###
###     DATA ANALYSIS                                                        ###
###                                                                          ###
################################################################################
# calculate mass spectrum
logger.info("Calculating mass spectrum…")
hist, bin_centers, _ = ms.get_mass_spectrum(
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
hist_data = np.column_stack((bin_centers, hist))
#
#
# get list and nested dictionary of all peaks
peaks_list, peaks_dict = mf.peaks_list(
    ast.literal_eval(args.species_properties),
    # the number of decimals used for grouping the mass-to-charge ratios is
    # based on the resolution of the mass spectrum, i.e. the bin width
    mass_decimals = int(np.ceil(-np.log10(spectrum_params['bin_width']))),
    verbose = True
)
#
# test whether all expected peaks are within input data range
for peak in peaks_list:
    if peak['mass_charge_ratio'] < hist_data[0, 0] or \
       peak['mass_charge_ratio'] > hist_data[-1, 0]:
        raise Exception(
            "Mass-to-charge ratio ({0:.2f}) for element \"{1:s}\" is out of "
            "input data range ({2:.1f}, {3:.1f}).".
            format(peak['mass_charge_ratio'], peak['element'],
                   hist_data[0, 0], hist_data[-1, 0])
        )
#
#
# fit mass spectrum
fit_spectrum()
#
#
# show plot
plot_spectrum(
    axis            = ax,
    abundance_thres = sl_abundance_thres.val,
    scale           = rb_log_scale.value_selected
)
plt.suptitle(
    "{0:s} ({1:.1f} M events)".format(custom_id, np.sum(hist_data[:, 1]) / 1e6)
)
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
# upload fit results to database
db_upload()
#
#
#
#
# export *all* isotope spectra if requested
if args.export_isotopes == True:
    # get list of peaks (isotopes) with abundance above certain threshold
    peaks_list_thres = [p for p in peaks_list if p['abundance'] >= 1e-3]
    #
    # set export format specifier
    fmt = "\t%32.3f" * len(peaks_list_thres)
    #
    #
    # loop through peaks (isotopes) to get individual counts
    counts = np.zeros((len(x_model), len(peaks_list_thres)))
    header = ""
    for i, peak in zip(range(len(peaks_list_thres)), peaks_list_thres):
        # get isotope spectrum for specified peak only
        counts[:, i] = mf.spectrum(
            x_model, [peak], rb_function.value_selected, fit_result.best_values
        )
        #
        # append table header
        # "<element name> (<column id>; <mass-to-charge ratio>; <abundance>)"
        header += "\t{0:>10s} ({1:3d}; {2:7.3f}; {3:.3f})".format(
            peak['element'], i + 4, peak['mass_charge_ratio'], peak['abundance']
        )
# otherwise export combined spectrum of all isotopes (default)
else:
    # get list of elements and charge states
    elements = ast.literal_eval(args.species_properties).keys()
    #
    # set export format specifier
    fmt = "\t%16.3f" * len(elements)
    #
    #
    # loop through all elements and charge states
    counts = np.zeros((len(x_model), len(elements)))
    header = ""
    for i, element in zip(range(len(elements)), elements):
        # get cumulated spectrum for *all* isotopes of the associated element
        counts[:, i] = mf.spectrum(
            x_model, peaks_list, rb_function.value_selected,
            fit_result.best_values, elements_list = [element]
        )
        #
        # append table header
        # "<element name> (<column id>)"
        header += "\t{0:>10s} ({1:3d})".format(element, i + 4)
#
#
#
#
# export fit results
np.savetxt(
    custom_id + "_mass_spectrum_fit.txt",
    np.column_stack((
        x_model, y_model,
        fit_result.values['base'] / np.sqrt(x_model), counts
    )),
    header = "m/q (amu/e)\t     counts\t   baseline" + header,
    fmt = "%+08.3f\t%11.3f\t%11.3f" + fmt
)
# also export raw spectrum (apyt_massspec.py may give slightly different
# spectrum due to loss of precision for spectrum parameters)
np.savetxt(
    custom_id + "_mass_spectrum_raw.txt", hist_data,
    fmt = '%+08.3f\t %7d', header = "m/q (amu/e)\t  counts"
)
