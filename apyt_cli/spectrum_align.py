#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#
desc = """Calculate a high-quality mass spectrum from raw measurement data.

This script processes a raw measurement file to compute the corresponding mass
spectrum. Voltage and flight length corrections are first applied to express the
correction functions as polynomials φ(U) and ψ(x, y). In the next step, the
polynomial coefficients can be optimized to produce peaks in the mass spectrum
with maximum sharpness. The resulting optimization parameters, which contain all
information required to reconstruct a high-quality mass spectrum, can then be
uploaded to the SQL database. In addition, the final mass spectrum is exported
as a plain ASCII text file.

This script acts as a wrapper for the Python mass spectrum module of the APyT
package. A detailed description of the package is available at:
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
import argparse
import logging
import matplotlib.pyplot as plt
import numpy as np
import pyperclip
import warnings
#
# import individual functions/modules
from apyt.gui.forms import login
from apyt.io.config import get_setting
from matplotlib.collections import LineCollection
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import Button, RadioButtons, Slider, TextBox
from numpy.polynomial.polynomial import polyval
from scipy.signal import find_peaks, peak_widths
from scipy.stats import binned_statistic
from threading import RLock
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
# reentrant lock for 'UPDATING' overlay
r_mutex = RLock()
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
The record ID in the APT SQL measurement database.
"""
)
parser.add_argument(
    "--binning", metavar = ("<λ_w>"), type = int, default = None,
    help = """\
The scaling factor for the adjustable bin width range,
given as a power of 10. Use this if the default binning
range does not fit the measurement data (e.g., due to
noise). Defaults to None (use internal defaults).
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
    "--check-voltage", action = 'store_true',
    help = """\
Whether to check the voltage curve. Use this to check
the stability of the measurement.
"""
)
parser.add_argument(
    "--detector-voltage", metavar = ("<U_det>"),
    type = np.float32, default = np.float32(0.0),
    help = """\
The detector voltage (in V). Defaults to zero.
"""
)
parser.add_argument(
    "--debug", action = 'store_true',
    help = """\
Whether to print debug messages.
"""
)
parser.add_argument(
    "-f", "--filter-detector", action = 'store_true',
    help = """\
Whether to filter detector hits to include only valid
positions. The valid detector size is determined from
the provided device. Use this if the raw input data may
contain invalid or nonsensical hit position values.
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
    "-I", "--interval", metavar = ("<min>", "<max>"), nargs = 2, type = float,
    default = None,
    help = """\
The interval of the measurement data to use, expressed
as a fraction of the total measured events, between 0.0
and 1.0. Defaults to 'None', i.e. use all events.
"""
)
parser.add_argument(
    "-M", "--mass-range", metavar = ("<min>", "<max>"), nargs = 2, type = float,
    default = None,
    help = """\
The mass range to use for voltage and flight length
correction. Defaults to a narrow range around the
maximum peak.
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
parser.add_argument(
    "--pulse-coupling", metavar = ("<φ_U>"),
    type = np.float32, default = np.float32(1.0),
    help = """\
The pulse coupling factor. Defaults to 1.0.
"""
)
parser.add_argument(
    "-V", "--voltage-range", metavar = ("<U_min>", "<U_max>"), nargs = 2,
    type = float, default = None,
    help = """\
The voltage range (in kV) to use for filtering the
measurement data. Defaults to 'None', i.e. all events
will be used.
"""
)
#
#
#
#
def check_fine_corr(val):
    """
    Perform fine correction of voltage and flight length.
    """
    #
    #
    # remove previous line for fine correction in correction range and voltage
    # plot if existing
    if val == "fine off":
        was_fine_active = False
        for line in ax_correction_range.lines:
            if plt.getp(line, 'label') == "fine corr.":
                line.remove()
                ax_correction_range.legend()
                was_fine_active = True
        for line in ax_voltage.lines:
            if plt.getp(line, 'label') == "fine":
                line.remove()
                ax_voltage.legend()
                was_fine_active = True
        # restart correction cycle if fine correction was active before
        if was_fine_active == True:
            update_voltage(None)
            return
    #
    #
    # do nothing if only flight length correction requested
    if rb_flight_length.value_selected == "flight on" and val == "fine off":
        return
    #
    #
    # check whether required flight length correction is active
    elif rb_flight_length.value_selected == "flight on" and val == "fine on":
        # do not trigger fine correction again if already present
        for line in ax_correction_range.lines:
            if plt.getp(line, 'label') == "fine corr.":
                return
        #
        #
        # print update notification
        print_update_notification(True)
        #
        #
        # set data range and histogram bin width
        data_range = (float(tb_correction_range_min.text),
                      float(tb_correction_range_max.text))
        hist_width = sl_hist_bin_width.val
        #
        #
        # optimize voltage correction
        coeffs[0] = ms.optimize_correction(
            data, (tof, flight_length, (coeffs[0], coeffs[1]), alpha),
            'voltage', hist = {'width': hist_width, 'range': data_range})
        #
        #
        #
        #
        # add result for fine correction to voltage plot
        #
        # get x-data and peak target position from plot
        for line in ax_voltage.lines:
            if plt.getp(line, 'label') == 'fit':
                x_data = line.get_xdata()
            if plt.getp(line, 'label') == 'peak target':
                peak_target = line.get_ydata()[0]
        #
        #
        # add plot for fine correction
        ax_voltage.plot(
            x_data,
            peak_target / (1.0 +
                polyval(1000 * x_data, coeffs[0]) / (1000 * x_data)),
            color = 'C2', label = "fine")
        #
        #
        # update legend
        ax_voltage.legend()
        #
        #
        #
        #
        # optimize flight length correction
        coeffs[1] = ms.optimize_correction(
            data, (tof, flight_length, (coeffs[0], coeffs[1]), alpha),
            'flight', hist = {'width': hist_width, 'range': data_range})
        #
        #
        # calculate optimized charge-to-mass ratio histogram
        hist, bin_centers, _ = ms.get_mass_spectrum(
            data, (tof, flight_length, (coeffs[0], coeffs[1]), alpha),
            hist = {'width': hist_width, 'range': data_range})
        ax_correction_range.plot(
            bin_centers, hist, color = 'C3', label = "fine corr.")
        #
        #
        # update legend
        ax_correction_range.legend()
        #
        #
        # remove update notification and update figures
        print_update_notification(False)
        fig.canvas.draw_idle()
    #
    #
    # do nothing if fine correction requested without prior flight length
    # correction
    elif rb_flight_length.value_selected == "flight off" and val == "fine on":
        # temporarily disable callback function to avoid recursive calls when
        # resetting radio button
        global cid_rb_fine_corr
        rb_fine_corr.disconnect(cid_rb_fine_corr)
        rb_fine_corr.set_active(0)
        cid_rb_fine_corr = rb_fine_corr.on_clicked(check_fine_corr)
    #
    #
    # conditionally plot full spectrum
    check_full_spec(rb_full_spec.value_selected)
#
#
#
#
def check_flight_length(val):
    """
    Check whether flight length correction shall be performed.
    """
    #
    #
    # do not trigger flight length correction again if already present
    if val == "flight on":
        for line in ax_correction_range.lines:
            if plt.getp(line, 'label') == "voltage & flight corr.":
                return
    #
    #
    # print update notification
    print_update_notification(True)
    #
    #
    # update plot for flight length correction
    plot_flight_length()
    #
    #
    # remove previous line for voltage and flight length correction in
    # correction range plot if existing
    for line in ax_correction_range.lines:
        if plt.getp(line, 'label') == "voltage & flight corr.":
            line.remove()
    #
    #
    # plot spectrum with voltage and flight length correction if requested
    if val == "flight on":
        hist, bin_centers, _ = ms.get_mass_spectrum(
            data, (tof, flight_length, (coeffs[0], coeffs[1]), alpha),
            hist = {'width': sl_hist_bin_width.val,
                    'range': (float(tb_correction_range_min.text),
                              float(tb_correction_range_max.text))})
        ax_correction_range.plot(bin_centers, hist, color = 'C2',
                                 label = "voltage & flight corr.")
    #
    #
    # update plot range
    ax_correction_range.relim()
    ax_correction_range.autoscale_view()
    #
    # update legend
    ax_correction_range.legend()
    #
    #
    # if fine correction was active, deactivate and call respective callback
    # function
    if val == "flight off" and rb_fine_corr.value_selected == "fine on":
        rb_fine_corr.set_active(0)
    #
    #
    # reset flight length correction coefficients
    if val == "flight off":
        coeffs[1] = None
    #
    #
    # conditionally plot full spectrum
    check_full_spec(rb_full_spec.value_selected)
    #
    #
    # remove update notification and update figures
    print_update_notification(False)
    fig.canvas.draw_idle()
#
#
#
#
def check_full_spec(val):
    """
    Check whether full mass spectrum shall be shown.
    """
    #
    #
    # print update notification
    print_update_notification(True)
    #
    # update plot for full spectrum
    plot_full_spectrum()
    #
    # remove update notification and update figures
    print_update_notification(False)
    fig.canvas.draw_idle()
#
#
#
#
def check_log_scale(label):
    """
    Apply logarithmic scale to respective plots.
    """
    #
    #
    # set log scale for respective plots
    ax_correction_range.set_yscale(label)
    ax_full_spec.set_yscale(label)
    #
    #
    # add some additional margin for peak index labels
    ax_full_spec.autoscale()
    if label == "linear":
        ax_full_spec.set_ylim(top = ax_full_spec.get_ylim()[1] * 1.08)
    elif label == "log":
        ax_full_spec.set_ylim(top = ax_full_spec.get_ylim()[1] * 3)
    #
    #
    # update figures
    fig.canvas.draw_idle()
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
    Query one or more fields from a SQL database record.

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
def db_upload(_):
    """
    Upload spectrum parameters to the database.

    This is a simple wrapper around the `apyt.io.localdb.update` and
    `apyt.io.sql.update` functions, respectively.
    """
    #
    #
    # set data filters
    data_filter = {}
    data_filter['detector_radius'] = float(detector_radius)
    if interval is not None:
        data_filter['interval'] = interval
    data_filter['mass_charge_range'] = [
        float(tb_full_spec_min.text),
        float(tb_full_spec_max.text)
    ]
    if args.voltage_range is not None:
        data_filter['voltage_range'] = args.voltage_range
    #
    #
    # set mass spectrum parameters (convert np.float32 to built-in data types)
    spectrum_params = {
        'alpha':            float(alpha),
        'L_0':              float(flight_length),
        't_0':              float(tof),
        'voltage_coeffs':   coeffs[0].tolist(),
        'flight_coeffs':    coeffs[1].tolist(),
        'pulse_coupling':   float(args.pulse_coupling),
        'detector_voltage': float(args.detector_voltage),
        'bin_width':        sl_hist_bin_width.val
    }
    #
    #
    # set parameters dictionary
    parameters = {
        'data_filter':     data_filter,
        'spectrum_params': spectrum_params
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
        "Do you want to upload your spectrum parameters to the database?",
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
        localdb.update(args.id, "parameters", parameters)
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
        status, response = sql.update(args.id, "parameters", parameters, auth)
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
def on_key_press(event):
    """
    Handle keyboard input events.

    'ctrl+v' shall paste content from the clipboard to the currently active
    TextBox widget.
    """
    #
    #
    # do nothing if keyboard input should not be handled
    if event.key != 'ctrl+v':
        return
    #
    #
    # loop through all global symbols to find TextBox widgets (This seems a bit
    # hacky, but there may be no more direct method in matplotlib to access the
    # currently active widget?)
    for symbol in globals().values():
        # check whether symbol is a TextBox widget and matches the current axes
        if isinstance(symbol, TextBox) and symbol.ax == event.inaxes:
            # set text and return
            symbol.set_val(pyperclip.paste())
            return
#
#
#
#
def plot_correction_range():
    """
    Plot mass spectrum with limited correction range.
    """
    #
    #
    # clear complete plot
    ax_correction_range.cla()
    #
    #
    # set data range and histogram bin width
    data_range = (float(tb_correction_range_min.text),
                  float(tb_correction_range_max.text))
    hist_width = sl_hist_bin_width.val
    #
    #
    # plot initial spectrum
    hist, bin_centers, _ = ms.get_mass_spectrum(
        data, (tof, flight_length, (None, None), alpha),
        hist = {'width': hist_width, 'range': data_range})
    ax_correction_range.plot(bin_centers, hist, label = "initial")
    #
    #
    # plot spectrum with voltage correction
    hist, bin_centers, _ = ms.get_mass_spectrum(
        data, (tof, flight_length, (coeffs[0], None), alpha),
        hist = {'width': hist_width, 'range': data_range})
    ax_correction_range.plot(bin_centers, hist, label = "voltage corr.")
    #
    #
    # set labels and legend
    ax_correction_range.set_xlabel("Mass-to-charge ratio (amu/e)")
    ax_correction_range.set_ylabel("Counts")
    ax_correction_range.legend()
    #
    # set log scale if requested
    ax_correction_range.set_yscale(rb_log_scale.value_selected)
#
#
#
#
def plot_flight_length():
    """
    Perform and plot flight length correction.
    """
    #
    #
    # remove color bar if existing
    if hasattr(plot_flight_length, 'cbar'):
        plot_flight_length.cbar.remove()
        del plot_flight_length.cbar
    #
    # clear complete plot
    ax_flight_length.cla()
    #
    #
    # return directly if requested
    if rb_flight_length.value_selected == "flight off":
        return
    #
    #
    # perform flight length correction
    coeffs[1], xyz_data, events, wireframe = ms.get_flight_correction(
        data, (tof, flight_length, (coeffs[0], None), alpha),
        deg = 2,
        hist = {'range': (float(tb_correction_range_min.text),
                          float(tb_correction_range_max.text)),
                'width': sl_flight_length_bin_width.val},
        steps = int(sl_flight_length_steps.val),
        thres = sl_flight_length_peak_thres.val)
    #
    #
    # create plot
    xyz_plot = ax_flight_length.scatter(*xyz_data, c = 1.0e-3 * events)
    plot_flight_length.cbar = fig.colorbar(
        xyz_plot, ax = ax_flight_length, fraction = 0.04, pad = 0.2)
    ax_flight_length.plot_wireframe(*wireframe, lw = 1, color = 'C1')
    #
    #
    # set labels
    ax_flight_length.set_xlabel("$x$ (mm)")
    ax_flight_length.set_ylabel("$y$ (mm)")
    ax_flight_length.set_zlabel("Peak position (amu/e)")
    plot_flight_length.cbar.set_label("Counts ($10^3$)")
#
#
#
#
def plot_full_spectrum():
    """
    Plot full mass spectrum with all corrections.
    """
    #
    #
    # clear complete plot
    ax_full_spec.cla()
    #
    #
    # return directly if requested
    if rb_full_spec.value_selected == "full off":
        return
    #
    #
    # calculate complete mass spectrum
    hist, bin_centers, mc_ratio = ms.get_mass_spectrum(
        data, (tof, flight_length, coeffs, alpha),
        hist = {'width': sl_hist_bin_width.val,
                'range': (float(tb_full_spec_min.text),
                          float(tb_full_spec_max.text))})
    #
    #
    # calculate average measurement interval for each mass spectrum bin
    avg_interval, *_ = binned_statistic(
        mc_ratio, np.arange(len(mc_ratio)) / (len(mc_ratio) - 1),
        bins = len(hist),
        range = (
            float(tb_full_spec_min.text) - sl_hist_bin_width.val / 2,
            float(tb_full_spec_max.text) + sl_hist_bin_width.val / 2
        )
    )
    # NaN is assigned to empty bins; fill with average interval 0.5
    avg_interval = np.where(np.isnan(avg_interval), 0.5, avg_interval)
    #
    #
    # create a set of line segments so that we can color them individually
    # (see https://matplotlib.org/stable/gallery/lines_bars_and_markers/multicolored_line.html)
    #
    # create points as an N x 1 x 2 array so that we can stack points together
    # easily to get the segments
    points = np.column_stack((bin_centers, hist)).reshape(-1, 1, 2)
    # segments array for line collection needs to be
    # (numlines) x (points per line) x 2 (for x and y).
    line_segments = np.hstack((points[:-1], points[1:]))
    #
    # construct sequence of connected line segments and set color-coding for
    # each segment
    line_collection = LineCollection(
        line_segments, norm = plt.Normalize(0.0, 1.0)
    )
    line_collection.set_array(avg_interval)
    #
    #
    # plot mass spectrum with color-coded line segments
    ax_full_spec.add_collection(line_collection)
    #
    #
    # determine maximum peak
    peaks, _ = find_peaks(hist, distance = np.iinfo(np.int32).max)
    peak_max = hist[peaks[0]]
    #
    #
    # determine all peaks with height above threshold
    peaks, _ = find_peaks(hist, prominence = sl_full_peak_thres.val * peak_max)
    #
    #
    # print device resolution for selected peaks if requested
    if args.debug == True:
        # get widths of selected peaks
        widths_half = peak_widths(hist, peaks, rel_height = 0.5)[0]
        #
        # calculate resolution for each individual peak
        res = list(zip(
            bin_centers[peaks],
            bin_centers[peaks] / (widths_half * sl_hist_bin_width.val)))
        #
        # print resolutions
        out_str = ""
        for r in res:
            out_str += "\n{0:7.3f}\t\t{1:11.3f}".format(r[0], r[1])
        ms._debug("Device resolution for selected peak positions:\n"
                  "# pos (amu/e)\t resolution{0:s}".format(out_str))
    #
    #
    # add all peak positions and heights to plot; also add list of peaks to the
    # right of the plot (we need to store all peak positions to perform
    # automatic alignment later)
    index = 0
    plot_full_spectrum.peak_pos = np.array([])
    string = "Peak list:"
    for i in peaks:
        ax_full_spec.text(bin_centers[i], hist[i], "{0:d}\n".format(index + 1),
                          horizontalalignment = 'center', clip_on = True)
        string += "\n{0:02d} {1:06.3f}".format(index + 1, bin_centers[i])
        plot_full_spectrum.peak_pos = \
            np.append(plot_full_spectrum.peak_pos, bin_centers[i])
        index += 1
        #
        # stop if too many peaks detected (probably nonsensical values provided)
        if index >= 20:
            warnings.warn("More than 20 peaks detected. Omitting subsequent "
                          "peaks.")
            break
    #
    ax_full_spec.text(1.02, 0.98, string, transform = ax_full_spec.transAxes,
                      verticalalignment = 'top')
    #
    #
    # set labels
    ax_full_spec.set_xlabel(
        "Mass-to-charge ratio (amu/e)\n"
        "(Color corresponds to avg. measurement interval)"
    )
    ax_full_spec.set_ylabel("Counts")
    #
    # set logarithmic scale if requested
    ax_full_spec.set_yscale(rb_log_scale.value_selected)
    #
    # add some additional margin for peak index labels
    ax_full_spec.autoscale()
    if rb_log_scale.value_selected == "linear":
        ax_full_spec.set_ylim(top = ax_full_spec.get_ylim()[1] * 1.08)
    elif rb_log_scale.value_selected == "log":
        ax_full_spec.set_ylim(top = ax_full_spec.get_ylim()[1] * 3)
#
#
#
#
def plot_init():
    """
    Update all plots in the figure.
    """
    #
    #
    # print update notification
    print_update_notification(True)
    #
    #
    # perform voltage and flight length correction
    plot_voltage()
    #
    # plot mass spectrum for correction range
    plot_correction_range()
    #
    # conditionally perform flight length correction
    check_flight_length(rb_flight_length.value_selected)
    #
    # conditionally draw full spectrum
    check_full_spec(rb_full_spec.value_selected)
    #
    #
    # remove update notification and update figures
    print_update_notification(False)
    fig.canvas.draw_idle()
#
#
#
#
def plot_voltage():
    """
    Perform and plot voltage correction.
    """
    #
    #
    # remove color bar if existing
    if hasattr(plot_voltage, 'cbar'):
        plot_voltage.cbar.remove()
        del plot_voltage.cbar
    #
    # clear complete plot
    ax_voltage.cla()
    #
    #
    # set optional radial range for filtering
    try:
        r_max = np.float32(tb_r_max.text)
    except:
        global cid_tb_r_max
        tb_r_max.disconnect(cid_tb_r_max)
        tb_r_max.set_val("-1.00")
        cid_tb_r_max = tb_r_max.on_submit(update_voltage)
        r_max = np.float32(-1.0)
    #
    #
    # perform voltage correction
    coeffs[0], xy_data, events, fit_data = ms.get_voltage_correction(
        data, (tof, flight_length, (None, None), alpha),
        deg = int(sl_voltage_degree.val),
        hist = {'range': (float(tb_correction_range_min.text),
                          float(tb_correction_range_max.text)),
                'width': sl_voltage_bin_width.val},
        r_max = r_max,
        size = 0.01,
        steps = int(sl_voltage_steps.val),
        thres = sl_voltage_peak_thres.val)
    #
    # set peak target position (weighted average of peak positions of all
    # sub-ranges)
    peak_target = np.average(xy_data[1], weights = events)
    #
    #
    # create plot
    xy_plot = ax_voltage.scatter(*xy_data, c = 1.0e-3 * events)
    plot_voltage.cbar = fig.colorbar(xy_plot, ax = ax_voltage)
    ax_voltage.plot(*fit_data, '-', color = 'C1', label = "fit")
    ax_voltage.axhline(y = peak_target, color = 'black', label = "peak target")
    #
    #
    # set labels and legend
    ax_voltage.set_xlabel("Voltage (kV)")
    ax_voltage.set_ylabel("Peak position (amu/e)")
    plot_voltage.cbar.set_label("Counts ($10^3$)")
    ax_voltage.legend()
#
#
#
#
def print_update_notification(show):
    """
    Display or remove an "UPDATING" overlay to indicate that a background
    operation is in progress.
    """
    #
    #
    # the following section must be executed exclusively
    with r_mutex:
        # initialize nesting counter
        if not hasattr(print_update_notification, 'counter'):
            print_update_notification.counter = 0
        #
        #
        if show:
            # increment counter
            print_update_notification.counter += 1
            #
            # show overlay if it's the first request (might be triggered by
            # several functions in a nested way)
            if not hasattr(print_update_notification, 'text'):
                print_update_notification.text = fig.text(
                    0.5, 0.5, "UPDATING",
                    horizontalalignment = 'center',
                    verticalalignment = 'center',
                    transform = fig.transFigure,
                    color = 'red', fontsize = 100, alpha = 0.40
                )
                fig.canvas.draw()
                fig.canvas.flush_events()
        else:
            # decrement counter
            print_update_notification.counter -= 1
            #
            # remove update notification
            if print_update_notification.counter == 0 and \
            hasattr(print_update_notification, 'text'):
                print_update_notification.text.remove()
                delattr(print_update_notification, 'text')
#
#
#
#
def tb_alpha_update(text):
    """
    Update spectrum scaling factor.
    """
    #
    #
    # get text
    try:
        if np.float32(text) <= 0.0:
            raise Exception
    except:
        warnings.warn("Invalid value for α.")
        return
    global alpha
    alpha = np.float32(text)
    #
    # update full spectrum
    check_full_spec(rb_full_spec.value_selected)
#
#
#
#
def tb_peak_align_update(text):
    """
    Update automatic peak alignment.
    """
    #
    #
    # do nothing when text box empty
    if text == "":
        return
    #
    #
    # get tokens from text box
    tokens = text.split(',')
    if len(tokens) != 4:
        raise Exception("There must be exactly 4 entries in the text box.")
    #
    # set initial peak positions
    if max(int(tokens[0]) - 1, int(tokens[1]) - 1) >= \
       len(plot_full_spectrum.peak_pos):
        raise Exception("Invalid peak number provided.")
    peak_init = plot_full_spectrum.peak_pos[
        [int(tokens[0]) - 1, int(tokens[1]) - 1]]
    #
    # set final peak target positions
    peak_target = np.array([tokens[2], tokens[3]], dtype = float)
    #
    #
    # get peak alignment parameters
    global alpha, tof
    params = ms.peak_align(
        peak_init, peak_target, coeffs[0], flight_length, alpha)
    alpha *= params[0]
    tof   += params[1]
    #
    #
    # set text box values for alpha and time-of-flight offset
    global cid_tb_alpha
    global cid_tb_tof
    tb_alpha.disconnect(cid_tb_alpha)
    tb_tof.disconnect(cid_tb_tof)
    tb_alpha.set_val("{0:.4f}".format(alpha))
    tb_tof.set_val("{0:.3f}".format(tof))
    cid_tb_alpha = tb_alpha.on_submit(tb_alpha_update)
    cid_tb_tof   = tb_tof.on_submit(tb_tof_update)
    #
    # clear text box
    tb_peak_align.set_val("")
    #
    #
    # trigger complete update with optimized parameters
    update_voltage(None)
#
#
#
#
def tb_tof_update(text):
    """
    Update time-of-flight offset.
    """
    #
    #
    # get text
    try:
        np.float32(text)
    except:
        warnings.warn("Invalid value for time-of-flight offset.")
        return
    global tof
    tof = np.float32(text)
    #
    # use callback function for α to replot full spectrum
    tb_alpha_update(tb_alpha.text)
#
#
#
#
def update_correction_range(val):
    """
    Update correction range plot.
    """
    #
    #
    # print update notification
    print_update_notification(True)
    #
    # plot mass spectrum for correction range
    plot_correction_range()
    #
    #
    # conditionally draw full spectrum
    check_full_spec(rb_full_spec.value_selected)
    #
    #
    # remove update notification and update figures
    print_update_notification(False)
    fig.canvas.draw_idle()
#
#
#
#
def update_flight_length(val):
    """
    Update plot for flight length correction.
    """
    #
    #
    # print update notification
    print_update_notification(True)
    #
    #
    # reset all radio buttons which rely on flight length correction
    global cid_rb_fine_corr
    was_fine_active = False
    if rb_fine_corr.value_selected == "fine on":
        was_fine_active = True
    rb_fine_corr.disconnect(cid_rb_fine_corr)
    rb_fine_corr.set_active(0)
    cid_rb_fine_corr = rb_fine_corr.on_clicked(check_fine_corr)
    #
    #
    # remove previous line for voltage and flight length correction in
    # correction range plot if existing
    for line in ax_correction_range.lines:
        if plt.getp(line, 'label') == "voltage & flight corr.":
            line.remove()
            ax_correction_range.legend()
    #
    #
    # conditionally execute radio button operations (do not perform flight
    # length correction if fine correction was active since a complete new cycle
    # is required anyway)
    if was_fine_active == False:
        check_flight_length(rb_flight_length.value_selected)
    check_fine_corr(rb_fine_corr.value_selected)
    check_full_spec(rb_full_spec.value_selected)
    #
    #
    # remove update notification and update figures
    print_update_notification(False)
    fig.canvas.draw_idle()
#
#
#
#
def update_full_spectrum(val):
    """
    Update plot for full spectrum.
    """
    #
    #
    # check for valid range
    try:
        if float(tb_full_spec_min.text) >= float(tb_full_spec_max.text):
            raise Exception
    except:
        warnings.warn("Invalid range for full spectrum.")
        return
    #
    #
    # print update notification
    print_update_notification(True)
    #
    # conditionally plot full spectrum
    check_full_spec(rb_full_spec.value_selected)
    #
    # remove update notification and update figures
    print_update_notification(False)
    fig.canvas.draw_idle()
#
#
#
#
def update_voltage(val):
    """
    Update plot for voltage correction.
    """
    #
    #
    # check for valid correction range
    try:
        if float(tb_correction_range_min.text) >= \
           float(tb_correction_range_max.text):
            raise Exception
    except:
        warnings.warn("Invalid correction range.")
        return
    #
    #
    # print update notification
    print_update_notification(True)
    #
    #
    # reset all radio buttons which rely on voltage correction
    global cid_rb_flight_length
    rb_flight_length.disconnect(cid_rb_flight_length)
    rb_flight_length.set_active(0)
    cid_rb_flight_length = rb_flight_length.on_clicked(check_flight_length)
    #
    global cid_rb_fine_corr
    rb_fine_corr.disconnect(cid_rb_fine_corr)
    rb_fine_corr.set_active(0)
    cid_rb_fine_corr = rb_fine_corr.on_clicked(check_fine_corr)
    #
    #
    # perform voltage and flight length correction
    plot_voltage()
    #
    # plot mass spectrum for correction range
    plot_correction_range()
    #
    #
    # conditionally execute radio button operations
    check_flight_length(rb_flight_length.value_selected)
    check_full_spec(rb_full_spec.value_selected)
    #
    #
    # remove update notification and update figures
    print_update_notification(False)
    fig.canvas.draw_idle()
#
#
#
#
# parse and print arguments
args = parser.parse_args()
print("\nSettings used:")
for key, value in sorted(vars(args).items()):
    print("{0:18s}{1}".format(key, value))
print("")
#
#
#
#
# initialize authorization credentials
auth = None
#
#
#
#
# get database record
record = db_query(args.id, ("device", "custom_id", "parameters"))
device          = record['device']
custom_id       = record['custom_id']
data_filter     = record['parameters'].get('data_filter',     {})
spectrum_params = record['parameters'].get('spectrum_params', {})
#
#
# get measurement interval
if args.interval is not None:
    interval = args.interval
else:
    interval = data_filter.get('interval', None)
#
# get binning width
if args.binning is not None:
    binning = 0.01 * 10**args.binning
else:
    binning = spectrum_params.get('bin_width', 0.01)
#
#
# set mass spectrum alignment parameters
alpha = np.float32(spectrum_params.get('alpha', 1.0))
tof   = np.float32(spectrum_params.get('t_0',   0.0))
#
# set mass-to-charge ratio range
full_range = data_filter.get('mass_charge_range', [0.1, 100.0])
#
#
#
#
# set flight length and detector radius based on provided device
flight_length = np.float32(get_setting(f"devices.{device}.flight_length"))
logger.info(f"Flight length for device \"{device}\" is {flight_length:.3f} mm.")
detector_radius = np.float32(get_setting(f"devices.{device}.detector_radius"))
logger.info(
    f"Detector radius for device \"{device}\" is {detector_radius:.1f} mm."
)
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
# sum voltages and pick relevant columns
data[:, 0] = \
    data[:, 0] + args.pulse_coupling * data[:, 1] + args.detector_voltage
data = data[:, [0, 3, 4, 5]]
#
#
#
#
# check voltage curve if requested
if args.check_voltage == True:
    # set voltage data
    voltage_data = np.column_stack(
        (np.arange(1, len(data) + 1) / len(data), data[:, 0] / 1e3)
    )
    #
    # reduce number of data points to 10.000
    voltage_data = voltage_data[::len(data) // 10000]
    #
    #
    # plot voltage curve
    plt.suptitle("{0:s} ({1:.1f} M events)".format(custom_id, len(data) / 1e6))
    plt.plot(voltage_data[:, 0], voltage_data[:, 1])
    plt.xlabel("Measurement interval")
    plt.ylabel("Voltage (kV)")
    plt.show()
    #
    #
    # export voltage curve
    np.savetxt(
        custom_id + "_voltage_curve.txt", voltage_data,
        fmt = '%.4f\t\t%.4e', header = "Interval\tU (kV)"
    )
    exit(0)
#
#
#
#
# filter measurement interval if requested
if interval is not None:
    # check for valid interval
    if interval[0] < 0.0 or interval[1] > 1.0 or interval[0] >= interval[1]:
        raise Exception(
            f"Invalid measurement interval {interval} has been specified."
        )
    #
    # filter data
    len_init = len(data)
    data = data[int(interval[0] * len_init) : int(interval[1] * len_init)]
    logger.info(
        f"Selected measurement interval {interval} contains {len(data)} events."
    )
#
#
# filter detector hit positions for valid events if requested
if args.filter_detector == True:
    # get detector radius
    r = detector_radius
    #
    # filter data
    len_init = len(data)
    data = data[(data[:, 1]**2 + data[:, 2]**2 <= r**2)]
    logger.info(
        f"{len_init - len(data)} events with invalid detector hit positions "
        "have been deleted."
    )
#
#
# filter voltage range if requested
if args.voltage_range is not None:
    logger.info(f"Using voltage range {args.voltage_range} kV.")
    data = data[(args.voltage_range[0] * 1000 <= data[:, 0]) &
                (data[:, 0] <= args.voltage_range[1] * 1000)]
#
logger.info(f"Number of events is {len(data)}.")
#
#
# enable debug output if requested
if args.debug == True:
    ms.enable_debug(True)
#
#
#
#
# check whether no range for correction has been provided
if args.mass_range is None:
    # use range around maximum peak for correction
    hist, bin_centers, _ = ms.get_mass_spectrum(
        data, (tof, flight_length, (None, None), alpha),
        hist = {'width': 0.03})
    #
    peak_max = bin_centers[np.argmax(hist)]
    args.mass_range = (peak_max - 3, peak_max + 3)
#
#
#
#
# set global correction coefficients to operate on
coeffs = [None, None]
#
#
#
#
# create figure
fig = plt.figure(figsize = (6.4 * 2, 4.8 * 2))
#
#
# register callback function for keyboard input
fig.canvas.mpl_connect('key_press_event', on_key_press)
#
#
# set figure title
fig.suptitle("{0:s} ({1:.1f} M events)".format(custom_id, len(data) / 1e6))
#
#
#
#
# set relative widths and heights of subplots
h_fig     = 15 # relative height of figure
h_spacing = 3  # relative height of spacing between figures
h_sl      = 1  # relative height of slider
#
w_fig     = 5  # relative width of figure
w_spacing = 2  # relative width of spacing between figures
#
#
# create GridSpec for subplots
gs = GridSpec(
    12, 8, hspace = 0.0, wspace = 0.0,
    left = 0.08, top = 0.95, bottom = 0.02, right = 0.92,
    height_ratios = [
        h_fig, h_spacing, h_sl, h_sl, h_sl, h_spacing / 1.5,
        h_fig / 4, h_fig / 4, h_fig / 4, h_fig / 4,
        h_spacing, h_sl],
    width_ratios = [
        1 * w_fig / 5, 2.5 * w_fig / 5, 1.5 * w_fig / 5, w_spacing / 2,
        w_spacing / 2, 1 * w_fig / 5, 2.5 * w_fig / 5, 1.5 * w_fig / 5])
#
#
#
#
# initialize current row and column in GridSpec
row = 0
col = 0
#
#
# create voltage axis
ax_voltage = plt.subplot(gs[row, col:col+4])
# create text box for radial range
ll, bb, ww, hh = ax_voltage.get_position().bounds
tb_r_max = TextBox(
    plt.axes([ll, bb - 0.06, 0.035, 0.025]), '$r_\\mathrm{max}$ (mm)',
    initial = "{0:.2f}".
              format(0.6 * detector_radius), label_pad = 0.05)
row += 2
#
# create voltage sliders
sl_voltage_bin_width = Slider(
    ax = plt.subplot(gs[row, col:col+3]), label = "Bin width",
    valmin = 0.01, valmax = 0.5, valinit = 0.02, valstep = 0.01,
    valfmt = '%0.2f', dragging = False)
row += 1
sl_voltage_steps = Slider(
    ax = plt.subplot(gs[row, col:col+3]), label = "Steps",
    valmin = 5, valmax = 50, valinit = 25, valstep = 1, valfmt = '%d',
    dragging = False)
row += 1
sl_voltage_degree = Slider(
    ax = plt.subplot(gs[row, col]), label = "DoF",
    valmin = 0, valmax = 5, valinit = 1, valstep = 1, valfmt = '%d',
    dragging = False)
sl_voltage_peak_thres = Slider(
    ax = plt.subplot(gs[row, col+2]), label = "Peak thresh.",
    valmin = 0.1, valmax = 1.0, valinit = 0.9, valstep = 0.1, valfmt = '%.1f',
    dragging = False)
row += 2
#
#
# create correction range axis
ax_correction_range = plt.subplot(gs[row:row+4, col:col+3])
#
# create text boxes for range
ll, bb, ww, hh = ax_correction_range.get_position().bounds
tb_correction_range_min = TextBox(
    plt.axes([ll, bb - 0.055, 0.03, 0.025]), 'min',
    initial = "{0:.1f}".format(args.mass_range[0]), label_pad = 0.05)
tb_correction_range_max = TextBox(
    plt.axes([ll + ww - 0.03, bb - 0.055, 0.03, 0.025]), 'max',
    initial = "{0:.1f}".format(args.mass_range[1]))
label = tb_correction_range_max.ax.get_children()[0]
label.set_position([1.05, 0.5])
label.set_horizontalalignment('left')
#
#
# create radio buttons
rb_flight_length = RadioButtons(
    ax = plt.subplot(gs[row, col+3]), labels = ('flight off', 'flight on'),
    active = 1)
rb_fine_corr = RadioButtons(
    ax = plt.subplot(gs[row+1, col+3]), labels = ('fine off', 'fine on'),
    active = 0)
rb_full_spec = RadioButtons(
    ax = plt.subplot(gs[row+2, col+3]), labels = ('full off', 'full on'),
    active = 0)
rb_log_scale = RadioButtons(
    ax = plt.subplot(gs[row+3, col+3]), labels = ('linear', 'log'),
    active = 0)
row += 5
#
#
# create sliders for correction range and full spectrum
sl_hist_bin_width = Slider(
    ax = plt.subplot(gs[row, col:col+3]), label = "Bin width",
    valmin  = 0.1 * binning,
    valmax  = binning,
    valinit = binning,
    valstep = 0.1 * binning,
    valfmt = "%0.{0:d}f".format(int(-np.log10(0.1 * binning))),
    dragging = False)
col += 5
#
#
# create flight length correction axis
row = 0
ax_flight_length = plt.subplot(gs[row, col:col+3], projection = '3d')
row += 2
#
# create flight length correction sliders
sl_flight_length_bin_width = Slider(
    ax = plt.subplot(gs[row, col:col+3]), label = "Bin width",
    valmin = 0.01, valmax = 0.5, valinit = 0.05, valstep = 0.01,
    valfmt = '%0.2f', dragging = False)
row += 1
sl_flight_length_steps = Slider(
    ax = plt.subplot(gs[row, col:col+3]), label = "Steps",
    valmin = 5, valmax = 50, valinit = 15, valstep = 1, valfmt = '%d',
    dragging = False)
row += 1
sl_flight_length_peak_thres = Slider(
    ax = plt.subplot(gs[row, col+2]), label = "Peak thresh.",
    valmin = 0.1, valmax = 1.0, valinit = 0.7, valstep = 0.1, valfmt = '%.1f',
    dragging = False)
row += 2
#
#
# create full spectrum axis
ax_full_spec = plt.subplot(gs[row:row+4, col:col+3])
row += 5
#
# create text boxes for range
ll, bb, ww, hh = ax_full_spec.get_position().bounds
tb_full_spec_min = TextBox(
    plt.axes([ll, bb - 0.055, 0.035, 0.025]), 'min',
    initial = full_range[0], label_pad = 0.05)
tb_full_spec_max = TextBox(
    plt.axes([ll + ww - 0.035, bb - 0.055, 0.035, 0.025]), 'max',
    initial = full_range[1])
label = tb_full_spec_max.ax.get_children()[0]
label.set_position([1.05, 0.5])
label.set_horizontalalignment('left')
#
#
# create text boxes for peak adjustment
ll, bb, ww, hh = ax_full_spec.get_position().bounds
tb_alpha = TextBox(
    plt.axes([ll + 0.0, bb + hh, 0.045, 0.025]), "$α$",
    initial = "{0:.4f}".format(alpha))
tb_tof = TextBox(
    plt.axes([ll + 0.090, bb + hh, 0.050, 0.025]), "$t_0$ (ns)",
    initial = "{0:.3f}".format(tof))
tb_peak_align = TextBox(
    plt.axes([ll + ww - 0.120, bb + hh, 0.120, 0.025]), "Peak selection",
    initial = "")
#
#
# create radio button for SQL upload
bt_upload = Button(plt.axes([ll + ww + 0.01, bb + hh, 0.050, 0.025]), "Upload")
#
#
# create sliders for full spectrum
sl_full_peak_thres = Slider(
    ax = plt.subplot(gs[row, col:col+3]), label = "Peak thresh.",
    valmin = 0.001, valmax = 0.1, valinit = 0.5, valstep = 0.001,
    valfmt = '%.3f', dragging = False)
#
#
#
#
# show empty figure while doing initial calculations
plt.show(block = False)
plot_init()
#
#
#
#
# register callback functions
#
# voltage correction triggers complete update
sl_voltage_bin_width.on_changed(update_voltage)
sl_voltage_steps.on_changed(update_voltage)
sl_voltage_degree.on_changed(update_voltage)
sl_voltage_peak_thres.on_changed(update_voltage)
cid_tb_r_max = tb_r_max.on_submit(update_voltage)
#
# adjusting correction range and bin width triggers complete update
tb_correction_range_min.on_submit(update_voltage)
tb_correction_range_max.on_submit(update_voltage)
sl_hist_bin_width.on_changed(update_voltage)
#
#
# flight length correction only triggers affected plots
sl_flight_length_bin_width.on_changed(update_flight_length)
sl_flight_length_steps.on_changed(update_flight_length)
sl_flight_length_peak_thres.on_changed(update_flight_length)
#
#
# radio buttons trigger specific check functions
cid_rb_flight_length = rb_flight_length.on_clicked(check_flight_length)
cid_rb_fine_corr = rb_fine_corr.on_clicked(check_fine_corr)
cid_rb_full_spec = rb_full_spec.on_clicked(check_full_spec)
cid_rb_log_scale = rb_log_scale.on_clicked(check_log_scale)
#
# full spectrum is updated by respective widgets
tb_full_spec_min.on_submit(update_full_spectrum)
tb_full_spec_max.on_submit(update_full_spectrum)
sl_full_peak_thres.on_changed(update_full_spectrum)
#
# adjusting peak alignment triggers specific update functions
cid_tb_alpha      = tb_alpha.on_submit(tb_alpha_update)
cid_tb_tof        = tb_tof.on_submit(tb_tof_update)
cid_tb_peak_align = tb_peak_align.on_submit(tb_peak_align_update)
#
#
# callback function for SQL upload
cid_bt_upload = bt_upload.on_clicked(db_upload)
#
#
#
#
# show plots
plt.show()
#
#
#
#
# upload spectrum parameters to SQL database
db_upload(None)
#
#
# auto-export complete spectrum
hist, bin_centers, _ = ms.get_mass_spectrum(
    data, (tof, flight_length, coeffs, alpha),
    hist = {'width': sl_hist_bin_width.val})
np.savetxt(custom_id + ".txt",
           np.column_stack((bin_centers, hist)),
           fmt = '%+08.3f\t %7d', header = "m/q (amu/e)\t  counts")
