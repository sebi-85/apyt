"""
The APyT mass spectrum module
=============================

This module enables the automatic evaluation of mass spectra from raw
measurement data including optimizing routines to obtain peaks with maximum
possible sharpness.


Howto
-----

The usage of this module is demonstrated in an auxiliary script
(``wrapper_scripts/apyt_massspec.py``) which basically serves as a wrapper for
this module. Detailed usage information can be obtained by invoking this script
with the ``"--help"`` option.


List of methods
---------------

This module provides some generic functions for the calculation of mass spectra
from raw measurement data.

The following methods are provided:

* :meth:`enable_debug`: Enable or disable debug output.
* :meth:`get_mass_spectrum`: Calculate mass spectrum.
* :meth:`get_flight_correction`: Obtain coefficients for flight length
  correction.
* :meth:`get_voltage_correction`: Obtain coefficients for voltage correction.
* :meth:`optimize_correction`: Automatically optimize correction coefficients.
* :meth:`write_xml`: Write XML file for subsequent usage.


.. sectionauthor:: Sebastian M. Eich <Sebastian.Eich@imw.uni-stuttgart.de>
.. moduleauthor::  Sebastian M. Eich <Sebastian.Eich@imw.uni-stuttgart.de>

"""
#
#
# TODO: check DOF for voltage correction
# TODO: check consistent use of float32
# TODO: improve speed of optimization
#
#
#
#
__version__ = '0.1.0'
__all__ = [
    'enable_debug',
    'get_mass_spectrum',
    'get_flight_correction',
    'get_voltage_correction',
    'optimize_correction',
    'write_xml'
]
#
#
#
#
# import modules
import matplotlib.pyplot as plt
import numba
import numpy as np
import warnings
import xml.etree.ElementTree as ET
#
# import some special functions/modules
from datetime import datetime
from inspect import getframeinfo, stack
from numpy.polynomial.polynomial import polyfit, polyval, polyval2d, \
                                        polyvander2d
from os import getpid
from psutil import Process
from resource import getrusage, RUSAGE_SELF
from scipy import constants
from scipy.optimize import minimize
from scipy.signal import find_peaks, peak_widths
from sys import stderr
from timeit import default_timer as timer
from xml.dom import minidom
#
#
#
#
# set numba configuration for parallelization
numba.config.THREADING_LAYER = 'omp'
numba.set_num_threads(4)
#
#
#
#
################################################################################
#
# private module-level variables
#
################################################################################
_default_bin_width = 0.05
"The default histogram bin width."
_dtype = np.float32
"""The type of the input data.

This enforces memory-intensive arrays to be of the same data type."""
_is_dbg = False
"The global flag for debug output"
_mc_conversion_factor = np.float32(
    2.0 * constants.value('elementary charge') /
    constants.value('atomic mass constant') * 1.0e-12)
"The internal constant conversion factor for the mass spectrum."
#
#
#
#
################################################################################
#
# public functions
#
################################################################################
def enable_debug(is_dbg):
    global _is_dbg
    _is_dbg = is_dbg
#
#
#
#
def get_mass_spectrum(data, spec_par, **kwargs):
    # get optional keyword arguments
    hist_par = kwargs.get('hist', {})
    #
    # get histogram parameters
    data_range = hist_par.get('range', None)
    width      = hist_par.get('width', _default_bin_width)
    #
    #
    # calculate mass-to-charge ratio
    mc_ratio = _get_mass_to_charge_ratio(data, spec_par)
    #
    # set default histogram range if nothing provided
    if data_range is None:
        data_range = (np.floor(mc_ratio.min()), np.ceil(mc_ratio.max()))
    # set number of histogram bins
    bins = int((data_range[1] - data_range[0]) / width)
    #
    # calculate histogram and bin centers
    hist, bin_edges = np.histogram(mc_ratio, bins = bins, range = data_range)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    #
    # return histogram, bin centers, and mass-to-charge ratios
    return hist, bin_centers, mc_ratio
#
#
#
#
def get_flight_correction(data, spec_par, **kwargs):
    start = timer()
    print("Performing flight length correction...")
    #
    # get optional keyword arguments
    deg      = kwargs.get('deg', 2)
    hist_par = kwargs.get('hist', {})
    size     = kwargs.get('size', 0.3)
    steps    = kwargs.get('steps', 15)
    thres    = kwargs.get('thres', 0.9)
    #
    #
    # limit data to certain mass-to-charge ratio range
    data = _filter_mass_to_charge_range(data, spec_par, hist_par)
    #
    #
    # get detector position range
    x_min, y_min = np.amin(data[:, 1:3], axis = 0)
    x_max, y_max = np.amax(data[:, 1:3], axis = 0)
    Δx = (x_max - x_min) / steps
    Δy = (y_max - y_min) / steps
    _debug("x-range is ({0:.2f}, {1:.2f}) mm (Δx = {2:.3f} mm, {3:d} steps).".
           format(x_min, x_max, Δx, steps))
    _debug("y-range is ({0:.2f}, {1:.2f}) mm (Δy = {2:.3f} mm, {3:d} steps).".
           format(y_min, y_max, Δy, steps))
    _debug("Minimum required events per range are {0:d} ({1:.2f}%).".
           format(int(size * len(data) / (steps * steps)),
                  size / (steps * steps) * 100))
    #
    #
    # initialize fit data
    x = np.array([])
    y = np.array([])
    z = np.array([])
    events = np.array([], dtype = int)
    #
    # loop through x steps
    for xi in range(0, steps):
        # set x-range for current step
        x_low  = x_min + xi * Δx
        x_high = x_low + Δx
        #
        # get data for current x-range
        data_x_cur = _filter_range(data, 1, x_low, x_high)
        #
        #
        # loop through y steps
        for yi in range(0, steps):
            # set y-range for current step
            y_low  = y_min + yi * Δy
            y_high = y_low + Δy
            #
            # get data for current xy-range
            data_xy_cur = _filter_range(data_x_cur, 2, y_low, y_high)
            #
            #
            # only use data if size is above threshold
            if len(data_xy_cur) >= size * len(data) / (steps * steps):
                # calculate histogram and bin centers for current data
                hist, bin_centers, _ = get_mass_spectrum(
                    data_xy_cur, spec_par, hist = hist_par)
                #
                #
                # find histogram peaks (there may be multiple peaks close to one
                # another, so we will always pick the first one later)
                hist_max = hist.max()
                peaks, _ = find_peaks(hist, height = thres * hist_max)
                #
                # check for valid peaks (should not fail!)
                if len(peaks) == 0:
                    raise Exception("No peaks detected for xy-range "
                                    "({0:.1f}, {1:.1f}), ({2:.1f}, {3:.1f})!".
                                    format(x_low, x_high, y_low, y_high))
                #
                #
                # append fit data
                x = np.append(x, x_low + 0.5 * Δx)
                y = np.append(y, y_low + 0.5 * Δy)
                z = np.append(z, bin_centers[peaks[0]])
                events = np.append(events, len(data_xy_cur))
    #
    #
    # check for sufficient peaks
    if len(z) < (deg + 1) * (deg + 2) // 2:
        raise Exception("Insufficient number of peaks ({0:d}) detected for "
                        "flight length correction (must be at least {1:d}).".
                        format(len(z), (deg + 1) * (deg + 2) // 2))
    #
    #
    # fit correction function to peak positions (with fixed absolute offset)
    _debug("Correcting flight length using polynomial of degree {0:d}.".
           format(deg))
    peak_target = polyval2d(0.0, 0.0, _polyfit2d(
        x, y, z, deg, weights = events))
    _debug("Peak target position is {0:.2f} amu/e.".format(peak_target))
    coeffs = _polyfit2d(x, y, peak_target / z, deg,
                        weights = events, offset = 1.0)
    _debug("Polynomial coefficients are: {0:s}.".format(str(coeffs)))
    #
    #
    # print detected peaks if requested
    if _is_dbg == True:
        peak_str = "Peaks (#{0:d}) have been detected at:\n" \
                   "# x (mm)\t  y (mm)\t  events\trel_size\tm/q (amu/e)\t" \
                   "m/q (amu/e) (corr.)".format(len(z))
        for elem in list(zip(x, y, events, events / len(data) * 100, z,
                             z * polyval2d(x, y, coeffs))):
            peak_str += "\n{0:+8.3f}\t{1:+8.3f}\t{2:8d}\t{3:7.1f}%\t" \
                        "{4:.3f}\t\t{5:.3f}". \
                        format(elem[0], elem[1], elem[2], elem[3], elem[4],
                               elem[5])
        _debug(peak_str)
    #
    #
    # set standard deviation before and after correction
    std_init = np.std(z)
    std = np.std(z * polyval2d(x, y, coeffs))
    _debug("Standard deviation of initial peak positions:   {0:.3f} amu/e.".
           format(std_init))
    _debug("Standard deviation of corrected peak positions: {0:.3f} amu/e.".
           format(std))
    #
    #
    # check for sufficient reduction in standard deviation after correction
    if std >= 0.3 * std_init:
        warnings.warn("Insufficient reduction in standard deviation detected "
                      "(multiple peaks in range?). You may try to reduce the "
                      "threshold.")
    #
    #
    # construct wireframe data obtained from fit function
    X = np.meshgrid(np.linspace(x_min + 0.5 * Δx, x_max - 0.5 * Δx, steps),
                    np.linspace(y_min + 0.5 * Δy, y_max - 0.5 * Δy, steps))
    wireframe = (*X, peak_target / polyval2d(*X, coeffs))
    #
    #
    # return coefficients for correction function
    end = timer()
    _debug("Flight correction took {0:.3f} seconds.".format(end - start))
    return coeffs.astype(_dtype), (x, y, z), events, wireframe
#
#
#
#
def get_voltage_correction(data, spec_par, **kwargs):
    start = timer()
    print("Performing voltage correction...")
    #
    # get optional keyword arguments
    deg      = kwargs.get("deg", 3)
    hist_par = kwargs.get("hist", {})
    size     = kwargs.get("size", 0.3)
    steps    = kwargs.get("steps", 20)
    thres    = kwargs.get("thres", 0.9)
    #
    #
    # limit data to certain mass-to-charge ratio range
    data = _filter_mass_to_charge_range(data, spec_par, hist_par)
    #
    #
    # get voltage range
    U_min = np.amin(data[:, 0])
    U_max = np.amax(data[:, 0])
    ΔU = (U_max - U_min) / steps
    _debug("Voltage range is ({0:.1f}, {1:.1f}) V "
           "(ΔU = {2:.3f} V, {3:d} steps).".format(U_min, U_max, ΔU, steps))
    _debug("Minimum required events per range are {0:d} ({1:.2f}%).".
           format(int(size * len(data) / steps), size / steps * 100))
    #
    #
    # initialize fit data
    x = np.array([])
    y = np.array([])
    events = np.array([], dtype = int)
    #
    # loop through voltage steps
    for i in range(0, steps):
        # set voltage range for current step
        U_low  = U_min + i * ΔU
        U_high = U_low + ΔU
        #
        # get data for current range
        data_cur = _filter_range(data, 0, U_low, U_high)
        #
        # only use data if size is above threshold
        if len(data_cur) >= size * len(data) / steps:
            # calculate histogram and bin centers for current data
            hist, bin_centers, _ = get_mass_spectrum(
                data_cur, spec_par, hist = hist_par)
            #
            #
            # find histogram peaks (there may be multiple peaks close to one
            # another, so we will always pick the first one later)
            hist_max = hist.max()
            peaks, _ = find_peaks(hist, height = thres * hist_max)
            #
            # check for valid peaks (should not fail!)
            if len(peaks) == 0:
                raise Exception("No peaks detected for voltage range "
                                "({0:.1f}, {1:.1f})!".format(U_low, U_high))
            #
            #
            # append fit data
            x = np.append(x, U_low + 0.5 * ΔU)
            y = np.append(y, bin_centers[peaks[0]])
            events = np.append(events, len(data_cur))
    #
    #
    # check for valid peaks
    if len(y) <= deg:
        raise Exception("Insufficient number of peaks ({0:d}) detected for "
                        "voltage correction (must be at least {1:d}).".
                        format(len(y), deg + 1))
    #
    #
    # fit correction function to peak positions
    _debug("Correcting voltage using polynomial of degree {0:d}.".format(deg))
    peak_target = np.average(y, weights = events)
    _debug("Peak target position is {0:.2f} amu/e.".format(peak_target))
    coeffs = polyfit(x, x * (peak_target / y - 1.0), deg)
    _debug("Polynomial coefficients are: {0:s}.".format(str(coeffs)))
    #
    #
    # print detected peaks if requested
    if _is_dbg == True:
        peak_str = "Peaks (#{0:d}) have been detected at:\n# U (V)\t\t  " \
                   "events\trel_size\tm/q (amu/e)\tm/q (amu/e) (corr.)". \
                   format(len(y))
        for elem in list(zip(x, events, events / len(data) * 100, y,
                             y * (1.0 + polyval(x, coeffs) / x))):
            peak_str += "\n{0:7.1f}\t\t{1:8d}\t{2:7.1f}%\t{3:.3f}\t\t{4:.3f}". \
                        format(elem[0], elem[1], elem[2], elem[3], elem[4])
        _debug(peak_str)
    _debug("Standard deviation of initial peak positions:   {0:.3f} amu/e.".
           format(np.std(y)))
    _debug("Standard deviation of corrected peak positions: {0:.3f} amu/e.".
           format(np.std(y * (1.0 + polyval(x, coeffs) / x))))
    #
    #
    # construct xy-data obtained from fit
    x_fit = np.linspace(U_min + 0.5 * ΔU, U_max - 0.5 * ΔU, 100)
    xy_fit = (0.001 * x_fit,
              peak_target / (1.0 + polyval(x_fit, coeffs) / x_fit))
    #
    #
    # return coefficients for correction function
    end = timer()
    _debug("Voltage correction took {0:.3f} seconds.".format(end - start))
    return coeffs.astype(_dtype), (0.001 * x, y), events, xy_fit
#
#
#
#
def optimize_correction(data, spec_par, mode, **kwargs):
    # get optional keyword arguments
    hist_par = kwargs.get('hist', {})
    #
    #
    if mode == 'voltage':
        return _optimize_voltage_correction(data, spec_par, hist_par)
    elif mode == 'flight':
        return _optimize_flight_correction(data, spec_par, hist_par)
    else:
        raise Exception("Unrecognized mode for minimization ({0:s}).".
                        format(mode))
#
#
#
#
def write_xml(file, data, spec_par, steps):
    # check for valid correction coefficients
    if spec_par[2][0] is None or spec_par[2][1] is None:
        raise Exception("Correction coefficients have not been set.")
    #
    #
    #
    #
    # set minimum and maximum voltage
    U_min = data[:, 0].min()
    if U_min <= 0.0:
        U_min = 1.0
    U_max = data[:, 0].max()
    #
    # set voltage grid points
    U = np.linspace(U_min, U_max, steps[0])
    _debug("Voltage grid points are:\n" + str(U))
    #
    # set voltage correction points
    U_corr = 1.0 + 1.0 / U * polyval(U, spec_par[2][0])
    _debug("Voltage correction values are:\n" + str(U_corr))
    #
    #
    #
    #
    # set detector diameter based on absolute maximum hit position
    diameter = 2.0 * max(
        abs(data[:, 1].min()), abs(data[:, 1].max()),
        abs(data[:, 2].min()), abs(data[:, 2].max()))
    _debug("Detector diameter is {0:.3f} mm.".format(diameter))
    #
    # create meshgrid for evaluation of detector correction points
    X, Y = np.meshgrid(
        np.linspace(-diameter / 2, diameter / 2, steps[1]),
        np.linspace(-diameter / 2, diameter / 2, steps[1]))
    _debug("Detector grid points are:\n" +
           str(np.vstack(list(map(np.ravel, (X, Y)))).T))
    #
    # set detector position correction points
    det_corr = polyval2d(X, Y, spec_par[2][1]).flatten()
    _debug("Detector correction values are:\n" + str(det_corr))
    #
    #
    #
    #
    # create xml document root
    root = ET.Element("TAP-Parameters")
    root.insert(0, ET.Comment(
        " Automatically created by the massspec module from the APyT package "
        "({0:s}). ".format(str(datetime.now()))))
    root.insert(1, ET.Comment(
        " Information available at: "
        "https://apyt.mp.imw.uni-stuttgart.de/apyt.massspec.html "))
    #
    #
    # create flight length and time offset elements
    ET.SubElement(root, "item", {
        "description": "Flightlength",
        "unit": "mm"}
    ).text = "{0:.3f}".format(spec_par[1])
    ET.SubElement(root, "item", {
        "description": "Time-of-flight offset",
        "unit": "ns"}
    ).text = "{0:.3f}".format(spec_par[0])
    #
    #
    # create voltage correction element
    ET.SubElement(root, "voltage-correction", {
        "delta": "{0:.6f}".format((U_max - U_min) / (steps[0] - 1)),
        "max":   "{0:.6f}".format(U_max),
        "min":   "{0:.6f}".format(U_min),
        "size":  "{0:d}".format(steps[0])}
    ).text = ','.join(map(lambda s: "{0:.6f}".format(s), U_corr))
    #
    #
    # create detector correction element
    ET.SubElement(root, "flightlength-correction", {
        "height":       "{0:.6f}".format(diameter),
        "height-delta": "{0:.6f}".format(diameter / (steps[1] - 1)),
        "height-size":  "{0:d}".format(steps[1]),
        "width":        "{0:.6f}".format(diameter),
        "width-delta":  "{0:.6f}".format(diameter / (steps[1] - 1)),
        "width-size":   "{0:d}".format(steps[1])}
    ).text = ','.join(map(lambda s: "{0:.6f}".format(s), det_corr))
    #
    #
    #
    #
    # create xml tree
    tree = ET.ElementTree(root)
    #
    #
    #
    # indentation for ElementTree requires Python 3.9 or higher; use minidom
    # for pretty indentaion
    xmlstr = minidom.parseString(ET.tostring(root)). \
             toprettyxml(encoding = 'ISO-8859-1')
    #
    #
    # write xml file
    with open(file, "wb") as f:
        f.write(xmlstr)
#
#
#
#
################################################################################
#
# private module-level functions
#
################################################################################
def _debug(msg):
    # do nothing in none-debug mode
    if _is_dbg == False:
        return
    #
    # print debug message including function name and line number
    frameinfo = getframeinfo(stack()[1].frame)
    print("[DEBUG] ({0:s}:{1:d}) {2:s}".
          format(frameinfo.function, frameinfo.lineno, msg), file = stderr)
#
#
#
#
def _filter_mass_to_charge_range(data, spec_par, hist_par):
    # get range from histogram parameter dictionary
    data_range = hist_par.get('range', None)
    #
    #
    # calculate histogram and mass-to-charge ratio
    hist, bin_centers, mc_ratio = get_mass_spectrum(
        data, spec_par, hist = hist_par)
    #
    #
    # if no range provided, use fixed width of 20 around maximum peak
    if data_range is None:
        _debug("Auto-detecting range...")
        peaks, _ = find_peaks(hist, distance = np.iinfo(np.int32).max)
        _debug("Maximum peak is at {0:.1f} amu/e.".
               format(bin_centers[peaks[0]]))
        data_range = (bin_centers[peaks[0]] - 10, bin_centers[peaks[0]] + 10)
    #
    #
    # filter range
    data = __filter_mass_to_charge_range(data, data_range, mc_ratio)
    _debug("Using range ({0:.1f}, {1:.1f}) amu/e ({2:d} events).".
           format(data_range[0], data_range[1], len(data)))
    #
    #
    # return filtered data
    return data
#
#
#
#
@numba.njit("f4[:, :](f4[:, :], UniTuple(f8, 2), f4[:])", parallel = True)
def __filter_mass_to_charge_range(data, range, mc_ratio):
    return data[(range[0] <= mc_ratio) & (mc_ratio <= range[1])]
#
#
#
#
@numba.njit("f4[:, :](f4[:, :], i8, f4, f4)", parallel = True)
def _filter_range(data, col, low, high):
    return data[(low < data[:, col]) & (data[:, col] <= high)]
#
#
#
#
@numba.njit("f4[:](f4[:, :], f4[:], f4, f4)", parallel = True)
def __get_mass_to_charge_ratio(data, U, t_0, L_0):
    return U * (data[:, 3] - t_0)**2 / \
           (L_0**2 + data[:, 1]**2 + data[:, 2]**2) * _mc_conversion_factor
#
#
#
#
def _get_mass_to_charge_ratio(data, par):
    # unpack parameters for better readability
    t_0, L_0, (voltage_coeffs, flight_coeffs) = par
    #
    #
    # check for correct input data type
    if data.dtype is not np.dtype(_dtype):
        raise TypeError("Wrong type for raw input data ({0:s}). Must be "
                        "'{1:s}'.".format(str(data.dtype),
                                          str(np.dtype(_dtype))))
    # check for valid flight length
    if L_0 <= 0.0:
        raise Exception("Flight length ({0:.1f}) must be positive.".format(L_0))
    #
    #
    # apply voltage correction if provided
    if voltage_coeffs is None:
        # (this only creates a view)
        U = data[:, 0]
    else:
        if voltage_coeffs.dtype is not np.dtype(_dtype):
            raise TypeError("Wrong type for voltage coefficients ({0:s}). Must "
                            "be '{1:s}'.".format(str(voltage_coeffs.dtype),
                                                 str(np.dtype(_dtype))))
        U = data[:, 0] + polyval(data[:, 0], voltage_coeffs)
    #
    #
    # calculate mass-to-charge ratio
    mc_ratio = __get_mass_to_charge_ratio(data, U, t_0, L_0)
    #
    #
    # apply positional correction if provided
    if flight_coeffs is not None:
        if flight_coeffs.dtype is not np.dtype(_dtype):
            raise TypeError("Wrong type for flight length coefficients "
                            "({0:s}). Must be '{1:s}'.".format(
                                str(flight_coeffs.dtype),
                                str(np.dtype(_dtype))))
        mc_ratio *= polyval2d(data[:, 1], data[:, 2], flight_coeffs)
    #
    #
    # return (corrected) mass-to-charge ratio
    if mc_ratio.dtype is not np.dtype(_dtype):
        raise TypeError("Wrong type for mass spectrum data ({0:s}). Must be "
                        "'{1:s}'.".format(str(mc_ratio.dtype),
                                          str(np.dtype(_dtype))))
    return mc_ratio
#
#
#
#
def _mem():
    # set debug message
    msg = "Current memory usage is {0:.1f} MB (peak {1:.1f} MB).".format(
              Process(getpid()).memory_info().rss / 1024**2,
              getrusage(RUSAGE_SELF).ru_maxrss / 1024)
    #
    # print debug message
    frameinfo = getframeinfo(stack()[1].frame)
    print("[DEBUG] ({0:s}:{1:d}) {2:s}".
          format(frameinfo.function, frameinfo.lineno, msg), file = stderr)
#
#
#
#
def _optimize_flight_correction(data, spec_par, hist_par):
    print("Optimizing flight length correction...")
    #
    #
    # get initial peak position and width
    peak_pos_init, peak_width_init = _peak_width(data, spec_par, hist_par)
    _debug("Initial peak position is at {0:.3f} amu/e (width {1:.3f} amu/e).".
           format(peak_pos_init, peak_width_init))
    #
    #
    # parse coefficients
    voltage_coeffs = spec_par[2][0]
    flight_coeffs  = _poly2d_coeff_mat_to_vec(spec_par[2][1])
    #
    #
    # optimize flight length correction
    minimization_result = minimize(
        _peak_width_minimizer, flight_coeffs[1:],
        args = (data, spec_par[0], spec_par[1],
                (voltage_coeffs, flight_coeffs[0]), hist_par, 'flight'),
        method = 'nelder-mead',
        options = {'fatol': 1e-2, 'disp': True, 'maxiter': 100})
    #
    #
    # re-assemble coefficients for flight length correction
    flight_coeffs = np.append(flight_coeffs[0], minimization_result.x)
    flight_coeffs = _poly2d_coeff_vec_to_mat(flight_coeffs).astype(_dtype)
    #
    #
    # get peak position and width for optimized coefficients
    peak_pos_final, peak_width_final = _peak_width(
        data, (spec_par[0], spec_par[1], (voltage_coeffs, flight_coeffs)),
        hist_par)
    _debug("Final peak position is at {0:.3f} amu/e (width {1:.3f} amu/e).".
           format(peak_pos_final, peak_width_final))
    _debug("Optimized coefficients for flight length correction are {0:s}.".
           format(str(flight_coeffs)))
    #
    #
    # return optimized coefficients for flight length correction
    return flight_coeffs.astype(_dtype)
#
#
#
#
def _optimize_voltage_correction(data, spec_par, hist_par):
    print("Optimizing voltage correction...")
    #
    #
    # get initial peak position and width
    peak_pos_init, peak_width_init = _peak_width(data, spec_par, hist_par)
    _debug("Initial peak position is at {0:.3f} amu/e (width {1:.3f} amu/e).".
           format(peak_pos_init, peak_width_init))
    #
    #
    # parse coefficients
    voltage_coeffs = spec_par[2][0]
    flight_coeffs  = spec_par[2][1]
    #
    #
    # optimize voltage correction
    minimization_result = minimize(
        _peak_width_minimizer, voltage_coeffs[1:],
        args = (data, spec_par[0], spec_par[1],
                (voltage_coeffs[0], flight_coeffs), hist_par, 'voltage',
                {'peak_target': peak_pos_init}),
        method = 'nelder-mead',
        options = {'fatol': 1e-2, 'disp': True, 'maxiter': 100})
    #
    #
    # re-assemble coefficients for voltage correction
    voltage_coeffs = np.append(voltage_coeffs[0],
                               minimization_result.x.astype(_dtype))
    #
    #
    # get peak position and width for optimized coefficients
    peak_pos_final, peak_width_final = _peak_width(
        data, (spec_par[0], spec_par[1], (voltage_coeffs, flight_coeffs)),
        hist_par)
    #
    #
    # set correction factor to maintain initial peak position
    alpha = peak_pos_init / peak_pos_final
    _debug("Correction factor to maintain peak position is {0:.6f}.".
           format(alpha))
    #
    # scale coefficients to maintain initial peak position
    voltage_coeffs    *= alpha
    voltage_coeffs[1] += alpha - 1.0
    _debug("Final peak position is at {0:.3f} amu/e (width {1:.3f} amu/e).".
           format(peak_pos_final * alpha, peak_width_final * alpha))
    _debug("Optimized coefficients for voltage correction are {0:s}.".
           format(str(voltage_coeffs)))
    #
    #
    # return optimized coefficients for voltage correction
    return voltage_coeffs.astype(_dtype)
#
#
#
#
def _peak_width(data, spec_par, hist_par):
    # calculate histogram and bin centers
    hist, bin_centers, _ = get_mass_spectrum(data, spec_par, hist = hist_par)
    #
    # get maximum peak and its width
    peaks, _   = find_peaks(hist, distance = np.iinfo(np.int32).max)
    width_half = peak_widths(hist, peaks, rel_height = 0.5)[0][0]
    #
    # return peak positions and widths
    return bin_centers[peaks[0]], width_half * hist_par["width"]
#
#
#
#
def _peak_width_minimizer(x, data, t_0, L_0, coeffs_stripped, hist_par, mode,
                          dict_args = None):
    # get arguments passed as optional dictionary (scipy.optimize.minimize does
    # not allow for regular **kwargs as additional function arguments)
    if dict_args is None:
        dict_args = {'peak_target': None}
    peak_target = dict_args.get('peak_target')
    #
    #
    # re-assemble complete set of coefficients for correction functions
    if mode == 'voltage':
        coeffs = (np.append(coeffs_stripped[0], x), coeffs_stripped[1])
    elif mode == 'flight':
        # coefficients are passed in vector form, but we need matrix
        # representation
        coeffs = (coeffs_stripped[0],
                  _poly2d_coeff_vec_to_mat(np.append(coeffs_stripped[1], x)))
    else:
        raise Exception("Unrecognized mode for minimization ({0:s}).".
                        format(mode))
    #
    # convert coefficients to required data type
    coeffs = (coeffs[0].astype(_dtype), coeffs[1].astype(_dtype))
    #
    #
    # get maximum peak and its width
    peak_pos, peak_width = _peak_width(data, (t_0, L_0, coeffs), hist_par)
    #
    #
    # if target position for peak provided, scale peak widths accordingly
    if peak_target is not None:
        peak_width *= peak_target / peak_pos
    #
    #
    # return width of maximum peak
    return peak_width
#
#
#
#
def _polyfit2d(x, y, f, deg, **kwargs):
    # get optional keyword arguments
    weights = kwargs.get('weights', np.ones_like(f))
    offset  = kwargs.get('offset', None)
    #
    # create pseudo-Vandermonde matrix
    vander = polyvander2d(x, y, [deg, deg])
    #
    # create mask to filter higher-order terms in Vandermode matrix, which also
    # contains terms up to x**deg * y**deg
    mask = np.rot90(np.tri(deg + 1, dtype = bool), k = -1).reshape(-1)
    vander = vander[:, mask]
    #
    #
    # if constant offset is provided, exclude absolute term from least-squares
    # fit
    if offset is not None:
        vander = vander[:, 1:]
        f = f - offset
    #
    #
    # perform least-squares fit (with weights)
    c = np.linalg.lstsq(vander * np.sqrt(weights[:, np.newaxis]),
                        f * np.sqrt(weights), rcond = None)[0]
    #
    #
    # re-insert constant offset for complete coefficient vector
    if offset is not None:
        c = np.insert(c, 0, offset)
    #
    #
    # return coefficient matrix which can directly be passed to polyval2d
    return _poly2d_coeff_vec_to_mat(c)
#
#
#
#
def _poly2d_coeff_mat_to_vec(M):
    # get degree from matrix
    deg = M.shape[0] - 1
    #
    # convert full coefficient matrix to vector
    vec = M.reshape(-1)
    #
    # create mask to filter higher-order terms
    mask = np.rot90(np.tri(deg + 1, dtype = bool), k = -1).reshape(-1)
    #
    # return sparse coefficient vector
    return vec[mask]
#
#
#
#
def _poly2d_coeff_vec_to_mat(v):
    # determine degree from length of vector representation
    deg = np.rint(-3.0 / 2.0 + np.sqrt(1.0 / 4.0 + 2 * len(v))).astype(int)
    #
    # initialize coefficient matrix
    M = np.zeros((deg + 1, deg + 1))
    #
    # set coefficients from sparse coefficient vector
    M[np.mask_indices(deg + 1, lambda m, k : np.rot90(np.triu(m, k)))] = v
    #
    # return full coefficient matrix
    return M
