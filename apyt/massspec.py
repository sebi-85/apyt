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


.. sectionauthor:: Sebastian M. Eich <Sebastian.Eich@imw.uni-stuttgart.de>
.. moduleauthor::  Sebastian M. Eich <Sebastian.Eich@imw.uni-stuttgart.de>

"""
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
    'optimize_correction'
]
#
#
#
#
# import modules
import matplotlib.pyplot as plt
import numpy as np
#
# import some special functions/modules
from inspect import getframeinfo, stack
from numba import njit
from numpy.polynomial.polynomial import polyval, polyval2d, polyvander2d
from scipy import constants
from scipy.optimize import minimize
from scipy.signal import find_peaks, peak_widths
from sys import stderr
from timeit import default_timer as timer
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
_is_dbg = False
"The global flag for debug output"
_mc_conversion_factor = 2.0 * constants.value('elementary charge') / \
                        constants.value('atomic mass constant') * 1.0e-12
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
def get_mass_spectrum(data, t_0, L_0, **kwargs):
    # get optional keyword arguments
    coeffs = kwargs.get('coeffs', (None, None))
    range  = kwargs.get('range', None)
    width  = kwargs.get('width', _default_bin_width)
    #
    # calculate mass-to-charge ratio
    mc_ratio = _get_mass_to_charge_ratio(data, t_0, L_0, coeffs)
    #
    # set default histogram range if nothing provided
    if range is None:
        range = (np.floor(mc_ratio.min()), np.ceil(mc_ratio.max()))
    # set number of histogram bins
    bins = int((range[1] - range[0]) / width)
    #
    # calculate histogram and bin centers
    hist, bin_edges = np.histogram(mc_ratio, bins = bins, range = range)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    #
    # return histogram and bin centers
    return hist, bin_centers, mc_ratio
#
#
#
#
def get_flight_correction(data, t_0, L_0, **kwargs):
    start = timer()
    _debug("Performing flight length correction...")
    #
    # get optional keyword arguments
    coeffs     = kwargs.get('coeffs', (None, None))
    data_range = kwargs.get('range', None)
    deg        = kwargs.get('deg', 2)
    plot       = kwargs.get('plot', False)
    size       = kwargs.get('size', 0.01)
    steps      = kwargs.get('steps', 15)
    thres      = kwargs.get('thres', 0.9)
    width      = kwargs.get('width', _default_bin_width)
    #
    #
    # limit data to certain mass-to-charge ratio range
    data = _filter_mass_to_charge_range(
        data, t_0, L_0, data_range, coeffs, width)
    #
    #
    # get detector position range
    x_min, y_min = np.amin(data[:, 1:3], axis = 0)
    x_max, y_max = np.amax(data[:, 1:3], axis = 0)
    Δx = (x_max - x_min) / steps
    Δy = (y_max - y_min) / steps
    _debug("x-range is ({0:.2f}, {1:.2f}) mm "
           "(Δx = {2:.3f} mm, {3:d} steps).".format(x_min, x_max, Δx, steps))
    _debug("y-range is ({0:.2f}, {1:.2f}) mm "
           "(Δy = {2:.3f} mm, {3:d} steps).".format(y_min, y_max, Δy, steps))
    _debug("Minimum required events per range are {0:d}.".format(
           int(size * len(data))))
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
            # only use data if size is above threshold
            if len(data_xy_cur) >= size * len(data):
                # calculate histogram and bin centers for current data
                hist, bin_centers, _ = get_mass_spectrum(
                    data_xy_cur, t_0, L_0, coeffs = coeffs, width = width)
                #
                # plot histogram if requested
                if plot == True:
                    plt.plot(bin_centers, hist,
                             label = "({0:+5.1f}, {1:+5.1f}) mm,"
                                     "({2:+5.1f}, {3:+5.1f}) mm , #{4:d}".
                                     format(x_low, x_low + Δx,
                                            y_low, y_low + Δy,
                                            len(data_xy_cur)))
                #
                #
                # find histogram peaks (there may be multiple peaks close to one
                # another, so we will always pick the first one later)
                hist_max = hist.max()
                peaks, _ = find_peaks(hist, height = thres * hist_max)
                #
                # check for valid peaks (should not fail!)
                if len(peaks) == 0:
                    _error("No peaks detected for xy-range "
                           "({0:.1f}, {1:.1f}), ({2:.1f}, {3:.1f})!".
                           format(x_low, x_high, y_low, y_high))
                #
                #
                # add peak to plot if requested
                if plot == True:
                    plt.plot(bin_centers[peaks[0]], hist[peaks[0]], "x")
                #
                #
                # append fit data
                x = np.append(x, x_low + 0.5 * Δx)
                y = np.append(y, y_low + 0.5 * Δy)
                z = np.append(z, bin_centers[peaks[0]])
                events = np.append(events, len(data_xy_cur))
    #
    #
    # check for valid peaks
    if len(z) == 0:
        _error("No peaks detected for flight length correction.")
    #
    #
    # fit correction function to peak positions
    _debug("Correcting flight length using polynomial of degree {0:d}.".
           format(deg))
    coeffs = _polyfit2d(
        x, y, np.average(z, weights = events) / z, deg, weights = events)
    #
    # we set correction in the center of the detector to unity by definition
    coeffs = coeffs / coeffs[0, 0]
    _debug("Polynomial coefficients are: {0:s}.".format(str(coeffs)))
    #
    #
    # print detected peaks if requested
    if _is_dbg == True:
        peak_str = "Peaks have been detected at:\n# x (mm)\ty (mm)\t\t" \
                   "  events\t\tm/q (amu/e)\tm/q (amu/e) (corr.)"
        for elem in list(zip(x, y, events, z,
                             z * np.polynomial.polynomial.polyval2d(
                                 x, y, coeffs))):
            peak_str += "\n{0:+7.3f}\t\t{1:+7.3f}\t\t{2:8d}\t\t{3:.3f}\t\t" \
                        "{4:.3f}".format(elem[0], elem[1], elem[2], elem[3],
                                         elem[4])
        _debug(peak_str)
    _debug("Variance of initial peak positions:   {0:.3f} amu/e.".format(
           np.var(z)))
    _debug("Variance of corrected peak positions: {0:.3f} amu/e.".format(
           np.var(z * np.polynomial.polynomial.polyval2d(x, y, coeffs))))
    #
    #
    # show plot if requested
    if plot == True:
        plt.legend(loc="upper left")
        plt.show()
    #
    #
    # return coefficients for correction function
    end = timer()
    _debug("Flight correction took {0:.3f} seconds.".format(end - start))
    return coeffs
#
#
#
#
def get_voltage_correction(data, t_0, L_0, **kwargs):
    start = timer()
    _debug("Performing voltage correction...")
    #
    # get optional keyword arguments
    coeffs     = kwargs.get('coeffs', (None, None))
    data_range = kwargs.get('range', None)
    deg        = kwargs.get('deg', 3)
    plot       = kwargs.get('plot', False)
    size       = kwargs.get('size', 0.01)
    steps      = kwargs.get('steps', 40)
    thres      = kwargs.get('thres', 0.9)
    width      = kwargs.get('width', _default_bin_width)
    #
    #
    # limit data to certain mass-to-charge ratio range
    data = _filter_mass_to_charge_range(
        data, t_0, L_0, data_range, coeffs, width)
    #
    #
    # get voltage range
    U_min = np.amin(data[:, 0])
    U_max = np.amax(data[:, 0])
    ΔU = (U_max - U_min) / steps
    _debug("Voltage range is ({0:.1f}, {1:.1f}) V "
           "(ΔU = {2:.3f} V, {3:d} steps).".format(U_min, U_max, ΔU, steps))
    _debug("Minimum required events per range are {0:d}.".format(
           int(size * len(data))))
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
        if len(data_cur) >= size * len(data):
            # calculate histogram and bin centers for current data
            hist, bin_centers, _ = get_mass_spectrum(
                data_cur, t_0, L_0, coeffs = coeffs, width = width)
            #
            # plot histogram if requested
            if plot == True:
                plt.plot(bin_centers, hist,
                         label = "({0:7.1f}, {1:7.1f}) V, #{2:d}".format(
                                  U_low, U_high, len(data_cur)))
            #
            #
            # find histogram peaks (there may be multiple peaks close to one
            # another, so we will always pick the first one later)
            hist_max = hist.max()
            peaks, _ = find_peaks(hist, height = thres * hist_max)
            #
            # check for valid peaks (should not fail!)
            if len(peaks) == 0:
                _error("No peaks detected for voltage range "
                       "({0:.1f}, {1:.1f})!". format(U_low, U_high))
            #
            #
            # add peak to plot if requested
            if plot == True:
                plt.plot(bin_centers[peaks[0]], hist[peaks[0]], "x")
            #
            #
            # append fit data
            x = np.append(x, U_low + 0.5 * ΔU)
            y = np.append(y, bin_centers[peaks[0]])
            events = np.append(events, len(data_cur))
    #
    #
    # check for valid peaks
    if len(y) == 0:
        _error("No peaks detected for voltage correction.")
    #
    #
    # fit correction function to peak positions
    _debug("Correcting voltage using polynomial of degree {0:d}.".format(deg))
    coeffs = np.polynomial.polynomial.polyfit(
        x, np.average(y, weights = events) / y, deg, w = events)
    _debug("Polynomial coefficients are: {0:s}.".format(str(coeffs)))
    #
    #
    # print detected peaks if requested
    if _is_dbg == True:
        peak_str = "Peaks have been detected at:\n# U (V)\t\t  events\t\t" \
                   "m/q (amu/e)\tm/q (amu/e) (corr.)"
        for elem in list(zip(x, events, y, y * polyval(x, coeffs))):
            peak_str += "\n{0:7.1f}\t\t{1:8d}\t\t{2:.3f}\t\t{3:.3f}".format(
                elem[0], elem[1], elem[2], elem[3])
        _debug(peak_str)
    _debug("Variance of initial peak positions:   {0:.3f} amu/e.".format(
           np.var(y)))
    _debug("Variance of corrected peak positions: {0:.3f} amu/e.".format(
           np.var(y * polyval(x, coeffs))))
    #
    #
    # show plot if requested
    if plot == True:
        plt.legend(loc="upper left")
        plt.show()
    #
    #
    # return coefficients for correction function
    end = timer()
    _debug("Voltage correction took {0:.3f} seconds.".format(end - start))
    return coeffs
#
#
#
#
def optimize_correction(mode, *args):
    if mode == 'voltage':
        return _optimize_voltage_correction(*args)
    elif mode == 'flight':
        return _optimize_flight_correction(*args)
    else:
        _error("Unrecognized mode for minimization ({0:s}).".format(mode))
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
    print("[DEBUG] ({0:s}:{1:d}) {2:s}".format(
        frameinfo.function, frameinfo.lineno, msg), file = stderr)
#
#
#
#
def _error(msg):
    # print error message including function name and line number
    frameinfo = getframeinfo(stack()[1].frame)
    print("[ERROR] ({0:s}:{1:d}) {2:s}".format(
        frameinfo.function, frameinfo.lineno, msg), file = stderr)
    exit(1)
#
#
#
#
def _filter_mass_to_charge_range(data, t_0, L_0, range, coeffs, width):
    # calculate histogram and mass-to-charge ratio
    hist, bin_centers, mc_ratio = get_mass_spectrum(
        data, t_0, L_0, coeffs = coeffs, width = width)
    #
    #
    # if no range provided, use fixed width of 20 around maximum peak
    if range is None:
        _debug("Auto-detecting range...")
        peaks, _ = find_peaks(hist, distance = np.iinfo(np.int16).max)
        _debug("Maximum peak is at {0:.1f} amu/e.".format(
               bin_centers[peaks[0]]))
        range = (bin_centers[peaks[0]] - 10, bin_centers[peaks[0]] + 10)
    _debug("Using range ({0:.1f}, {1:.1f}) amu/e.".format(range[0], range[1]))
    #
    # return filtered data
    return data[(range[0] <= mc_ratio) & (mc_ratio <= range[1])]
#
#
#
#
@njit
def _filter_range(data, col, low, high):
    return data[(low < data[:, col]) & (data[:, col] <= high)]
#
#
#
#
def _get_mass_to_charge_ratio(data, t_0, L_0, coeffs):
    # calculate mass-to-charge ratio
    mc_ratio = data[:, 0] * (data[:, 3] - t_0)**2 / \
               (L_0**2 + data[:, 1]**2 + data[:, 2]**2) * _mc_conversion_factor
    #
    # apply voltage correction if provided
    if coeffs[0] is not None:
        mc_ratio *= polyval(data[:, 0], coeffs[0])
    # apply positional correction if provided
    if coeffs[1] is not None:
        mc_ratio *= polyval2d(data[:, 1], data[:, 2], coeffs[1])
    #
    # return (corrected) mass-to-charge ratio
    return mc_ratio
#
#
#
#
def _optimize_flight_correction(data, t_0, L_0, coeffs):
    _debug("Optimizing flight length correction...")
    #
    # parse coefficients
    voltage_coeffs = coeffs[0]
    flight_coeffs  = _poly2d_coeff_mat_to_vec(coeffs[1])
    _debug("Initial width is {0:.3f} amu/e.".format(
        _peak_width(flight_coeffs[1:], data, t_0, L_0,
        (voltage_coeffs, flight_coeffs[0]), 'flight')))
    #
    #
    # optimize flight length correction
    minimization_result = minimize(
        _peak_width, flight_coeffs[1:],
        args = (data, t_0, L_0, (voltage_coeffs, flight_coeffs[0]), 'flight'),
        method = 'nelder-mead',
        options = {'fatol': 1e-2, 'disp': True, 'maxiter': 100})
    #
    #
    # re-assemble coefficients for flight length correction
    flight_coeffs = np.append(flight_coeffs[0], minimization_result.x)
    flight_coeffs = _poly2d_coeff_vec_to_mat(flight_coeffs)
    _debug("Optimized width is {0:.3f} amu/e.".format(minimization_result.fun))
    _debug("Optimized coefficients for flight length correction are {0:s}.".
           format(str(flight_coeffs)))
    #
    #
    # return optimized coefficients for flight length correction
    return flight_coeffs
#
#
#
#
def _optimize_voltage_correction(data, t_0, L_0, coeffs):
    _debug("Optimizing voltage correction...")
    #
    # parse coefficients
    voltage_coeffs = coeffs[0]
    flight_coeffs  = coeffs[1]
    _debug("Initial width is {0:.3f} amu/e.".format(
        _peak_width(voltage_coeffs[1:], data, t_0, L_0,
        (voltage_coeffs[0], flight_coeffs), 'voltage')))
    #
    #
    # optimize voltage correction
    minimization_result = minimize(
        _peak_width, voltage_coeffs[1:],
        args = (data, t_0, L_0, (voltage_coeffs[0], flight_coeffs), 'voltage'),
        method = 'nelder-mead',
        options = {'fatol': 1e-2, 'disp': True, 'maxiter': 100})
    #
    #
    # re-assemble coefficients for voltage correction
    voltage_coeffs = np.append(voltage_coeffs[0], minimization_result.x)
    _debug("Optimized width is {0:.3f} amu/e.".format(minimization_result.fun))
    _debug("Optimized coefficients for voltage correction are {0:s}.".
           format(str(voltage_coeffs)))
    #
    #
    # return optimized coefficients for voltage correction
    return voltage_coeffs
#
#
#
#
def _peak_width(x, data, t_0, L_0, coeffs, mode):
    # re-assemble complete set of coefficients for correction functions
    if mode == 'voltage':
        coeffs = (np.append(coeffs[0], x), coeffs[1])
    elif mode == 'flight':
        # coefficients are passed in vector form, but we need matrix
        # representation
        coeffs = (coeffs[0],
                  _poly2d_coeff_vec_to_mat(np.append(coeffs[1], x)))
    else:
        _error("Unrecognized mode for minimization ({0:s}).".format(mode))
    #
    #
    # calculate histogram and bin centers
    ### PASS WIDTH AND RANGE ??? ###
    hist, bin_centers, _ = get_mass_spectrum(data, t_0, L_0, coeffs = coeffs)
    #
    # get maximum peak and its width
    peaks, _   = find_peaks(hist, distance = np.iinfo(np.int16).max)
    width_half = peak_widths(hist, peaks, rel_height = 0.5)[0][0]
    #
    #
    # return width of maximum peak
    return width_half
#
#
#
#
def _polyfit2d(x, y, f, deg, **kwargs):
    # get optional keyword arguments
    weights = kwargs.get('weights', np.ones_like(f))
    #
    # create pseudo-Vandermonde matrix
    vander = polyvander2d(x, y, [deg, deg])
    #
    # create mask to filter higher-order terms in Vandermode matrix, which also
    # contains terms up to x**deg * y**deg
    mask = np.rot90(np.tri(deg + 1, dtype = bool), k = -1).reshape(-1)
    vander = vander[:, mask]
    #
    # perform least-squares fit (with weights)
    c = np.linalg.lstsq(vander * np.sqrt(weights[:, np.newaxis]),
                        f * np.sqrt(weights), rcond = None)[0]
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
