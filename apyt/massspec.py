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
import warnings
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
    _debug("Performing flight length correction...")
    #
    # get optional keyword arguments
    deg      = kwargs.get('deg', 2)
    hist_par = kwargs.get('hist', {})
    thres    = kwargs.get('thres', 0.9)
    #
    #
    # limit data to certain mass-to-charge ratio range
    data = _filter_mass_to_charge_range(data, spec_par, hist_par)
    #
    #
    #
    #
    # perform auto-tuning of peak threshold parameter
    iteration = 1
    iteration_max = 5
    while True:
        # get peak positions per detector segment
        x, y, z, events = _get_segment_peaks(data, spec_par, hist_par, thres,
                                             **kwargs)
        #
        # check for sufficient peaks
        if len(z) < (deg + 1) * (deg + 2) // 2:
            raise Exception("Insufficient number of peaks ({0:d}) detected for "
                            "flight length correction "
                            "(must be at least {1:d}).".
                            format(len(z), (deg + 1) * (deg + 2) // 2))
        #
        #
        # fit correction function to peak positions
        _debug("Correcting flight length using polynomial of degree {0:d}.".
               format(deg))
        coeffs = _polyfit2d(x, y, np.average(z, weights = events) / z, deg,
                            weights = events)
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
                peak_str += "\n{0:+7.3f}\t\t{1:+7.3f}\t\t{2:8d}\t\t" \
                            "{3:.3f}\t\t{4:.3f}".format(
                                elem[0], elem[1], elem[2], elem[3], elem[4])
            _debug(peak_str)
        #
        #
        # set variances before and after correction
        var_init = np.var(z)
        var = np.var(z * np.polynomial.polynomial.polyval2d(x, y, coeffs))
        _debug("Variance of initial peak positions:   {0:.3f} amu/e.".
               format(var_init))
        _debug("Variance of corrected peak positions: {0:.3f} amu/e.".
               format(var))
        #
        #
        # check for sufficient reduction of variance after correction
        if var >= 0.1 * var_init:
            # check whether maximum number of iterations reached
            if iteration == iteration_max:
                raise Exception("Flight length correction with automatic "
                                "parameter tuning failed.")
            #
            # reduce peak threshold for next iteration
            warnings.warn("Insufficient reduction in variance detected "
                          "(multiple peaks in range?). Reducing peak threshold "
                          "from {0:.1f} to {1:.1f}.".format(thres, thres - 0.1))
            thres = thres - 0.1
        else:
            # exit loop on success
            break;
        #
        #
        # increment iteration counter
        iteration += 1
    #
    #
    #
    #
    # show plot if requested
    if kwargs.get('plot', False) == True:
        plt.legend(loc = "upper left")
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
def get_voltage_correction(data, spec_par, **kwargs):
    start = timer()
    _debug("Performing voltage correction...")
    #
    # get optional keyword arguments
    deg      = kwargs.get('deg', 3)
    hist_par = kwargs.get('hist', {})
    plot     = kwargs.get('plot', False)
    size     = kwargs.get('size', 0.3)
    steps    = kwargs.get('steps', 40)
    thres    = kwargs.get('thres', 0.9)
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
    _debug("Minimum required events per range are {0:d}.".
           format(int(size * len(data))))
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
            # plot histogram if requested
            if plot == True:
                plt.plot(bin_centers, hist,
                         label = "({0:7.1f}, {1:7.1f}) V, #{2:d}".
                                 format(U_low, U_high, len(data_cur)))
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
    if len(y) <= deg:
        raise Exception("Insufficient number of peaks ({0:d}) detected for "
                        "voltage correction (must be at least {0:d}).".
                        format(len(z), deg + 1))
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
    _debug("Variance of initial peak positions:   {0:.3f} amu/e.".
           format(np.var(y)))
    _debug("Variance of corrected peak positions: {0:.3f} amu/e.".
           format(np.var(y * polyval(x, coeffs))))
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
        peaks, _ = find_peaks(hist, distance = np.iinfo(np.int16).max)
        _debug("Maximum peak is at {0:.1f} amu/e.".
               format(bin_centers[peaks[0]]))
        data_range = (bin_centers[peaks[0]] - 10, bin_centers[peaks[0]] + 10)
    _debug("Using range ({0:.1f}, {1:.1f}) amu/e.".
           format(data_range[0], data_range[1]))
    #
    # return filtered data
    return data[(data_range[0] <= mc_ratio) & (mc_ratio <= data_range[1])]
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
def _get_mass_to_charge_ratio(data, par):
    # check for valid flight length
    if par[1] <= 0.0:
        raise Exception("Flight length ({0:.1f}) must be positive.".
                        format(par[1]))
    #
    #
    # calculate mass-to-charge ratio
    mc_ratio = data[:, 0] * (data[:, 3] - par[0])**2 / \
               (par[1]**2 + data[:, 1]**2 + data[:, 2]**2) * \
               _mc_conversion_factor
    #
    # apply voltage correction if provided
    if par[2][0] is not None:
        mc_ratio *= polyval(data[:, 0], par[2][0])
    # apply positional correction if provided
    if par[2][1] is not None:
        mc_ratio *= polyval2d(data[:, 1], data[:, 2], par[2][1])
    #
    # return (corrected) mass-to-charge ratio
    return mc_ratio
#
#
#
#
def _get_segment_peaks(data, spec_par, hist_par, thres_local, **kwargs):
    # get optional keyword arguments
    plot  = kwargs.get('plot', False)
    size  = kwargs.get('size', 0.3)
    steps = kwargs.get('steps', 15)
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
    _debug("Minimum required events per range are {0:d}.".
           format(int(size * len(data) / (steps * steps))))
    #
    #
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
            if len(data_xy_cur) >= size * len(data) / (steps * steps):
                # calculate histogram and bin centers for current data
                hist, bin_centers, _ = get_mass_spectrum(
                    data_xy_cur, spec_par, hist = hist_par)
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
                peaks, _ = find_peaks(hist, height = thres_local * hist_max)
                #
                # check for valid peaks (should not fail!)
                if len(peaks) == 0:
                    raise Exception("No peaks detected for xy-range "
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
    # return peak position per sector
    return x, y, z, events
#
#
#
#
def _optimize_flight_correction(data, spec_par, hist_par):
    _debug("Optimizing flight length correction...")
    #
    # parse coefficients
    voltage_coeffs = spec_par[2][0]
    flight_coeffs  = _poly2d_coeff_mat_to_vec(spec_par[2][1])
    _debug("Initial width is {0:.3f} amu/e.".
           format(_peak_width(flight_coeffs[1:], data, spec_par[0], spec_par[1],
                              (voltage_coeffs, flight_coeffs[0]), hist_par,
                              'flight')))
    #
    #
    # optimize flight length correction
    minimization_result = minimize(
        _peak_width, flight_coeffs[1:],
        args = (data, spec_par[0], spec_par[1],
                (voltage_coeffs, flight_coeffs[0]), hist_par, 'flight'),
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
def _optimize_voltage_correction(data, spec_par, hist_par):
    _debug("Optimizing voltage correction...")
    #
    # parse coefficients
    voltage_coeffs = spec_par[2][0]
    flight_coeffs  = spec_par[2][1]
    _debug("Initial width is {0:.3f} amu/e.".
           format(_peak_width(voltage_coeffs[1:], data, spec_par[0],
                  spec_par[1], (voltage_coeffs[0], flight_coeffs), hist_par,
                  'voltage')))
    #
    #
    # optimize voltage correction
    minimization_result = minimize(
        _peak_width, voltage_coeffs[1:],
        args = (data, spec_par[0], spec_par[1],
                (voltage_coeffs[0], flight_coeffs), hist_par, 'voltage'),
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
def _peak_width(x, data, t_0, L_0, coeffs_stripped, hist_par, mode):
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
    #
    # calculate histogram and bin centers
    hist, bin_centers, _ = get_mass_spectrum(
        data, (t_0, L_0, coeffs), hist = hist_par)
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
