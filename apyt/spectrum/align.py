"""
The APyT mass spectrum alignment module
=======================================

This module provides methods for the semi-automatic alignment of high-quality
mass spectra from raw atom probe measurement data. It includes routines for
correcting instrumental and geometric distortions to optimize peak sharpness in
the mass-to-charge spectrum.

The resulting alignment parameters can be used in subsequent processing steps,
such as analytical peak fitting or 3D reconstruction.


General function parameter description
--------------------------------------

Event data
^^^^^^^^^^

Input data is expected as an *ndarray* of shape *(n, 4)* and data type
*float32*, where each row represents one of the :math:`n` measured events. The
four columns must contain:

- Measured voltage :math:`U` (in V);
- Detector hit positions :math:`x`, :math:`y` (in mm);
- Time-of-flight :math:`t` (in ns).

(Refer to the raw file :ref:`format<apyt.io.conv:Raw file format>` for more
details.)


Physical spectrum parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The mass-to-charge ratio is computed using the following expression:

.. math::
    \\frac m q = \\alpha \\frac{2 (U + \\varphi(U)) (t - t_0)^2}
                               {(L_0^2 + x^2 + y^2) \\psi(x, y)},

where:

- :math:`\\varphi(U)` — Voltage correction function;
- :math:`t_0` — Time-of-flight offset;
- :math:`L_0` — Nominal tip-to-detector distance;
- :math:`\\psi(x, y)` — Flight length correction based on detector hit position;
- :math:`\\alpha` — Scaling factor adjusting the voltage-to-distance ratio.

The corrections are defined as follows:

- :math:`\\varphi(U)` and :math:`\\psi(x, y)` are represented by polynomial
  functions using |polyval| and |polyval2d| from NumPy.
- The complete correction and alignment are specified by a tuple of parameters:
  (:math:`t_0`, :math:`L_0`, (voltage_coeffs, flight_coeffs), :math:`\\alpha`),
  where:

  - Coefficients are given as an *ndarray* (or `None` to disable correction);
  - All values must be of type *float32*.

Note:

- The center of the detector is assumed to have no correction:
  :math:`\\psi(0, 0) ≡ 1`.
- The fix point of voltage correction, :math:`U_\\textnormal{fix}`, is defined
  where :math:`\\varphi(U_\\textnormal{fix}) = 0`, as demonstrated in the
  following figure:

  .. figure:: img/apyt.spectrum.align.voltage_correction.png
      :align: center
      :alt: Voltage correction
      :width: 500

      Example of a voltage correction curve. The fix point is the intersection
      of the peak-weighted average (black line) with the fit to the peak
      position data from voltage subranges (orange line). Here:
      :math:`U_\\textnormal{fix} \\approx 12\\,\\text{kV}`.

Because :math:`\\alpha` and :math:`t_0` remain adjustable even after correction,
they are used in a final alignment step (see :func:`peak_align`).


Histogram parameters
^^^^^^^^^^^^^^^^^^^^

The mass spectrum is generated using the |numpy.histogram| function. You may
pass the following parameters via a dictionary:

- ``range`` — Lower and upper bounds for histogram binning.
- ``width`` — Desired bin width (used to construct the bins). Defaults to
  **0.05 amu/e**.


Automatic peak alignment
------------------------

After voltage and flight path corrections have been applied to optimize peak
sharpness, the spectrum must still be aligned to known peak positions.

This is achieved by automatically adjusting the two free parameters:

- The scaling factor :math:`\\alpha`;
- The time offset :math:`t_0`.

Alignment is performed by selecting two prominent peaks and assigning them known
target positions. These two known points are sufficient to determine the optimal
values of :math:`\\alpha` and :math:`t_0`.

.. figure:: img/apyt.spectrum.align.peak_align_init.png
    :align: center
    :alt: Peak alignment
    :width: 500

    Peak selection for spectrum alignment.

After optimization, the resulting values are applied to finalize the spectrum:

.. figure:: img/apyt.spectrum.align.peak_align_final.png
    :align: center
    :alt: Peak alignment
    :width: 500

    Final aligned spectrum with corrected peak positions.


List of functions
-----------------

This module provides the following functions for spectrum correction and
alignment:

* :func:`enable_debug`: Enable or disable debug output.
* :func:`get_flight_correction`: Obtain coefficients for flight length
  correction.
* :func:`get_mass_spectrum`: Calculate mass spectrum.
* :func:`get_voltage_correction`: Obtain coefficients for voltage correction.
* :func:`optimize_correction`: Automatically optimize correction coefficients.
* :func:`peak_align`: Automatically align peak positions.


.. |polyval| raw:: html

    <a href="https://numpy.org/doc/stable/reference/generated/
    numpy.polynomial.polynomial.polyval.html" target="_blank">polyval</a>

.. |polyval2d| raw:: html

    <a href="https://numpy.org/doc/stable/reference/generated/
    numpy.polynomial.polynomial.polyval2d.html" target="_blank">polyval2d</a>

.. |numpy.histogram| raw:: html

    <a href="https://numpy.org/doc/stable/reference/generated/
    numpy.histogram.html" target="_blank">numpy.histogram</a>


.. sectionauthor:: Sebastian M. Eich <Sebastian.Eich@imw.uni-stuttgart.de>
.. codeauthor::    Sebastian M. Eich <Sebastian.Eich@imw.uni-stuttgart.de>
"""
#
#
#
#
__version__ = '0.1.0'
__all__ = [
    'enable_debug',
    'get_flight_correction',
    'get_mass_spectrum',
    'get_voltage_correction',
    'optimize_correction',
    'peak_align',
]
#
#
#
#
# import modules
import numba
import numpy as np
import warnings
#
# import some special functions/modules
from datetime import datetime
from inspect import getframeinfo, stack
from numpy.polynomial.polynomial import polyfit, polyval, polyval2d, \
                                        polyvander2d
from os import getpid
from psutil import Process
from scipy import constants
from scipy.optimize import fsolve, minimize
from scipy.signal import find_peaks, peak_widths
from sys import stderr
from timeit import default_timer as timer
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
"The default histogram bin width (amu/e)."
_dtype = np.float32
"""The type of the input data.

This enforces memory-intensive arrays to be of the same data type and avoids
implicit type casting."""
_np_float = np.float32
"The function to be used to set the default float type of numpy scalars."
_is_dbg = False
"""The global flag for debug output.

This flag can be set through the :func:`enable_debug` function."""
_mc_conversion_factor = _np_float(
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
    """Enable or disable debug output.

    Parameters
    ----------
    is_dbg : bool
        Whether to enable or disable debug output.
    """
    #
    #
    global _is_dbg
    _is_dbg = is_dbg
#
#
#
#
def get_flight_correction(data, spec_par, **kwargs):
    """Obtain coefficients for flight length correction.

    In order to perform the flight length correction, the detector is first
    divided into a regular grid and for each of the corresponding detector
    segments, the position of the maximum peak is determined. By definition, the
    peak position in the very center of the detector is used as the peak target
    position for all detector segments. A 2d polynomial will then be used to map
    all determined peak positions to the peak target position, resulting in the
    flight length correction.

    Parameters
    ----------
    data : ndarray, shape (n, 4)
        The *n* measured events, as described in
        :ref:`event data<apyt.spectrum.align:Event data>`.
    spec_par : tuple
        The physical parameters used to calculate the mass-to-charge spectrum,
        as described in :ref:`spectrum parameters
        <apyt.spectrum.align:Physical spectrum parameters>`.

    Keyword Arguments
    -----------------
    deg : int
        The degree of the polynomial used for correction. Defaults to ``2``.
    hist : dict
        The (optional) histogram parameters used to create the mass-to-charge
        histogram, as described in
        :ref:`histogram parameters<apyt.spectrum.align:Histogram parameters>`.
    size : float
        The minimum required size to take a certain detector segment into
        account for correction. The size is given relative to the number of
        events which would be expected for a homogeneous distribution. This
        option effectively filters segments with a low number of events.
        Defaults to ``0.3``.
    steps : int
        The number of segments (for both :math:`x`- and :math:`y`-direction)
        into which the detector is divided. Defaults to ``15``.
    thres : float
        The threshold used to determine peaks (relative to the maximum peak). If
        multiple peaks of similar height fall into the correction window,
        different (maximum) peaks may be picked up for different detector
        segments. By reducing the threshold for the peak detection, all of these
        peaks with similar height will be picked up which are above the
        specified threshold. In order to ensure consistent evaluation, always
        the first of these peaks is then chosen for the correction. Defaults to
        ``0.9``.

    Returns
    -------
    coeffs : ndarray, shape (deg + 1,)
        The polynomial coefficients obtained for correction (of type *float32*).
    (x, y, peak_pos) : tuple
        The tuple containing the data of the detected peak positions for every
        detector segment, each of type *ndarray* with *shape (m,)*.
    events : ndarray, shape (m,)
        The number of events in each detector segment. Can be used for
        color-coding.
    wireframe : tuple
        The :math:`(x, y, z)` data needed to construct a wireframe, as obtained
        by the fit function. The result can be passed directly to the
        |plot_wireframe| function from the *matplotlib* module.


    .. |plot_wireframe| raw:: html

        <a href="https://matplotlib.org/stable/api/_as_gen/
        mpl_toolkits.mplot3d.axes3d.Axes3D.html
        #mpl_toolkits.mplot3d.axes3d.Axes3D.plot_wireframe" target="_blank">
        plot_wireframe()</a>
    """
    #
    #
    start = timer()
    print("Performing flight length correction…")
    #
    # get optional keyword arguments
    deg = kwargs.get('deg', 2)
    if deg < 0:
        warnings.warn("Polynomial degree must be positive. Resetting to \"2\".")
        deg = 2
    hist_par = kwargs.get('hist', {})
    size = kwargs.get('size', 0.3)
    if size < 0.0:
        warnings.warn("Size must not be negative. Resetting to \"0.3\".")
        size = 0.3
    steps = kwargs.get('steps', 15)
    if steps <= 0:
        warnings.warn("Steps must be positive. Resetting to \"15\".")
        steps = 15
    thres = kwargs.get('thres', 0.9)
    if thres < 0.0 or thres > 1.0:
        warnings.warn("Peak threshold must be between 0.0 and 1.0. Resetting "
                      "to \"0.9\".")
        thres = 0.9
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
        data_x_cur = __filter_range(data, 1,
                                    _np_float(x_low), _np_float(x_high))
        #
        #
        # loop through y steps
        for yi in range(0, steps):
            # set y-range for current step
            y_low  = y_min + yi * Δy
            y_high = y_low + Δy
            #
            # get data for current xy-range
            data_xy_cur = __filter_range(data_x_cur, 2,
                                        _np_float(y_low), _np_float(y_high))
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
                peaks, _ = find_peaks(hist, height = thres * hist.max())
                #
                # check for valid peaks (might fail if we only capture the flank
                # of the highest peak, i.e. peak maximum is slightly out of
                # range; simply issue a warning and ignore this segment)
                if len(peaks) == 0:
                    warnings.warn("No peaks detected for xy-range "
                                  "({0:.1f}, {1:.1f}), ({2:.1f}, {3:.1f})!".
                                  format(x_low, x_high, y_low, y_high))
                    continue
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
    coeffs = _polyfit2d(x, y, z / peak_target, deg,
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
                             z / polyval2d(x, y, coeffs))):
            peak_str += "\n{0:+8.3f}\t{1:+8.3f}\t{2:8d}\t{3:7.1f}%\t" \
                        "{4:.3f}\t\t{5:.3f}". \
                        format(elem[0], elem[1], elem[2], elem[3], elem[4],
                               elem[5])
        _debug(peak_str)
    #
    #
    # set standard deviation before and after correction
    std_init = np.std(z)
    std = np.std(z / polyval2d(x, y, coeffs))
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
    wireframe = (*X, peak_target * polyval2d(*X, coeffs))
    #
    #
    # return coefficients for correction function
    print("Flight length correction took {0:.3f} seconds.".
          format(timer() - start))
    return coeffs.astype(_dtype), (x, y, z), events, wireframe
#
#
#
#
def get_mass_spectrum(data, spec_par, **kwargs):
    """Calculate mass spectrum.

    Calculate the mass-to-charge spectrum of the input *data* using the
    ``numpy.histogram()`` function.

    Parameters
    ----------
    data : ndarray, shape (n, 4)
        The *n* measured events, as described in
        :ref:`event data<apyt.spectrum.align:Event data>`.
    spec_par : tuple
        The physical parameters used to calculate the mass-to-charge spectrum,
        as described in :ref:`spectrum parameters
        <apyt.spectrum.align:Physical spectrum parameters>`.

    Keyword Arguments
    -----------------
    hist : dict
        The (optional) histogram parameters used to create the mass-to-charge
        histogram, as described in
        :ref:`histogram parameters<apyt.spectrum.align:Histogram parameters>`.

    Returns
    -------
    hist : ndarray, shape (m,)
        The *m* histogram counts.
    bin_centers : ndarray, shape (m,)
        The *m* bin centers of the histogram.
    mc_ratio: ndarray, shape (n,)
        The *n* mass-to-charge ratios for each event.
    """
    #
    #
    start = timer()
    _debug("Calculating mass spectrum for {0:.1f} M events…".
           format(len(data) / 1e6))
    #
    # get optional keyword arguments
    hist_par = kwargs.get('hist', {})
    #
    # get histogram parameters
    data_range = hist_par.get('range', None)
    if data_range is not None and data_range[0] >= data_range[1]:
        warnings.warn("Invalid data range detected (({0:.3f}, {1:.3}) amu/e). "
                      "Resetting to \"None\".".
                      format(data_range[0], data_range[1]))
        data_range = None
    width = hist_par.get('width', _default_bin_width)
    if width <= 0.0:
        warnings.warn("Histogram bin width must be positive. Resetting to "
                     "{0:.3f} amu/e.".format(_default_bin_width))
        width = _default_bin_width
    #
    #
    # calculate mass-to-charge ratio
    mc_ratio = _get_mass_to_charge_ratio(data, spec_par)
    #
    # set default histogram range if nothing provided
    if data_range is None:
        data_range = (np.floor(mc_ratio.min()), np.ceil(mc_ratio.max()))
    # set number of histogram bins (add one to account for extended range below)
    bins = int((data_range[1] - data_range[0]) / width) + 1
    #
    # calculate histogram and bin centers
    hist, bin_edges = np.histogram(mc_ratio, bins = bins,
        range = (data_range[0] - width / 2, data_range[1] + width / 2))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    #
    # return histogram, bin centers, and mass-to-charge ratios
    _debug("Calculation of mass spectrum took {0:.3f} seconds.".
           format(timer() - start))
    return hist, bin_centers, mc_ratio
#
#
#
#
def get_voltage_correction(data, spec_par, **kwargs):
    """Obtain coefficients for voltage correction.

    In order to perform the voltage correction, the full voltage range is first
    divided into several (equidistant) subranges and for each of the
    corresponding ranges, the position of the maximum peak is determined. By
    definition, the weighted average (with respect to the number of events in
    the respective range) of all peak positions is used as the peak target
    position for all voltage ranges. A 1d polynomial will then be used to map
    all determined peak positions to the peak target position, resulting in the
    voltage correction.

    Parameters
    ----------
    data : ndarray, shape (n, 4)
        The *n* measured events, as described in
        :ref:`event data<apyt.spectrum.align:Event data>`.
    spec_par : tuple
        The physical parameters used to calculate the mass-to-charge spectrum,
        as described in:ref:`spectrum parameters
        <apyt.spectrum.align:Physical spectrum parameters>`.

    Keyword Arguments
    -----------------
    deg : int
        The degree of the polynomial used for correction. Defaults to ``2``.
    hist : dict
        The (optional) histogram parameters used to create the mass-to-charge
        histogram, as described in
        :ref:`histogram parameters<apyt.spectrum.align:Histogram parameters>`.
    r_max : _np_float
        The optional radial range for filtering.
    size : float
        The minimum required size to take a certain voltage range into account
        for correction. The size is given relative to the number of events which
        would be expected for a homogeneous distribution. This option
        effectively filters voltage ranges with a low number of events. Defaults
        to ``0.3``.
    steps : int
        The number of steps into which the full voltage range is divided.
        Defaults to ``20``.
    thres : float
        The threshold used to determine peaks (relative to the maximum peak). If
        multiple peaks of similar height fall into the correction window,
        different (maximum) peaks may be picked up for different voltage ranges.
        By reducing the threshold for the peak detection, all of these peaks
        with similar height will be picked up which are above the specified
        threshold. In order to ensure consistent evaluation, always the first of
        these peaks is then chosen for the correction. Defaults to ``0.9``.

    Returns
    -------
    coeffs : ndarray, shape (deg + 1,)
        The polynomial coefficients obtained for correction (of type *float32*).
    (U, peak_pos) : tuple
        The tuple containing the data of the detected peak positions for every
        voltage range, each of type *ndarray* with *shape (m,)*.
    events : ndarray, shape (m,)
        The number of events in each voltage range. Can be used for
        color-coding.
    fit_data : tuple
        Smooth data for the peak position in dependence on the voltage, as
        obtained by the fit function (each of type *ndarray* with
        *shape (100,)*).
    """
    #
    #
    start = timer()
    print("Performing voltage correction…")
    #
    # get optional keyword arguments
    deg = kwargs.get('deg', 2)
    if deg < 0:
        warnings.warn("Polynomial degree must be positive. Resetting to \"2\".")
        deg = 2
    hist_par = kwargs.get('hist', {})
    r_max = kwargs.get('r_max', -1.0)
    size = kwargs.get('size', 0.3)
    if size < 0.0:
        warnings.warn("Size must not be negative. Resetting to \"0.3\".")
        size = 0.3
    steps = kwargs.get('steps', 20)
    if steps <= 0:
        warnings.warn("Steps must be positive. Resetting to \"20\".")
        steps = 20
    thres = kwargs.get('thres', 0.9)
    if thres < 0.0 or thres > 1.0:
        warnings.warn("Peak threshold must be between 0.0 and 1.0. Resetting "
                      "to \"0.9\".")
        thres = 0.9
    #
    #
    # filter radial detector range if requested
    if r_max > 0.0:
        data = __filter_radial_range(data, _np_float(r_max * r_max))
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
        data_cur = __filter_range(data, 0, _np_float(U_low), _np_float(U_high))
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
            peaks, _ = find_peaks(hist, height = thres * hist.max())
            #
            # check for valid peaks (might fail if we only capture the flank of
            # the highest peak, i.e. peak maximum is slightly out of range;
            # simply issue a warning and ignore this segment)
            if len(peaks) == 0:
                warnings.warn("No peaks detected for voltage range "
                              "({0:.1f}, {1:.1f})!".format(U_low, U_high))
                continue
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
    print("Voltage correction took {0:.3f} seconds.".format(timer() - start))
    return coeffs.astype(_dtype), (0.001 * x, y), events, xy_fit
#
#
#
#
def optimize_correction(data, spec_par, mode, **kwargs):
    """Automatically optimize correction coefficients.

    This function can be used to fine-tune the coefficients used for the voltage
    and flight length correction, respectively. The coefficients are varied
    systematically (using the Nelder--Mead algorithm for the |scipy_minimize|
    function from the *scipy.optimize* module) so that the width of the
    maximum peak in the spectrum reaches maximum possible sharpness.

    Parameters
    ----------
    data : ndarray, shape (n, 4)
        The *n* measured events, as described in
        :ref:`event data<apyt.spectrum.align:Event data>`.
    spec_par : tuple
        The physical parameters used to calculate the mass-to-charge spectrum,
        as described in:ref:`spectrum parameters
        <apyt.spectrum.align:Physical spectrum parameters>`.
    mode : str
        The string indicating which coefficients shall be optimized. Must be
        either ``voltage`` or ``flight``.

    Keyword Arguments
    -----------------
    hist : dict
        The (optional) histogram parameters used to create the mass-to-charge
        histogram, as described in
        :ref:`histogram parameters<apyt.spectrum.align:Histogram parameters>`.

    Returns
    -------
    coeffs : ndarray, shape of input array
        The optimized polynomial coefficients (of type *float32*).


    .. |scipy_minimize| raw:: html

        <a href="https://docs.scipy.org/doc/scipy/reference/generated/
        scipy.optimize.minimize.html" target="_blank">minimize()</a>
    """
    #
    #
    # check for valid correction coefficients
    if spec_par[2][0] is None or spec_par[2][1] is None:
        raise Exception("Correction coefficients have not been set.")
    #
    #
    # get optional keyword arguments
    hist_par = kwargs.get('hist', {})
    #
    #
    # filter range for faster processing
    if 'range' in hist_par:
        data = _filter_mass_to_charge_range(data, spec_par, hist_par)
    #
    #
    if mode == 'voltage':
        if len(spec_par[2][0]) == 1:
            warnings.warn("Cannot perform voltage correction for constant "
                          "polynomial function of degree 0 due to insufficient "
                          "degrees of freedom. Skipping…")
            return spec_par[2][0]
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
def peak_align(peaks_init, peaks_final, voltage_coeffs, L_0, alpha,
               U_guess = 10e3):
    """Automatically align peak positions.

    After the coefficients for the voltage and flight correction length have
    been determined in order to obtain sharp peaks, the mass spectrum still
    needs to be aligned properly. In principle, there are two parameters to be
    determined: The scaling factor :math:`\\alpha` of the voltage-to-flight
    length ratio and the time offset :math:`t_0` (see :ref:`spectrum parameters
    <apyt.spectrum.align:Physical spectrum parameters>`. These parameters can be
    obtained by solving

    .. math::
        \\frac m q = \\alpha \\left(\\frac m q\\right)_0 \
                     \\left(1 - \\frac{t_0}{L_0} \
                                \\sqrt{\\frac{2 U_\\textnormal{fix}} \
                                             {\\left(\\frac m q\\right)_0} \
                                }\\right)^2

    for two known peaks, where :math:`\\left(\\frac m q\\right)_0` and
    :math:`\\frac m q` are the initial and peak target position, respectively,
    and :math:`U_\\textnormal{fix}` is the voltage fix point at which no voltage
    correction is applied by definition.

    Parameters
    ----------
    peaks_init : ndarray, shape (2,)
        The two initial peak positions (amu/e) **before** alignment.
    peaks_final : ndarray, shape (2,)
        The two final peak target positions (amu/e) **after** alignment.
    voltage_coeffs : ndarray, shape(n,)
        The voltage correction coefficients, as described in
        :ref:`spectrum parameters
        <apyt.spectrum.align:Physical spectrum parameters>`.
    L_0 : _np_float
        The nominal flight length.
    alpha : _np_float
        The current constant spectrum scaling factor, as described in
        :ref:`spectrum parameters
        <apyt.spectrum.align:Physical spectrum parameters>`.
        Required for iterative calls of this function if the current value is
        not equal to unity.
    U_guess : float, optional
        The initial guess for the voltage fix point (i.e. the voltage at which
        no correction is applied by definition). Defaults to 10 kV.

    Returns
    -------
    params : ndarray, shape (2,)
        The coefficients :math:`(t_0, \\alpha)` for the peak alignment.
    """
    #
    #
    # local helper function to be solved with optimize.fsolve()
    def _peak_align(x, peak_init, peak_final):
        return x[0] * peak_init * \
               (1.0 - x[1] * np.sqrt(2.0 * alpha * U_fix / peak_init) / \
                      L_0 * 1e-6)**2 - \
               peak_final
    #
    #
    # convert peak positions from amu/e to corresponding SI units
    print("Selected peaks (amu/e) for automatic adjustment:"
          "\n{0:.3f} --> {1:.3f}\n{2:.3f} --> {3:.3f}".
          format(peaks_init[0], peaks_final[0], peaks_init[1], peaks_final[1]))
    mc_ratio_unit = constants.value('atomic mass constant') / \
                    constants.value('elementary charge')
    peaks_init  *= mc_ratio_unit
    peaks_final *= mc_ratio_unit
    #
    #
    # get voltage fix point (i.e. voltage at which no correction is applied)
    U_fix = fsolve(lambda U: polyval(U, voltage_coeffs), [U_guess])[0]
    _debug("Voltage fix point is {0:.3f} kV.".format(U_fix * 1e-3))
    #
    #
    # return parameters for peak alignment
    params = fsolve(
        lambda x: [
            _peak_align(x, peaks_init[0], peaks_final[0]),
            _peak_align(x, peaks_init[1], peaks_final[1])
        ], [1.0, 1.0])
    print("Automatically determined parameters for peak adjustment:\n"
          "α:  {0:10.6f}\nt₀: {1:+10.6f} ns".format(*params))
    return params
#
#
#
#
################################################################################
#
# private module-level functions
#
################################################################################
@numba.njit('f4[:, :](f4[:, :], f4[:], f4, f4)', parallel = True)
def __filter_mass_to_charge_range(data, mc_ratio, min, max):
    """Filter data for specific mass-to-charge ratio range.

    Parameters
    ----------
    data : ndarray, shape (n, 4)
        The *n* measured events, as described in
        :ref:`event data<apyt.spectrum.align:Event data>`.
    mc_ratio : ndarray, shape (n,)
        The calculated mass-to-charge ratio for each event.
    min : float32
        The minimum mass-to-charge ratio used for filtering.
    max : float32
        The maximum mass-to-charge ratio used for filtering.

    Returns
    -------
    data_f : ndarray, shape (m, 4)
        The filtered data.
    """
    #
    #
    return data[(min <= mc_ratio) & (mc_ratio <= max)]
#
#
#
#
@numba.njit('f4[:, :](f4[:, :], f4)', parallel = True)
def __filter_radial_range(data, r_sqr):
    """Filter data for specific radial detector range.

    Parameters
    ----------
    data : ndarray, shape (n, 4)
        The *n* measured events, as described in
        :ref:`event data<apyt.spectrum.align:Event data>`.
    r_sqr : _np_float
        The squared maximum radius used for filtering.

    Returns
    -------
    data_f : ndarray, shape (m, 4)
        The filtered data.
    """
    #
    #
    return data[(data[:, 1]**2 + data[:, 2]**2 <= r_sqr)]
#
#
#
#
@numba.njit('f4[:, :](f4[:, :], i8, f4, f4)', parallel = True)
def __filter_range(data, col, min, max):
    """Filter data for specific range in column.

    Parameters
    ----------
    data : ndarray, shape (n, 4)
        The *n* measured events, as described in
        :ref:`event data<apyt.spectrum.align:Event data>`.
    col : int
        The column of the input data used for filtering.
    min : float32
        The minimum value used for filtering.
    max : float32
        The maximum value used for filtering.

    Returns
    -------
    data_f : ndarray, shape (m, 4)
        The filtered data.
    """
    #
    #
    return data[(min < data[:, col]) & (data[:, col] <= max)]
#
#
#
#
@numba.njit('f4[:](f4[:, :], f4[:], f4, f4, f4)', parallel = True)
def __get_mass_to_charge_ratio(data, U, t_0, L_0, alpha):
    """Calculate mass-to-charge ratio for every event.

    Parameters
    ----------
    data : ndarray, shape (n, 4)
        The *n* measured events, as described in
        :ref:`event data<apyt.spectrum.align:Event data>`.
    U : ndarray, shape (n,)
        The voltage for every event (with possible correction).
    t_0 : float32
        The time-of-flight offset.
    L_0 : float32
        The (nominal) distance between tip and detector.
    alpha : float32
        The scaling factor of the mass-to-charge spectrum, as described in
        :ref:`spectrum parameters
        <apyt.spectrum.align:Physical spectrum parameters>`.

    Returns
    -------
    mc_ratio : ndarray, shape (n,)
        The mass-to-charge ratio for every event.
    """
    #
    #
    alpha *= _mc_conversion_factor
    return U * (data[:, 3] - t_0)**2 / \
           (L_0**2 + data[:, 1]**2 + data[:, 2]**2) * alpha
#
#
#
#
def _debug(msg):
    """Print debug message to *stderr*.

    Parameters
    ----------
    msg : str
        The message to be written.
    """
    #
    #
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
    """Wrapper function to filter mass-to-charge ratio range.

    Parameters
    ----------
    data : ndarray, shape (n, 4)
        The *n* measured events, as described in
        :ref:`event data<apyt.spectrum.align:Event data>`.
    spec_par : tuple
        The physical parameters used to calculate the mass-to-charge spectrum,
        as described in :ref:`spectrum parameters
        <apyt.spectrum.align:Physical spectrum parameters>`.
    hist_par : dict
        The (optional) histogram parameters used to create the mass-to-charge
        histogram, as described in
        :ref:`histogram parameters<apyt.spectrum.align:Histogram parameters>`.

    Returns
    -------
    data_f : ndarray, shape (m, 4)
        The filtered data.
    """
    #
    #
    # get range from histogram parameter dictionary
    data_range = hist_par.get('range', None)
    if data_range is not None and data_range[0] >= data_range[1]:
        warnings.warn("Invalid data range detected (({0:.3f}, {1:.3}) amu/e). "
                      "Resetting to \"None\".".
                      format(data_range[0], data_range[1]))
        data_range = None
    #
    #
    # calculate histogram and mass-to-charge ratio
    hist, bin_centers, mc_ratio = get_mass_spectrum(
        data, spec_par, hist = hist_par)
    #
    #
    # if no range provided, use fixed width of 20 around maximum peak
    if data_range is None:
        _debug("Auto-detecting range…")
        peaks, _ = find_peaks(hist, distance = np.iinfo(np.int32).max)
        _debug("Maximum peak is at {0:.1f} amu/e.".
               format(bin_centers[peaks[0]]))
        data_range = (bin_centers[peaks[0]] - 10, bin_centers[peaks[0]] + 10)
    #
    #
    # filter range
    data = __filter_mass_to_charge_range(
        data, mc_ratio, _np_float(data_range[0]), _np_float(data_range[1]))
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
def _get_mass_to_charge_ratio(data, spec_par):
    """Wrapper function to calculate mass-to-charge ratio.

    Parameters
    ----------
    data : ndarray, shape (n, 4)
        The *n* measured events, as described in
        :ref:`event data<apyt.spectrum.align:Event data>`.
    spec_par : tuple
        The physical parameters used to calculate the mass-to-charge spectrum,
        as described in :ref:`spectrum parameters
        <apyt.spectrum.align:Physical spectrum parameters>`.

    Returns
    -------
    mc_ratio : ndarray, shape (n,)
        The mass-to-charge ratio for every event.
    """
    #
    #
    # unpack parameters for better readability
    t_0, L_0, (voltage_coeffs, flight_coeffs), alpha = spec_par
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
    mc_ratio = __get_mass_to_charge_ratio(data, U, t_0, L_0, alpha)
    #
    #
    # apply flight length correction if provided
    if flight_coeffs is not None:
        if flight_coeffs.dtype is not np.dtype(_dtype):
            raise TypeError("Wrong type for flight length coefficients "
                            "({0:s}). Must be '{1:s}'.".format(
                                str(flight_coeffs.dtype),
                                str(np.dtype(_dtype))))
        mc_ratio /= polyval2d(data[:, 1], data[:, 2], flight_coeffs)
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
    "Print current and peak memory usage to *stderr*."
    #
    #
    # resource module may not be available on all platforms
    try:
        from resource import getrusage, RUSAGE_SELF
    except:
        return
    #
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
    """Optimize coefficients for flight length correction.

    Parameters
    ----------
    data : ndarray, shape (n, 4)
        The *n* measured events, as described in
        :ref:`event data<apyt.spectrum.align:Event data>`.
    spec_par : tuple
        The physical parameters used to calculate the mass-to-charge spectrum,
        as described in :ref:`spectrum parameters
        <apyt.spectrum.align:Physical spectrum parameters>`.
    hist_par : dict
        The (optional) histogram parameters used to create the mass-to-charge
        histogram, as described in
        :ref:`histogram parameters<apyt.spectrum.align:Histogram parameters>`.

    Returns
    -------
    coeffs : ndarray, shape of input array
        The optimized coefficients for the flight length correction (of type
        *float32*).
    """
    #
    #
    start = timer()
    print("Optimizing flight length correction…")
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
                (voltage_coeffs, flight_coeffs[0]), spec_par[3], hist_par,
                'flight'),
        method = 'nelder-mead',
        options = {'fatol': 1e-2, 'disp': _is_dbg, 'maxiter': 100})
    #
    #
    # re-assemble coefficients for flight length correction
    flight_coeffs = np.append(flight_coeffs[0],
                              minimization_result.x.astype(_dtype))
    flight_coeffs = _poly2d_coeff_vec_to_mat(flight_coeffs)
    #
    #
    # get peak position and width for optimized coefficients
    peak_pos_final, peak_width_final = _peak_width(data,
        (spec_par[0], spec_par[1], (voltage_coeffs, flight_coeffs),
         spec_par[3]), hist_par)
    _debug("Final peak position is at {0:.3f} amu/e (width {1:.3f} amu/e).".
           format(peak_pos_final, peak_width_final))
    _debug("Optimized coefficients for flight length correction are {0:s}.".
           format(str(flight_coeffs)))
    #
    #
    # return optimized coefficients for flight length correction
    print("Optimization of flight length correction took {0:.3f} seconds.".
          format(timer() - start))
    print("Final peak width is {0:.3f} amu/u (initial: {1:.3f} amu/e).".
          format(peak_width_final, peak_width_init))
    return flight_coeffs.astype(_dtype)
#
#
#
#
def _optimize_voltage_correction(data, spec_par, hist_par):
    """Optimize coefficients for voltage correction.

    Parameters
    ----------
    data : ndarray, shape (n, 4)
        The *n* measured events, as described in
        :ref:`event data<apyt.spectrum.align:Event data>`.
    spec_par : tuple
        The physical parameters used to calculate the mass-to-charge spectrum,
        as described in :ref:`spectrum parameters
        <apyt.spectrum.align:Physical spectrum parameters>`.
    hist_par : dict
        The (optional) histogram parameters used to create the mass-to-charge
        histogram, as described in
        :ref:`histogram parameters<apyt.spectrum.align:Histogram parameters>`.

    Returns
    -------
    coeffs : ndarray, shape of input array
        The optimized coefficients for the voltage correction (of type
        *float32*).
    """
    #
    #
    start = timer()
    print("Optimizing voltage correction…")
    #
    #
    # fine correction of the voltage coefficients may lead to a drastic drift of
    # the spectrum in rare cases due to numerical instabilities, so we unset the
    # histogram range here and default to the intrinsic data range so that all
    # peaks can still be detected in the spectrum
    if 'range' in hist_par:
        hist_par['range'] = None
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
                (voltage_coeffs[0], flight_coeffs), spec_par[3], hist_par,
                'voltage', {'peak_target': peak_pos_init}),
        method = 'nelder-mead',
        options = {'fatol': 1e-2, 'disp': _is_dbg, 'maxiter': 100})
    #
    #
    # re-assemble coefficients for voltage correction
    voltage_coeffs = np.append(voltage_coeffs[0],
                               minimization_result.x.astype(_dtype))
    #
    #
    # get peak position and width for optimized coefficients
    peak_pos_final, peak_width_final = _peak_width(data,
        (spec_par[0], spec_par[1], (voltage_coeffs, flight_coeffs),
         spec_par[3]), hist_par)
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
    print("Optimization of voltage correction took {0:.3f} seconds.".
          format(timer() - start))
    print("Final peak width is {0:.3f} amu/u (initial: {1:.3f} amu/e).".
          format(peak_width_final, peak_width_init))
    return voltage_coeffs.astype(_dtype)
#
#
#
#
def _peak_width(data, spec_par, hist_par):
    """Get position and width of maximum peak.

    Parameters
    ----------
    data : ndarray, shape (n, 4)
        The *n* measured events, as described in
        :ref:`event data<apyt.spectrum.align:Event data>`.
    spec_par : tuple
        The physical parameters used to calculate the mass-to-charge spectrum,
        as described in :ref:`spectrum parameters
        <apyt.spectrum.align:Physical spectrum parameters>`.
    hist_par : dict
        The (optional) histogram parameters used to create the mass-to-charge
        histogram, as described in
        :ref:`histogram parameters<apyt.spectrum.align:Histogram parameters>`.

    Returns
    -------
    pos : float
        The position of the maximum peak.
    width : float
        The width of the maximum peak.
    """
    #
    #
    # check for valid histogram bin width
    width = hist_par.get('width', _default_bin_width)
    if width <= 0.0:
        warnings.warn("Histogram bin width must be positive. Resetting to "
                     "{0:.3f} amu/e.".format(_default_bin_width))
        width = _default_bin_width
    #
    #
    # calculate histogram and bin centers
    hist, bin_centers, _ = get_mass_spectrum(data, spec_par, hist = hist_par)
    #
    # get maximum peak and its width
    peaks, _   = find_peaks(hist, distance = np.iinfo(np.int32).max)
    width_half = peak_widths(hist, peaks, rel_height = 0.5)[0][0]
    #
    # return peak positions and widths
    return bin_centers[peaks[0]], width_half * width
#
#
#
#
def _peak_width_minimizer(x, data, t_0, L_0, coeffs_stripped, alpha, hist_par,
                          mode, dict_args = None):
    """The minimizer function for the calculation of the peak width.

    Parameters
    ----------
    x : ndarray, shape (k,)
        The correction coefficients to vary (excluding the absolute
        coefficient).
    data : ndarray, shape (n, 4)
        The *n* measured events, as described in
        :ref:`event data<apyt.spectrum.align:Event data>`.
    t_0 : float32
        The time-of-flight offset.
    L_0 : float32
        The (nominal) distance between tip and detector.
    coeffs_stripped : tuple
        The voltage and flight length correction coefficients in stripped form,
        i.e. excluding the coefficients already given in *x*.
    alpha : float32
        The scaling factor of the mass-to-charge spectrum, as described in
        :ref:`spectrum parameters
        <apyt.spectrum.align:Physical spectrum parameters>`.
    hist_par : dict
        The (optional) histogram parameters used to create the mass-to-charge
        histogram, as described in
        :ref:`histogram parameters<apyt.spectrum.align:Histogram parameters>`.
    mode : str
        The string indicating which coefficients shall be optimized. Must be
        either ``voltage`` or ``flight``.
    dict_args : dict
        The optional dictionary arguments passed to this minimizer function,
        consisting of the peak target position (``'peak_target'``).

    Returns
    -------
    width : float
        The width of the maximum peak.
    """
    #
    #
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
    peak_pos, peak_width = _peak_width(
        data, (t_0, L_0, coeffs, alpha), hist_par)
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
def _poly2d_coeff_mat_to_vec(M):
    """Convert 2d coefficient matrix to sparse vector representation.

    Note that only polynomial terms :math:`x^i y^j` with :math:`i + j \\leq d`
    are used, i.e. the 2d coefficient matrix contains zeros in the lower right
    corner.

    Parameters
    ----------
    M : ndarray, shape (n, n)
        The 2d coefficient matrix.

    Returns
    -------
    v : ndarray, shape (n * (n + 1) / 2,)
        The sparse vector representation of the 2d coefficient matrix.
    """
    #
    #
    # get degree from matrix
    deg = M.shape[0] - 1
    #
    # convert full coefficient matrix to vector
    v = M.reshape(-1)
    #
    # create mask to filter higher-order terms
    mask = np.rot90(np.tri(deg + 1, dtype = bool), k = -1).reshape(-1)
    #
    # return sparse coefficient vector
    return v[mask]
#
#
#
#
def _poly2d_coeff_vec_to_mat(v):
    """Convert sparse vector representation to 2d coefficient matrix.

    Note that only polynomial terms :math:`x^i y^j` with :math:`i + j \\leq d`
    are used, i.e. the returned 2d coefficient matrix will contain zeros in the
    lower right corner.

    Parameters
    ----------
    v : ndarray, shape (n * (n + 1) / 2,)
        The sparse vector representation of the 2d coefficient matrix.

    Returns
    -------
    M : ndarray, shape (n, n)
        The 2d coefficient matrix.
    """
    #
    #
    # determine degree from length of vector representation
    deg = np.rint(-3.0 / 2.0 + np.sqrt(1.0 / 4.0 + 2 * len(v))).astype(int)
    #
    # initialize coefficient matrix
    M = np.zeros((deg + 1, deg + 1), dtype = v.dtype)
    #
    # set coefficients from sparse coefficient vector
    M[np.mask_indices(deg + 1, lambda m, k : np.rot90(np.triu(m, k)))] = v
    #
    # return full coefficient matrix
    return M
#
#
#
#
def _polyfit2d(x, y, f, deg, **kwargs):
    """Custom 2d fitting routine to allow for optional weights.

    Note that only polynomial terms :math:`x^i y^j` with :math:`i + j \\leq d`
    are used, i.e. the returned 2d coefficient matrix will contain zeros in the
    lower right corner.

    Parameters
    ----------
    x : ndarray, shape (n,)
        The function arguments in the first dimension.
    y : ndarray, shape (n,)
        The function arguments in the second dimension.
    f : ndarray, shape (n,)
        The function values used for fitting.
    deg : int
        The polynomial degree used for fitting.

    Keyword Arguments
    -----------------
    weights : ndarray, shape (n,)
        The optional weights used for fitting. Default to ``1.0``.
    offset : float
        The optional fixed offset for the fit function (effectively keeps the
        absolute term fixed at that value and reduces the degrees of freedom).
        Defaults to ``None``.

    Returns
    -------
    coeffs : ndarray, shape (deg + 1, deg + 1)
        The coefficients obtained through fitting.
    """
    #
    #
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
