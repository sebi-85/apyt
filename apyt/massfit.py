"""
The APyT mass spectrum fitting module
=====================================

This module enables semi-automatic fitting of high-quality mass spectra which
have been obtained before from the
:ref:`APyT mass spectrum module<apyt.massspec:The APyT mass spectrum module>`.
This module makes intensive use of the isotope database provided by the
|periodictable| package.


Howto
-----

The usage of this module is demonstrated in an auxiliary script
(``wrapper_scripts/apyt_massfit.py``) which basically serves as a wrapper for
this module. Detailed usage information can be obtained by invoking this script
with the ``"--help"`` option.


General peak shape parameter description
----------------------------------------

The general peak shape :math:`p(x)` consists of an
:ref:`exponential decay<apyt.massfit:Decay function>` function :math:`d(x)` and
an :ref:`activation<apyt.massfit:Activation function>` function :math:`a(x)`,
which describes the onset of the peak:

.. math::
    p(x) = a(x) d(x).

Decay function
^^^^^^^^^^^^^^

The exponential decay is simply modeled through

.. math::
    d(x) = \\exp(-x).

Activation function
^^^^^^^^^^^^^^^^^^^

The onset of the peak is modeled through a sigmoid-type error function of the
form

.. math::
    a(x) = \\frac 1 2 \\left(
        \\operatorname{erf}\\left(\\frac{\\sqrt \\pi}{2} x\\right) + 1
        \\right),

where the prefactor :math:`\\frac{\\sqrt \\pi}{2}` ensures that the peak
position is at zero, i.e. :math:`p'(0) = 0`.

Implementation notes
^^^^^^^^^^^^^^^^^^^^

Fitting a complete spectrum requires a sum of peaks as described above, in
principle one for each element, isotope, and charge state combination, each of
these with a unique intensity parameter. However, the ratios of the intensity
parameters are pre-determined from the natural abundances of the elements,
reducing the number of independent fitting parameters, effectively resulting in
only one intensity parameter per isotope peak group.

Also, the peak shape may be more complex so that the tailing decay may only be
described by e.g. a sum of two exponential decay functions, i.e. using two decay
constants. The general peak shape function to be used can be passed to several
methods (cf. :ref:`List of methods<apyt.massfit:List of methods>`) and is
identified through one of the following strings:

* ``error-expDecay``: Error function activation with exponential decay.
* ``error-doubleExpDecay``: Error function activation with double exponential
  decay.

If a new general peak shape function shall be implemented, the method
:meth:`_peak_generic` needs to be extended accordingly. (Note that this method
is for internal use only and is not exposed by this module.)

Data type description
---------------------

This modules relies mainly on dictionaries for the input/output parameter
interface which are described in the following.

Element dictionary
^^^^^^^^^^^^^^^^^^

The element dictionary consists of key--value pairs, where each key represents
one element (may also be a molecule) and the value is a (nested) tuple
containing the occurring charge states of the respective element (also as a
tuple) and the corresponding nominal volume (in nm³) of the element/molecule
used for reconstruction.

Peak dictionary
^^^^^^^^^^^^^^^

The peak dictionary consists of the following keys:

* ``element``: The associated element name (may also be a molecule).
* ``charge``: The associated charge state.
* ``mass_charge_ratio``: The mass-to-charge ratio for this peak.
* ``abundance``: The isotope abundance (as a decimal fraction).
* ``is_max``: Whether this peak has maximum abundance for the associated
  element.
* ``volume``: The atomic volume (in nm³) of the associated element required for
  reconstruction.

Element count dictionary
^^^^^^^^^^^^^^^^^^^^^^^^

The element count dictionary consists of the following keys:

* ``element``: The element name (may also be a molecule).
* ``charge``: The charge state.
* ``count``: The number of counts for this element.
* ``fraction``: The fraction of the counts for this element in relation to the
  total counts (without background).


List of methods
---------------

This module provides some generic functions for the fitting of mass spectra
from histogram data.

The following methods are provided:

* :meth:`counts`: Get counts for all elements.
* :meth:`enable_debug`: Enable or disable debug output.
* :meth:`fit`: Fit mass spectrum.
* :meth:`map_ids`: Map mass-to-charge ratios to chemical IDs.
* :meth:`peaks_list`: Get list of all peaks for specified elements and
  charge states.
* :meth:`spectrum`: Calculate spectrum for specified list of elements.
* :meth:`split_molecules`: Split molecular events into individual atoms.


.. |periodictable| raw:: html

        <a href="https://periodictable.readthedocs.io/en/latest/"
        target="_blank">periodic table</a>


.. sectionauthor:: Sebastian M. Eich <Sebastian.Eich@imw.uni-stuttgart.de>
.. moduleauthor::  Sebastian M. Eich <Sebastian.Eich@imw.uni-stuttgart.de>

"""
#
#
#
#
__version__ = '0.1.0'
__all__ = [
    'counts',
    'enable_debug',
    'fit',
    'map_ids',
    'peaks_list',
    'spectrum',
    'split_molecules'
]
#
#
#
#
# import modules
import itertools
import lmfit
import numba
import numpy as np
import periodictable
import re
import warnings
#
# import some special functions/modules
from scipy.signal import find_peaks
from scipy.special import erf
from scipy.stats import multinomial
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
_abundance_thres = 1e-9
"The abundance threshold for discarding isotopes."
_is_dbg = False
"""The global flag for debug output.

This flag can be set through the :meth:`enable_debug` function."""
#
#
#
#
################################################################################
#
# module-level initialization
#
################################################################################
def _add_element(name, symbol, number, table, isotopes, density):
    """
    Add new element to periodic table.
    """
    #
    #
    # create new element
    element = table.core.Element(name, symbol, number, None, table)
    element._density = density
    #
    # add isotopes to element
    for isotope in isotopes:
        element.add_isotope(isotope[0])
        element[isotope[0]]._mass      = isotope[1]
        element[isotope[0]]._abundance = isotope[2]
    #
    #
    # add new element to table (append 'X' to indicate custom element)
    table.elements._element["{0:d}X".format(number)] = element
    setattr(table,          symbol, element)
    setattr(table.elements, symbol, element)
#
#
# add gallium used by FIB (only one isotope!)
_add_element(
    "gallium (FIB)", "Gx", 31, periodictable,
    [(69, periodictable.Ga[69].mass, 100.0)], periodictable.Ga.density
)
#
#
#
#
################################################################################
#
# public functions
#
################################################################################
def counts(peaks_list, function, params, data_range, bin_width,
           ignore_list = [], verbose = False):
    """
    Get counts for all elements.

    This functions loops trough all peaks in *peaks_list* with the ``is_max``
    key set to ``True`` and returns the counts for each element and charge state
    combination.

    Parameters
    ----------
    peaks_list : list of dicts
        The list of all occurring peaks in the mass spectrum, as described in
        :ref:`peak dictionary<apyt.massfit:Peak dictionary>`.
    function : str
        The string identifying the general peak shape function, as described in
        :ref:`implementation notes<apyt.massfit:Implementation notes>`.
    params : dict
        The dictionary with the fit parameter names as keys, and best-fit values
        as values, as described in the |best_params| |model_result| attribute of
        the |lmfit| module.
    data_range : float
        The covered data range of the spectrum, i.e. the difference between
        maximum and minimum.
    bin_width : float
        The histogram bin width.

    Keyword Arguments
    -----------------
    ignore_list : list of str
        The list of elements/molecules to ignore when calculating compositions.
        Defaults to empty list.
    verbose : bool
        Whether to print the content of all element count dictionaries. Defaults
        to ``False``.

    Returns
    -------
    counts_list : list of dicts
        The list of all element count dictionaries, as described in
        :ref:`element count dictionary<apyt.massfit:Element count dictionary>`.
    total_counts : int
        The total number of counts (without background).
    background : int
        The number of background counts.


    .. |best_params| raw:: html

        <a href="https://lmfit.github.io/lmfit-py/model.html
        #lmfit.model.best_values" target="_blank">
        best_values</a>


    .. |model_result| raw:: html

        <a href="https://lmfit.github.io/lmfit-py/model.html
        #modelresult-attributes" target="_blank">
        ModelResult</a>


    .. |lmfit| raw:: html

        <a href="https://lmfit.github.io/lmfit-py/" target="_blank">
        LMFIT</a>
    """
    #
    #
    # get unified area for count normalization
    area = _peak_generic(function, params, 'count', None)
    #
    #
    # loop through all peaks
    counts_list = []
    total_counts = 0.0
    for peak in peaks_list:
        # calculate element count only from peak with maximum abundance
        if peak['is_max'] == False:
            continue
        #
        # get intensity parameter for current peak
        I = params[_get_intensity_name(peak)]
        #
        # calculate count
        count = I * area / bin_width
        #
        # cumulate total counts
        total_counts += count
        #
        # append count dictionary to count list
        counts_list.append({
            'element':  peak['element'],
            'charge':   peak['charge'],
            'count':    count,
            'fraction': 0.0
            })
    #
    # add fractions
    for count in counts_list:
        count['fraction'] = count['count'] / total_counts
    #
    #
    # calculate background
    background = params['base'] * data_range / bin_width
    #
    #
    # print element counts if requested
    if verbose == True:
        print("Total number of counts (without background) is {0:.0f}:".
              format(total_counts))
        print("Number of background counts is {0:.0f} ({1:.1f}%).\n".format(
              background, background / (total_counts + background) * 100))
        print("element\t\tcharge\t   count\tfraction")
        print("--------" * 5 + "--------")
        for count in counts_list:
            print("{0:12s}\t{1:d}\t{2:8.0f}\t{3:.4f}".
                  format(*count.values()))
        print("========" * 5 + "========")
        print("\t\ttotal\t{0:8.0f}\n".format(total_counts))
        #
        #
        # combine counts for different charge states for individual elements
        print("element\t\t   count\tfraction")
        print("--------" * 4 + "--------")
        element_count = 0
        element       = counts_list[0]['element']
        for count in counts_list:
            if count['element'] == element:
                element_count += count['count']
            else:
                # print previous element and reset
                print("{0:12s}\t{1:8.0f}\t{2:.4f}".format(
                      element, element_count, element_count / total_counts))
                element_count = count['count']
                element       = count['element']
        # print last element
        print("{0:12s}\t{1:8.0f}\t{2:.4f}".
              format(element, element_count, element_count / total_counts))
        print("========" * 4 + "========")
        print("total\t\t{0:8.0f}\n".format(total_counts))
        #
        #
        # break down molecules into individual elements
        count_dict = {}
        is_molecule = False
        for count in counts_list:
            # check whether element is in ignore list
            if count['element'] in ignore_list:
                if _is_dbg:
                    print("Ignoring element \"{0:s}\".".
                          format(count['element']))
                continue
            #
            #
            # get molecule items
            molecule_items = \
                periodictable.formula(count['element']).atoms.items()
            #
            # check whether multiple elements present
            if len(molecule_items) > 1:
                is_molecule = True
            #
            # loop through elements in molecule
            for element, element_count in molecule_items:
                # check whether multiple atoms are present for individual
                # element
                if element_count > 1:
                    is_molecule = True
                #
                # create new dictionary key-value pair if element not found,
                # increment otherwise
                if element.symbol not in count_dict:
                    count_dict[element.symbol]  = element_count * count['count']
                else:
                    count_dict[element.symbol] += element_count * count['count']
        #
        # only print if molecule found or ignore list active
        if is_molecule == True or len(ignore_list) > 0:
            print("element\t    count\tfraction")
            print("--------" * 3 + "--------")
            for element, element_count in count_dict.items():
                print("{0:s}\t{1:9.0f}\t{2:.4f}".
                      format(element, element_count,
                             element_count / sum(count_dict.values())))
            print("========" * 3 + "========")
            print("total\t{0:9.0f}\n".format(sum(count_dict.values())))
    #
    #
    # return element counts, total counts, and background counts
    return counts_list, total_counts, background
#
#
#
#
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
def fit(spectrum, peaks_list, function, verbose = False, **kwargs):
    """
    Fit mass spectrum.

    This function internally uses the |lmfit| module to fit the complete mass
    spectrum, where the peak positions are provided by the *peaks_list*
    argument.

    Parameters
    ----------
    spectrum : ndarray, shape (n, 2)
        The mass spectrum histogram data.
    peaks_list : list of dicts
        The list of all occurring peaks in the mass spectrum, as described in
        :ref:`peak dictionary<apyt.massfit:Peak dictionary>`.
    function : str
        The string identifying the general peak shape function, as described in
        :ref:`implementation notes<apyt.massfit:Implementation notes>`.

    Keyword Arguments
    -----------------
    oxygen_shift : bool
        Whether to use an additional shift for the (pure) oxygen peaks. The
        parameter is implemented as an absolute shift which scales with the
        square root of the peak position and applies to all peaks matching the
        regular expression ``^I_O[0-9]*_[0-9]+$``. This shift was first
        discovered in the vanadium pentoxide measurements of Simone Bauder in
        2024. Defaults to ``False``, i.e. do not shift oxygen peaks, which
        resembles a fixed value of ``0.0``.
    scale_width : bool
        Whether to use a varying parameter for the peak width scaling.
        Theoretically, the peak width in the mass-to-charge scale is expected to
        be proportional to the square root of the peak position, but may show
        different behavior. The parameter is implemented as an exponent.
        Defaults to ``False``, i.e. assume square root behavior, which resembles
        a fixed exponent value of ``0.5``.
    verbose : bool
        Whether to print the fit results and statistics. Defaults to
        ``False``.

    Returns
    -------
    result : ModelResult
        The result of the fit, as described by |model_result| of the |lmfit|
        module.
    """
    #
    #
    # define fit model
    start = timer()
    model = lmfit.Model(_model_spectrum, independent_vars = ["x"])
    #
    # estimate fit parameters
    parameters = _estimate_fit_parameters(
        spectrum, peaks_list, function, **kwargs
    )
    #
    #
    # perform fit and print results
    print("Fitting mass spectrum using \"{0:s}\"…".format(function))
    with warnings.catch_warnings():
        warnings.filterwarnings(
            'ignore',
            message = "The keyword argument (function|peaks_list) does not " +
            "match any arguments of the model function. It will be ignored.",
            category = UserWarning)
        result = model.fit(spectrum[:, 1], parameters, x = spectrum[:, 0],
                           peaks_list = peaks_list, function = function)
    if verbose == True:
        print(result.fit_report(show_correl = False))
        print("Fitting of mass spectrum took {0:.3f}s.".format(timer() - start))
        print("")
    #
    #
    # return fit results
    return result
#
#
#
#
def map_ids(mc_ratio, r, x, peaks_list, function, params, group_charge = True,
            verbose = False):
    """
    Map mass-to-charge ratios to chemical IDs.

    This function calculates a probability vector which contains the
    probabilities to find a specific element (with individual charge state if
    requested) associated with the given peaks in *peaks_list* (including
    background) at every position *x* in the mass-to-charge spectrum. These
    probabilities are eventually used for peak de-convolution and background
    subtraction to assign the corresponding chemical IDs. The atomic volumes of
    the events are also mapped and returned.

    Parameters
    ----------
    mc_ratio : ndarray, shape (n,)
        The mass-to-charge ratios of the *n* events.
    r : ndarray, shape (n,)
        The *n* random numbers (between ``0.0`` and ``1.0``) required for
        mapping the probability vector to an actual chemical ID.
    x : ndarray, shape (m,)
        The positions at which the probability vectors are calculated. The
        elements of the array must be equidistant and would typically represent
        the histogram bin centers.
    peaks_list : list of dicts
        The list of all occurring peaks in the mass spectrum, as described in
        :ref:`peak dictionary<apyt.massfit:Peak dictionary>`.
    function : str
        The string identifying the general peak shape function, as described in
        :ref:`implementation notes<apyt.massfit:Implementation notes>`.
    params : dict
        The dictionary with the fit parameter names as keys, and best-fit values
        as values, as described in the |best_params| |model_result| attribute of
        the |lmfit| module.
    group_charge : bool
        Whether to group all charge states of an individual element. Defaults to
        ``True``.
    verbose : bool
        Whether to print a list of all chemical IDs and their fractions in
        relation to the total counts (with background). Defaults to ``False``.

    Returns
    -------
    ids : ndarray, shape (n,)
        The chemical IDs of the *n* events.
    Ω : ndarray, shape (n,)
        The atomic volumes of the *n* events.
    """
    #
    #
    @numba.njit("i2[:](f4[:], f8[:,:], f8[:])", cache = True, parallel = True)
    def _get_ids(bin_ids, p, r):
        """
        Simple helper function which performs the actual chemical mapping from
        the probability vectors to an ID.

        The probability vectors are linearly interpolated based on the
        fractional bin ID of each event in the mass spectrum.
        """
        #
        #
        # get integral and fractional part required for interpolation
        integ = np.floor(bin_ids)
        frac  = bin_ids - integ
        integ = integ.astype(np.int32)
        #
        #
        # loop through all events
        ids = np.empty(len(bin_ids), dtype = np.int16)
        for i in numba.prange(len(bin_ids)):
            # interpolate probability vector
            p_i = (1.0 - frac[i]) * p[integ[i]] + frac[i] * p[integ[i] + 1]
            #
            # map probability vector to chemical ID
            ids[i] = np.searchsorted(p_i, r[i])
        #
        #
        # return chemical IDs
        return ids
    #
    #
    #
    #
    # start timer
    print("Performing chemical mapping…")
    start = timer()
    #
    #
    # add one grid point at either end to allow interpolation at margin
    Δx = (x[-1] - x[0]) / (len(x) - 1)
    x  = np.concatenate(([x[0] - Δx], x, [x[-1] + Δx]))
    #
    #
    # group peaks for the same element (and charge state if requested)
    peak_groups = _group_peaks(peaks_list, group_charge)
    #
    #
    # set background counts (always *first* entry)
    p_vec = np.zeros((len(x), len(peak_groups) + 1))
    p_vec[:, 0] = params['base']
    #
    # loop through peak groups to calculate probability vectors (one vector for
    # each x-position containing the probabilities for every peak group
    for i in range(len(peak_groups)):
        p_vec[:, i + 1] = spectrum(x, peak_groups[i], function, params)
    #
    #
    # normalize probability vector
    p_vec /= np.sum(p_vec, axis = 1)[:, np.newaxis]
    #
    # calculate cumulated probability vector
    p_vec = np.cumsum(p_vec, axis = 1)
    #
    #
    # get chemical IDs of each event
    # (map mass-to-charge ratios to fractional bin IDs)
    ids = _get_ids((mc_ratio - x[0]) / Δx, p_vec, r)
    #
    #
    # print counts of all IDs if requested
    if verbose == True:
        # get counts for all IDs
        counts = np.bincount(ids, minlength = p_vec.shape[1])
        #
        #
        print("\nid\telement\t\tcharge\t   count\tfraction\tvolume")
        print("--------" * 8 + "--------")
        #
        # background counts (always first entry)
        print("{0:3d}\tBackground\t\t{1:8d}\t{2:.4f}".
            format(0, counts[0], counts[0] / len(mc_ratio)))
        #
        # loop through all peaks
        for i, group in zip(range(1, len(peak_groups) + 1), peak_groups):
            print(
                "{0:3d}\t{1:12s}\t{2:d}\t{3:8d}\t{4:.4f}\t\t{5:.6f}".
                format(
                    i, group[0]['element'],
                    0 if group_charge == True else group[0]['charge'],
                    counts[i], counts[i] / len(ids), group[0]['volume']
                )
            )
        #
        print("========" * 8 + "========")
        print("\t\t\ttotal\t{0:8d}\n".format(len(mc_ratio)))
    #
    #
    # map atomic volumes (prepend zero volume for background events)
    Ω = np.append(0.0, [group[0]['volume'] for group in peak_groups])[ids]
    #
    #
    # return chemical IDs and atomic volumes
    print("Chemical mapping took {0:.3f}s.".format(timer() - start))
    return ids, Ω
#
#
#
#
def peaks_list(element_dict, mode = 'mass', mass_decimals = 3, verbose = False):
    """
    Get list of all peaks for specified elements and charge states.

    Parameters
    ----------
    element_dict : dict
        The dictionary containing the elements and their charge states, as
        described in :ref:`element dictionary<apyt.massfit:Element dictionary>`.

    Keyword Arguments
    -----------------
    mode : str
        Which mode to use for the calculation of the mass-to-charge ratios. In
        ``mass`` mode (recommended), the actual isotopic masses are used, while
        in ``isotope`` mode, the mass numbers of the isotopes are used. Defaults
        to ``mass``.
    mass_decimals : int
        The number of decimal places for the mass-to-charge ratios in ``mass``
        mode should be limited to reduce the number of molecular isotopes. This
        setting effectively groups isotopes whose masses do not differ by more
        than the specified *mass_decimals*. This value should be set based on
        the resolution of the mass spectrum. Defaults to ``3``, i.e. group
        isotopes whose masses do not differ by more than ``0.001``.
    verbose : bool
        Whether to print the content of all determined peak dictionaries.
        Defaults to ``False``.

    Returns
    -------
    peaks_list : list of dicts
        The list of all occurring peaks in the mass spectrum, as described in
        :ref:`peak dictionary<apyt.massfit:Peak dictionary>`.
    """
    #
    #
    # check for valid mode
    if not (mode == 'mass' or mode == 'isotope'):
        raise Exception("Unknown mode \"{0:s}\".".format(mode))
    #
    #
    # loop through elements (element properties are passed as a tuple containing
    # the charge states and the volume for reconstruction)
    print("Using \"{0:s}\" mode for calculation of mass-to-charge ratios.\n".
          format(mode))
    peaks_dict = {}
    for element, (charge_states, volume) in element_dict.items():
        # check whether charge states contains only one element
        if type(charge_states) is not tuple:
            charge_states = (charge_states, )
        #
        #
        # check for possible molecule
        counts = list(periodictable.formula(element).atoms.values())
        if len(counts) > 1 or max(counts) > 1:
            isotopes = _get_molecular_isotopes_list(
                element, mode = mode, mass_decimals = mass_decimals
            )
        else:
            isotopes = periodictable.elements.symbol(element)
        #
        #
        # loop through charge states
        peaks_dict[element] = {}
        for q in tuple(charge_states):
            # get isotope with maximum abundance
            abundance_max = 0.0
            for isotope in isotopes:
                if isotope.abundance > abundance_max:
                    abundance_max = isotope.abundance
                    isotope_max = isotope
            #
            #
            # loop through isotopes
            peaks_dict[element][str(q)] = []
            for isotope in isotopes:
                # skip invalid isotope
                if isotope.abundance <= _abundance_thres * 100:
                    continue
                #
                # append peak dictionary to peak list
                peak_dict = {
                    'element':           element,
                    'charge':            q,
                    # set mass-to-charge ratio according to specified mode
                    # either from actual isotopic mass ("mass") or from mass
                    # number ("isotope")
                    'mass_charge_ratio': getattr(isotope, mode) / q,
                    'abundance':         isotope.abundance / 100,
                    'is_max':            False,
                    'volume':            volume
                }
                if isotope == isotope_max:
                    peak_dict['is_max'] = True
                peaks_dict[element][str(q)].append(peak_dict)
    #
    #
    # loop through all elements and charge states and create flat peak list
    peaks_list = []
    for element in peaks_dict.values():
        for charge_states_peaks_list in element.values():
            peaks_list.extend(charge_states_peaks_list)
    #
    #
    #
    #
    # print all peaks if requested
    if verbose == True:
        print("Total number of peaks is {0:d}:".format(len(peaks_list)))
        print('\t'.join(peaks_list[0].keys()))
        print("--------" * 8 + "--------")
        #
        # sort peaks by mass-to-charge ratio
        peaks_list_sorted = sorted(
            peaks_list, key = lambda peak: peak['mass_charge_ratio']
        )
        for peak in peaks_list_sorted:
            print("{0:s}\t{1:d}\t{2:.3f}\t\t\t{3:.6f}\t{4!r}\t{5:.6f}".
                  format(*peak.values()))
        print("")
    #
    #
    # return peak list and nested peaks dictionary
    return peaks_list, peaks_dict
#
#
#
#
def spectrum(x, peaks_list, function, params, elements_list = None):
    """
    Calculate spectrum for specified list of elements.

    Parameters
    ----------
    x : ndarray or scalar
        The position(s) where to evaluate the mass spectrum.
    peaks_list : list of dicts
        The list of all occurring peaks in the mass spectrum, as described in
        :ref:`peak dictionary<apyt.massfit:Peak dictionary>`.
    function : str
        The string identifying the general peak shape function, as described in
        :ref:`implementation notes<apyt.massfit:Implementation notes>`.
    params : dict
        The dictionary with the fit parameter names as keys, and best-fit values
        as values, as described in the |best_params| |model_result| attribute of
        the |lmfit| module.

    Keyword Arguments
    -----------------
    elements_list : list of str
        The list specifying for which elements the mass spectrum should be
        evaluated. Defaults to ``None``, indicating to use all elements
        occurring in *peaks_list*. Note that the background is not included.

    Returns
    -------
    y : ndarray or scalar
        The cumulated spectrum value at position *x* for the list of provided
        elements. This is a scalar if *x* is a scalar.
    """
    #
    #
    # cumulate all peak contributions for specified elements
    y = 0.0
    for peak in peaks_list:
        if elements_list is None or \
           _get_intensity_name(peak).split('_')[1] in elements_list:
            y += _peak_generic(function, params, 'eval', (x, peak))
    #
    #
    return y
#
#
#
#
def split_molecules(ids, xyz, peaks_list,
                    group_charge = True, shuffle = False, verbose = False):
    """
    Split molecular events into individual atoms.

    This function examines all events for potential molecules and decomposes
    them into their constituent atoms. The chemical IDs are remapped to unique
    elements in alphabetical order, and the three-dimensional coordinates of
    each molecule are assigned to all its individual atoms. The optional
    argument *shuffle* allows for random reordering of the atoms within a
    molecule.

    Parameters
    ----------
    ids : ndarray, shape (n,)
        The chemical IDs of the *n* events.
    xyz : ndarray, shape (n, 3)
        The three-dimensional reconstructed positions of the *n* events.
    peaks_list : list of dicts
        The list of all occurring peaks in the mass spectrum, as described in
        :ref:`peak dictionary<apyt.massfit:Peak dictionary>`.
    group_charge : bool
        Whether to group all charge states of an individual element. Defaults to
        ``True``.
    shuffle : bool
        Whether to randomly shuffle the order of the atoms after splitting.
        Defaults to ``False``.
    verbose : bool
        Whether to print a list of all chemical IDs and their fractions in
        relation to the total counts (with background). Defaults to ``False``.

    Returns
    -------
    ids_split : ndarray, shape (m,)
        The chemical IDs of the *m* events after splitting.
    xyz_split : ndarray, shape (m, 3)
        The three-dimensional reconstructed positions of the *m* events after
        splitting.
    """
    #
    #
    # start timer
    start = timer()
    print("Splitting molecules…")
    #
    #
    #
    #
    # get all molecule names from peak groups
    molecules = [
        group[0]['element']
            for group in _group_peaks(peaks_list, group_charge)
    ]
    #
    # get counts of all occurring molecular IDs (including background)
    molecule_counts = np.bincount(ids, minlength = len(molecules) + 1)
    #
    #
    # break down all molecules into list of individual elements
    # (with possible repetition)
    elements = [
        element.symbol
            for molecule in molecules
                for element in periodictable.formula(molecule).atoms.keys()
    ]
    #
    # convert unique, sorted elements into PeriodicTable element objects
    elements = [
        periodictable.elements.symbol(element)
            for element in sorted(set(elements))
    ]
    if verbose == True:
        print("\tList of unique elements is:", elements)
    #
    #
    # calculate total number of individual atoms *after* splitting
    # (including background)
    count_tot = molecule_counts[0]
    for molecule, count in zip(molecules, molecule_counts[1:]):
        count_tot += \
            count * np.sum(list(periodictable.formula(molecule).atoms.values()))
    #
    #
    # initialize arrays for split events (infer datatype from original arrays)
    ids_split = np.empty(count_tot, dtype = ids.dtype)
    xyz_split = np.empty((count_tot, xyz.shape[1]), dtype = xyz.dtype)
    #
    #
    #
    #
    # set background counts (id = 0)
    offset = molecule_counts[0]
    ids_split[0:offset] = 0
    xyz_split[0:offset] = xyz[(ids == 0)]
    #
    #
    # loop through all (unique) elements
    for element, element_id in zip(elements, range(1, len(elements) + 1)):
        if verbose == True:
            print("\tProcessing element '{0:s}' with new id {1:d}…".
                  format(element.symbol, element_id))
        # loop through all molecules
        for molecule, molecule_id, molecule_count in zip(
            molecules, range(1, len(molecules) + 1), molecule_counts[1:]):
            # test whether molecule contains element
            if element in periodictable.formula(molecule).atoms.keys():
                # set element count
                element_count = periodictable.formula(molecule).atoms[element]
                if verbose == True:
                    print(
                        "\t\tFound in molecule {0:>8s} with original id {1:2d} "
                        "({2:d} x {3:8d} counts).".
                        format("'" + molecule + "'", molecule_id, element_count,
                               molecule_count)
                )
                #
                #
                # remap element ID and assign molecule position to its
                # individual atoms
                sl = np.index_exp[offset:offset+molecule_count*element_count]
                ids_split[sl] = element_id
                xyz_split[sl] = np.repeat(
                    xyz[(ids == molecule_id)], element_count, axis = 0
                )
                #
                #
                # increment offset position
                offset += molecule_count * element_count
    #
    #
    #
    #
    # randomize order if requested
    if shuffle == True:
        print("Randomizing order…")
        #
        # initialize random number generator
        rng = np.random.default_rng(seed = 0)
        #
        # get random permutation
        permutation = rng.permutation(len(ids_split))
        #
        # shuffle IDs and corresponding positions
        ids_split = np.take(ids_split, permutation, axis = 0)
        xyz_split = np.take(xyz_split, permutation, axis = 0)
    #
    #
    #
    #
    # print counts of all IDs if requested
    if verbose == True:
        # get counts for all IDs (including background)
        counts = np.bincount(ids_split, minlength = len(elements) + 1)
        #
        #
        print("\nid\telement\t\t   count\tfraction")
        print("--------" * 5 + "--------")
        #
        # background counts (always first entry)
        print("{0:2d}\tBackground\t{1:8d}\t{2:.4f}".format(
            0, counts[0], counts[0] / count_tot
        ))
        #
        # loop through all elements
        for i, element in zip(range(1, len(elements) + 1), elements):
            print("{0:2d}\t{1:12s}\t{2:8d}\t{3:.4f}".format(
                i, element.symbol, counts[i], counts[i] / count_tot
            ))
        #
        print("========" * 5 + "========")
        print("\t\ttotal\t{0:8d}\n".format(count_tot))
    #
    #
    #
    #
    print(
        "Splitting of {0:d} molecules into {1:d} atoms took {2:.3f}s.".
        format(len(ids), count_tot, timer() - start)
    )
    return ids_split, xyz_split
#
#
#
#
################################################################################
#
# private module-level functions
#
################################################################################
def _activation(x):
    """
    Activation function

    The slope at the inflection point (0, 1/2) is identical to 1/2, which can
    be considered as a measure of the transition width.
    """
    #
    #
    return 0.5 * (erf(0.5 * np.sqrt(np.pi) * x) + 1.0)
#
#
#
#
def _decay(x):
    """
    Exponential decay

    The decay constant, at which the function value is reduced to 1/e, is
    defined to be unity. The caller needs to make sure that the exponential
    growth is limited in negative direction to avoid overflow.
    """
    #
    #
    return np.exp(-x)
#
#
#
#
def _estimate_fit_parameters(data, peaks_list, function, **kwargs):
    """
    Estimate initial fit parameters.
    """
    #
    #
    # initialize fit parameters object
    params = lmfit.Parameters()
    #
    #
    #
    #
    # add peak width scaling parameter (implemented as an exponent; defaults to
    # square root behavior, i.e. γ = 0.5)
    params.add(
        'γ', value = 0.5, min = 0.0, max = 2.0,
        vary = kwargs.get('scale_width', False)
    )
    if kwargs.get('scale_width', False) == True:
        print("Using peak width scaling γ.")
    #
    #
    # add oxygen peak shift parameter (implemented as an absolute shift which
    # scales with the square root of the peak position)
    params.add(
        'ΔO', value = 0.0, min = -0.1, max = 0.1,
        vary = kwargs.get('oxygen_shift', False)
    )
    if kwargs.get('oxygen_shift', False) == True:
        print("Using oxygen peak shift ΔO.")
    #
    #
    #
    #
    # estimate general peaks shape parameters and p(0), i.e. peak shape function
    # value at 0
    params, p_0 = _peak_generic(function, params, 'estimate', (data))
    #
    #
    # estimate baseline
    _, props = find_peaks(
        data[:, 1],
        distance = np.iinfo(np.int32).max,
        width = np.finfo(np.float32).eps, rel_height = 1.0)
    base = props['width_heights'][0]
    print("Estimated base line is {0:.1f}.".format(base))
    #
    # add baseline parameter
    params.add('base', value = base)
    #
    #
    # estimate peak intensities
    for peak in peaks_list:
        # only consider peak with maximum abundance
        if peak['is_max'] == False:
            continue
        #
        #
        # select peak region
        data_sel = data[
            (peak['mass_charge_ratio'] - 0.25 <= data[:, 0]) &
            (data[:, 0] <= peak['mass_charge_ratio'] + 0.25)
        ]
        #
        # find (maximum) peak in selected region
        _, props = find_peaks(
            data_sel[:, 1],
            distance = np.iinfo(np.int32).max, height = 0.0)
        #
        # check for correctly identified peak
        if len(props['peak_heights']) == 0:
            raise Exception(
                "No peak found for element '{0:s}' with maximum at "
                "{1:.3f} amu/e. You may want to check the alignment of the "
                "mass spectrum.".
                format(peak['element'], peak['mass_charge_ratio'])
            )
        #
        #
        # estimate intensity (for entire isotopic peak group at hypothetical
        # position at unity)
        I = props['peak_heights'][0] / p_0 / peak['abundance'] * \
            peak['mass_charge_ratio']
        params.add(_get_intensity_name(peak), value = I, min = 0.0)
    #
    #
    #
    #
    # return fit parameters object
    print("")
    return params
#
#
#
#
def _get_intensity_name(peak):
    """
    Simple wrapper to obtain intensity parameter name for element/charge state
    associated with specified peak.
    """
    #
    #
    return "I_{0:s}_{1:d}".format(peak['element'], peak['charge'])
#
#
#
#
def _get_molecular_isotopes_list(molecule, mode = 'mass', mass_decimals = 3):
    """
    Return (artificial) Element object with all isotopes for specified molecule.

    *mode* determines which mode should be used when the mass-to-charge ratio is
    calculated for the molecular isotopes. ``mass`` (the default) uses the
    actual mass of the molecular isotope, while ``isotope`` uses the mass number
    of the molecular isotope. The number of decimal places for the
    mass-to-charge ratio in ``mass`` mode should be limited to reduce the number
    of molecular isotopes. This setting effectively groups isotopes whose masses
    do not differ by more than the specified *mass_decimals*.
    """
    #
    #
    print("Determining isotope combinations for molecule \"{0:s}\"…".
          format(molecule))
    #
    #
    # get molecular constituents and their isotopes
    atomic_number = 0
    mass_numbers_list = []
    for element, element_count in periodictable.formula(molecule).atoms.items():
        # cumulate (artificial) atomic number
        atomic_number += element_count * element.number
        #
        #
        # get list of isotopes with non-zero abundance for current element
        isotopes_list, masses, abundances = _get_nonzero_isotopes(element)
        #
        #
        # calculate all possible isotope combinations (Cartesian product) (from
        # zero to element_count for each individual isotope which may form the
        # molecule for the current element); each element of the resulting list
        # is an n-tuple, where n corresponds to the number of (non-zero)
        # isotopes and the elements of the tuple denote the counts of the
        # respective isotopes; note that by definition, not all isotope
        # combinations in the Cartesian product add up to element_count
        isotope_combinations = itertools.product(
            range(element_count + 1), repeat = len(isotopes_list)
        )
        # only use isotope combinations where all counts add up to total element
        # count
        isotope_combinations = \
            [ic for ic in isotope_combinations if sum(ic) == element_count]
        #
        #
        # calculate probabilities and mass numbers for all valid isotope
        # combinations
        mass_numbers_list_element = []
        for ic in isotope_combinations:
            # calculate probability for current isotope combination according to
            # multinomial distribution and store together with mass number and
            # mass as dictionary
            mass_numbers_list_element.append({
                'mass_number': np.sum(
                    np.asarray(isotopes_list) * np.asarray(ic)
                ),
                # round mass to mass_decimals decimals to allow grouping below
                'mass': np.round(
                    np.sum(masses * np.asarray(ic)), decimals = mass_decimals
                ),
                'probability': multinomial.pmf(
                    ic, element_count, p = abundances / 100
                )
            })
        # sum total probability
        p_tot = sum([mn['probability'] for mn in mass_numbers_list_element])
        #
        #
        # print debug output if requested
        if _is_dbg == True:
            print("\nIsotope combinations for {0:s} ({1:d} atom(s) in "
                  "molecule):".format(element.name, element_count))
            for isotope in isotopes_list:
                print("#{0:s}\t".format(element[isotope].__repr__()), end = '')
            print("mass number\tmass\t\tprobability\n" +
                  "--------" * len(isotopes_list) +
                  "-------------------------------------------")
            #
            for ic in zip(isotope_combinations, mass_numbers_list_element):
                # skip combinations with negligible contribution
                if ic[1]['probability'] < _abundance_thres:
                    continue
                #
                print(("{:d}\t" * len(isotopes_list)).format(*ic[0]), end = '')
                print("{0:d}\t\t{1:07.3f}\t\t{2:.9f}".format(*ic[1].values()))
            print("========" * len(isotopes_list) +
                  "===========================================")
            print("\t" * len(isotopes_list) + "\t\t\ttotal\t{0:.9f}".
                  format(p_tot))
        #
        #
        # test whether all valid isotope combinations add up to 100%
        if abs(p_tot - 1.0) > np.finfo(np.float32).eps:
            raise Exception("Total probability ({0:.6f}) for \"{1:s}\" differs "
                            "from unity.".format(p_tot, element.name))
        #
        #
        # append mass numbers list for current element to overall mass numbers
        # list
        mass_numbers_list.append(mass_numbers_list_element)
    #
    #
    # loop through all element combinations and calculate combined probability
    # for molecule
    mass_numbers_list_molecule = []
    for mn_tuple in itertools.product(*mass_numbers_list):
        mass_numbers_list_molecule.append({
            # use 'isotope' key in conformation with periodictable module
            'isotope':   np.sum( [mn['mass_number'] for mn in mn_tuple]),
            'mass':      np.sum( [mn['mass']        for mn in mn_tuple]),
            'abundance': np.prod([mn['probability'] for mn in mn_tuple])
        })
    #
    #
    # create new Element object
    artificial_element = periodictable.core.Element(
        molecule, molecule, atomic_number, None, periodictable
    )
    if _is_dbg == True: print("")
    print("Artificial atomic number for \"{0:s}\" is {1:d}.".
          format(artificial_element.name, artificial_element.number))
    #
    #
    # add all possible isotopes to the artificial element
    for mn in mass_numbers_list_molecule:
        artificial_element.add_isotope(mn[mode])
    #
    #
    # initialize all abundances to zero
    for isotope in artificial_element:
        isotope._mass      = 0.0
        isotope._abundance = 0.0
    #
    #
    # cumulate abundances for every molecular isotope from mass numbers list
    # (multiple mass numbers may correspond to the same isotope number)
    for mn in mass_numbers_list_molecule:
        # in isotope mode, the mass would need to be weighted accordingly;
        # simply invalidate here
        if mode == 'mass':
            artificial_element[mn[mode]]._mass = mn['mass']
        else:
            artificial_element[mn[mode]]._mass = np.nan
        #
        # cumluate abundance
        artificial_element[mn[mode]]._abundance += mn['abundance'] * 100
        #
        # mass number is only needed in debug mode
        if _is_dbg == True:
            artificial_element[mn[mode]]._mass_number = mn['isotope']
    #
    #
    # sum total abundance
    total_abundance = 0.0
    for isotope in artificial_element:
        total_abundance += isotope.abundance
    #
    #
    # print debug output if requested
    if _is_dbg == True:
        print("\nIsotope combinations for molecule \"{0:s}\":".format(molecule))
        print("mass number\tmass\t\tprobability")
        print("-------------------------------------------")
        for isotope in artificial_element:
            if isotope.abundance >= _abundance_thres * 100:
                spec = '{1:07.3f}'  if mode == 'mass' else '{1:f}'
                mass = isotope.mass if mode == 'mass' else np.nan
                print(("{0:d}\t\t" + spec + "\t\t{2:.9f}").format(
                    isotope._mass_number, mass, isotope.abundance / 100)
                )
            #
            # remove _mass_number attribute (only needed in debug mode)
            delattr(isotope, "_mass_number")
            #
            #
        print("===========================================")
        print("\t\t\ttotal\t{0:.9f}\n".format(total_abundance / 100))
    #
    #
    # test whether all isotopes add up to 100%
    if abs(total_abundance - 100.0) > np.finfo(np.float32).eps:
        raise Exception("Total abundance ({0:.6f}) for molecule \"{1:s}\" "
                        "differs from unity.".
                        format(total_abundance / 100, molecule))
    #
    #
    # return artificial element
    print("")
    return artificial_element
#
#
#
#
def _get_nonzero_isotopes(element):
    """
    Return list of isotopes with non-zero abundances.
    """
    #
    #
    # loop through isotopes with non-zero abundances
    isotopes_list = []
    masses        = []
    abundances    = []
    for isotope in element.isotopes:
        if element[isotope].abundance > 0.0:
            isotopes_list.append(isotope)
            masses.append(element[isotope].mass)
            abundances.append(element[isotope].abundance)
    #
    #
    # return list of isotopes, their masses, and their abundances
    return isotopes_list, np.asarray(masses), np.asarray(abundances)
#
#
#
#
def _group_peaks(peaks_list, group_charge):
    """
    Group all peaks (isotopes) belonging to the same element (and charge state
    if requested).

    Parameters
    ----------
    peaks_list : list of dicts
        The list of all occurring peaks in the mass spectrum, as described in
        :ref:`peak dictionary<apyt.massfit:Peak dictionary>`.
    group_charge : bool
        Whether to group all charge states of an individual element.

    Returns
    -------
    peak_groups : list of lists
        The list of all peak groups, where each group is a list of the
        corresponding peaks (isotopes) of the same element (and charge state).
    """
    #
    #
    # loop through all peaks for grouping
    peak_groups  = []
    peaks        = []
    element      = peaks_list[0]['element']
    charge_state = peaks_list[0]['charge']
    for peak in peaks_list:
        # if peak belongs to same element (and charge state), append to current
        # peak group
        if peak['element'] == element and (
            peak['charge'] == charge_state or group_charge == True
        ):
            peaks.append(peak)
        # otherwise continue with next element (and charge state)
        else:
            # append previous peak group and start new peak group
            peak_groups.append(peaks)
            peaks = [peak]
            #
            # reset element and charge state
            element      = peak['element']
            charge_state = peak['charge']
    #
    # append last peak group
    peak_groups.append(peaks)
    #
    #
    # return peak groups
    return peak_groups
#
#
#
#
def _model_spectrum(x, peaks_list = None, function = None, **params):
    """
    Model function to describe complete mass spectrum.
    """
    #
    #
    # cumulate all peak contributions
    result = 0.0
    for peak in peaks_list:
        result += _peak_generic(function, params, 'eval', (x, peak))
    #
    #
    # return function value
    return result + params['base']
#
#
#
#
def _peak_generic(function, params, mode, arg_tuple):
    """
    Generic handle for peak shape function.

    This function is intended to serve as a generic wrapper to handle all peak
    shape functions defined below, as given by the *function* argument. This
    method is called from several other functions with a specified *mode* to
    indicate whether the initial peak shape parameters should be estimated
    (``estimate``), or whether the spectrum should be evaluated (``eval``), or
    whether the internal intensity parameter should be converted to the actual
    counts (``count``). New peak shape functions need to be implemented only
    here.
    """
    #
    #
    def get_peak_range(x, x0, Δmax):
        """
        Small helper function to determine the evaluation range around a peak
        position, returned as a slice object for the input array.

        Δmax can be either a list or tuple to determine individual ranges to the
        left and right, respectively, or a scalar where the the same range is
        used on both sides. Note that all ranges are positive.
        """
        #
        #
        # convert scalar to tuple with identical ranges on both sides
        if not isinstance(Δmax, (list, tuple)):
            Δmax = (Δmax, Δmax)
        #
        #
        # if input is a scalar, simply determine whether its value is within
        # evaluation range
        if np.isscalar(x):
            return -Δmax[0] <= x - x0 and x - x0 <= Δmax[1]
        #
        #
        # calculate separation between data points
        Δx = (x[-1] - x[0]) / (len(x) - 1)
        #
        # get nearest index of corresponding peak position
        i0 = int(np.rint((x0 - x[0]) / Δx))
        #
        # calculate index increments to be within valid evaluation range
        Δi = (int(Δmax[0] / Δx) + 1, int(Δmax[1] / Δx) + 1)
        #
        # return array slicing object
        return np.s_[max(0, i0 - Δi[0]) : min(len(x) - 1, i0 + Δi[1] + 1)]
    #
    #
    def peak_shift(peak_name, params, x0):
        """
        Small helper function to apply an additional specific shift based on the
        peak name.
        """
        #
        #
        # shift all matching pure oxygen peaks (scaled with square root of peak
        # position)
        if re.fullmatch(r"^I_O[0-9]*_[0-9]+$", peak_name) is not None:
            return params['ΔO'] * np.sqrt(x0)
        # no shift for all other peaks
        else:
            return 0.0
    #
    #
    #
    #
    # check for valid mode
    if mode != 'count' and mode != 'estimate' and mode != 'eval':
        raise Exception("Internal error: Unknown mode \"{0:s}\".".format(mode))
    #
    #
    #
    #
    # peak width scaling parameter common to all peak shape functions (may be
    # fixed to a constant value of 0.5)
    γ = params['γ']
    #
    #
    #
    #
    # find maximum peak in estimation mode for *all* peak shape functions
    if mode == 'estimate':
        # unpack additional mode-specific arguments
        data = arg_tuple
        #
        # determine maximum peak to estimate general peak shape parameters
        print("Finding maximum peak to estimate general peak shape parameters…")
        peak, props = find_peaks(
            data[:, 1],
            distance = np.iinfo(np.int32).max,
            width = 0.0, rel_height = 1.0 - 1.0 / np.e
        )
        peak_pos = data[peak[0], 0]
        print("Position of maximum peak is {0:.2f} amu/e.".format(peak_pos))
    #
    #
    #
    #
    # peak shape: error function activation with exponential decay
    if function == "error-expDecay":
        # estimate general peak shape parameters
        if mode == 'estimate':
            # estimate decay constant
            τ = props['widths'][0] * (data[2, 0] - data[1, 0]) / peak_pos**γ
            params.add('τ', value = τ, min = 0.0)
            print("Estimated decay constant is {0:.6f} amu/e.".format(τ))
            #
            # return estimated fit parameters and p(0), i.e. peak shape function
            # value at 0
            return params, 0.5
        #
        #
        #
        #
        # parse common parameters for 'count' and 'eval' mode
        τ = params['τ']
        #
        #
        # return unified area to normalize counts
        if mode == 'count':
            return np.exp(1.0 / np.pi) * τ
        #
        #
        # evaluate function
        if mode == 'eval':
            # unpack additional mode-specific arguments
            x, peak = arg_tuple
            #
            # parse parameters
            I  = params[_get_intensity_name(peak)]
            x0 = peak['mass_charge_ratio']
            #
            #
            # set additional possible shift based on peak name
            Δ = peak_shift(_get_intensity_name(peak), params, x0)
            #
            #
            # evaluation range is limited to multiple of decay constant on
            # either side of the peak; 0.0 elsewhere
            # (prevents possible overflow and is considerably faster)
            Δmax = 20.0 * τ * x0**γ
            #
            #
            # get evaluation range around peak
            peak_range = get_peak_range(x, x0, Δmax)
            #
            #
            # input was array; use slice
            if isinstance(peak_range, slice):
                # transform data (only in evaluation interval)
                x_t = (x[peak_range] - x0 - Δ) / (x0**γ * τ)
                #
                # pre-fill results with zeros
                y = np.zeros_like(x)
                #
                # intensities are calculated for hypothetical peak position at
                # unity to account for the scaling with respect to the mass-to-
                # charge ratio
                y[peak_range] = I / x0**γ * peak['abundance'] * \
                    _activation(x_t) * _decay(x_t)
                #
                # return result
                return y
            #
            # input was scalar and in evaluation range
            elif peak_range == True:
                # transform data
                x_t = (x - x0 - Δ) / (x0**γ * τ)
                #
                return I / x0**γ * peak['abundance'] * \
                    _activation(x_t) * _decay(x_t)
            #
            # input was scalar, but outside evaluation range
            elif peak_range == False:
                return 0.0
    #
    #
    #
    #
    # peak shape: error function activation with double exponential decay
    elif function == "error-doubleExpDecay":
        # estimate general peak shape parameters
        if mode == 'estimate':
            # estimate decay constant
            τ = props['widths'][0] * (data[2, 0] - data[1, 0]) / peak_pos**γ
            params.add('τ1', value =       τ, min = 0.0)
            params.add('τ2', value = 0.1 * τ, min = 0.0)
            params.add('φ',  value = 0.5,     min = 0.0, max = 1.0)
            print("Estimated decay constant is {0:.6f} amu/e.".format(τ))
            #
            # return estimated fit parameters and p(0), i.e. peak shape function
            # value at 0
            return params, 0.5
        #
        #
        #
        #
        # parse common parameters for 'count' and 'eval' mode
        τ1 = params['τ1']
        τ2 = params['τ2']
        φ  = params['φ']
        #
        # activation parameter is fixed through maximum condition at nominal
        # mass-to-charge ratio
        w = τ1 * τ2 / ((1.0 - φ) * τ1 + φ * τ2)
        #
        #
        # return unified area to normalize counts
        if mode == 'count':
            return     φ  * np.exp((w / τ1)**2 / np.pi) * τ1 + \
                (1.0 - φ) * np.exp((w / τ2)**2 / np.pi) * τ2
        #
        #
        # evaluate function
        if mode == 'eval':
            # unpack additional mode-specific arguments
            x, peak = arg_tuple
            #
            # parse parameters
            I  = params[_get_intensity_name(peak)]
            x0 = peak['mass_charge_ratio']
            #
            #
            # set additional possible shift based on peak name
            Δ = peak_shift(_get_intensity_name(peak), params, x0)
            #
            #
            # evaluation range is limited to multiple of width of error function
            # onset on the left side and largest decay constant on the right
            # side; 0.0 elsewhere (prevents possible overflow and is
            # considerably faster)
            Δmax = (20.0 * w * x0**γ, 20.0 * max(τ1, τ2) * x0**γ)
            #
            #
            # get evaluation range around peak
            peak_range = get_peak_range(x, x0, Δmax)
            #
            #
            # input was array; use slice
            if isinstance(peak_range, slice):
                # transform data (only in evaluation interval)
                x_t = (x[peak_range] - x0 - Δ) / x0**γ
                #
                # pre-fill results with zeros
                y = np.zeros_like(x)
                #
                # intensities are calculated for hypothetical peak position at
                # unity to account for the scaling with respect to the mass-to-
                # charge ratio
                y[peak_range] = \
                    I / x0**γ * peak['abundance'] * _activation(x_t / w) * (
                        φ * _decay(x_t / τ1) + (1.0 - φ) * _decay(x_t / τ2)
                    )
                #
                # return result
                return y
            #
            # input was scalar and in evaluation range
            elif peak_range == True:
                # transform data
                x_t = (x - x0 - Δ) / x0**γ
                #
                return \
                    I / x0**γ * peak['abundance'] * _activation(x_t / w) * (
                        φ * _decay(x_t / τ1) + (1.0 - φ) * _decay(x_t / τ2)
                    )
            #
            # input was scalar, but outside evaluation range
            elif peak_range == False:
                return 0.0
    #
    #
    else:
        raise Exception(
            "Unknown peak shape function \"{0:s}\".".format(function)
        )
