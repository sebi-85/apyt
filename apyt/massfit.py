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

Also, the peak shape may be more complex so that the tailing decay can only be
described by a sum of two exponential decay functions, i.e. using two decay
constants. This approach is used in this module for *all* peaks with identical
decay constants.


Data type description
---------------------

This modules relies mainly on dictionaries for the input/output parameter
interface which are described in the following.

Element dictionary
^^^^^^^^^^^^^^^^^^

The element dictionary consists of key--value pairs, where each key represents
one element (may also be a molecule) and the value is a tuple containing the
occurring charge states of the respective element.

Isotope dictionary
^^^^^^^^^^^^^^^^^^

The isotope dictionary consists of the following keys:

* ``element``: The element name (may also be a molecule).
* ``charge``: The charge state.
* ``mass_charge_ratio``: The mass-to-charge ratio for this isotope.
* ``abundance``: The isotope abundance (as a decimal fraction).
* ``is_max``: Whether this isotope has maximum abundance for the respective
  element.

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
* :meth:`isotope_list`: Get list of all isotopes for specified elements and
  charge states.
* :meth:`spectrum`: Calculate spectrum for specified list of elements.


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
    'isotope_list',
    'spectrum'
]
#
#
#
#
# import modules
import itertools
import lmfit
import numpy as np
import periodictable
import warnings
#
# import some special functions/modules
from scipy.signal import find_peaks, peak_widths
from scipy.special import erf
from scipy.stats import multinomial
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
def _add_element(name, symbol, number, table, isotopes):
    """
    Add new element to periodic table.
    """
    #
    #
    # create new element
    element = table.core.Element(name, symbol, number, None, table)
    #
    # add isotopes to element
    for isotope in isotopes:
        element.add_isotope(isotope[0])
        element[isotope[0]]._abundance = isotope[1]
    #
    #
    # add new element to table (append 'X' to indicate custom element)
    table.elements._element["{0:d}X".format(number)] = element
    setattr(table,          symbol, element)
    setattr(table.elements, symbol, element)
#
#
# add gallium used by FIB (only one isotope!)
_add_element("gallium (FIB)", "Ga_FIB", 31, periodictable, [(69, 100.0)])
#
#
#
#
################################################################################
#
# public functions
#
################################################################################
def counts(isotope_list, params, data_range, bin_width, ignore_list = [],
           verbose = False):
    """
    Get counts for all elements.

    This functions loops trough all isotopes in *isotope_list* with the
    ``is_max`` key set to ``True`` and returns the counts for each element and
    charge state combination.

    Parameters
    ----------
    isotope_list : list of dicts
        The list of all occurring isotopes in the mass spectrum, as described in
        :ref:`isotope dictionary<apyt.massfit:Isotope dictionary>`.
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
    count_list : list of dicts
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
    # parse parameters
    τ1 = params['τ1']
    τ2 = params['τ2']
    φ  = params['φ']
    #
    # activation parameter is fixed through maximum condition at nominal
    # mass-to-charge ratio
    w = τ1 * τ2 / ((1.0 - φ) * τ1 + φ * τ2)
    #
    #
    # loop through all isotopes
    count_list = []
    total_counts = 0.0
    for isotope in isotope_list:
        # calculate element count only from isotope with maximum abundance
        if isotope['is_max'] == False:
            continue
        #
        # get intensity parameter for current isotope
        I = params[_get_intensity_name(isotope)]
        #
        # calculate count
        count = I / bin_width * (
                   φ  * np.exp((w / τ1)**2 / np.pi) * τ1 +
            (1.0 - φ) * np.exp((w / τ2)**2 / np.pi) * τ2
        )
        #
        # cumulate total counts
        total_counts += count
        #
        # append count dictionary to count list
        count_list.append({
            'element':  isotope['element'],
            'charge':   isotope['charge'],
            'count':    count,
            'fraction': 0.0
            })
    #
    # add fractions
    for count in count_list:
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
        print("element\tcharge\t   count\tfraction")
        print("--------" * 4 + "--------")
        for count in count_list:
            print("{0:s}\t{1:d}\t{2:8.0f}\t{3:.4f}".
                  format(*count.values()))
        print("========" * 4 + "========")
        print("\ttotal\t{0:8.0f}\n".format(total_counts))
        #
        #
        # combine counts for different charge states for individual elements
        print("element\t   count\tfraction")
        print("--------" * 3 + "--------")
        element_count = 0
        element       = count_list[0]['element']
        for count in count_list:
            if count['element'] == element:
                element_count += count['count']
            else:
                # print previous element and reset
                print("{0:s}\t{1:8.0f}\t{2:.4f}".format(
                      element, element_count, element_count / total_counts))
                element_count = count['count']
                element       = count['element']
        # print last element
        print("{0:s}\t{1:8.0f}\t{2:.4f}".
              format(element, element_count, element_count / total_counts))
        print("========" * 3 + "========")
        print("total\t{0:8.0f}\n".format(total_counts))
        #
        #
        # break down molecules into individual elements
        count_dict = {}
        is_molecule = False
        for count in count_list:
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
            print("element\t   count\tfraction")
            print("--------" * 3 + "--------")
            for element, element_count in count_dict.items():
                print("{0:s}\t{1:8.0f}\t{2:.4f}".
                      format(element, element_count,
                             element_count / sum(count_dict.values())))
            print("========" * 3 + "========")
            print("total\t{0:8.0f}\n".format(sum(count_dict.values())))
    #
    #
    # return element counts, total counts, and background counts
    return count_list, total_counts, background
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
def fit(spectrum, isotope_list, verbose = False):
    """
    Fit mass spectrum.

    This function internally uses the |lmfit| module to fit the complete mass
    spectrum, where the peak positions are provided by the *isotope_list*
    argument.

    Parameters
    ----------
    spectrum : ndarray, shape (n, 2)
        The mass spectrum histogram data.
    isotope_list : list of dicts
        The list of all occurring isotopes in the mass spectrum, as described in
        :ref:`isotope dictionary<apyt.massfit:Isotope dictionary>`.

    Keyword Arguments
    -----------------
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
    model = lmfit.Model(_model_spectrum)
    #
    # estimate fit parameters
    parameters = _estimate_fit_parameters(spectrum, isotope_list)
    #
    #
    # perform fit and print results
    with warnings.catch_warnings():
        warnings.filterwarnings(
            'ignore',
            message = "The keyword argument isotope_list does not match any " +
            "arguments of the model function. It will be ignored.",
            category = UserWarning)
        result = model.fit(spectrum[:, 1], parameters, x = spectrum[:, 0],
                           isotope_list = isotope_list)
    if verbose == True:
        print(result.fit_report(show_correl = False))
        print("")
    #
    #
    # return fit results
    return result
#
#
#
#
def isotope_list(element_dict, verbose = False):
    """
    Get list of all isotopes for specified elements and charge states.

    Parameters
    ----------
    element_dict : dict
        The dictionary containing the elements and their charge states, as
        described in :ref:`element dictionary<apyt.massfit:Element dictionary>`.

    Keyword Arguments
    -----------------
    verbose : bool
        Whether to print the content of all determined isotope dictionaries.
        Defaults to ``False``.

    Returns
    -------
    isotope_list : list of dicts
        The list of all occurring isotopes in the mass spectrum, as described in
        :ref:`isotope dictionary<apyt.massfit:Isotope dictionary>`.
    """
    #
    #
    # loop through elements
    isotope_list = []
    for element, charge_states in element_dict.items():
        # check whether charge states contains only one element
        if type(charge_states) is not tuple:
            charge_states = (charge_states, )
        #
        #
        # check for possible molecule
        counts = list(periodictable.formula(element).atoms.values())
        if len(counts) > 1 or max(counts) > 1:
            isotopes = _get_molecular_isotope_list(element)
        else:
            isotopes = periodictable.elements.symbol(element)
        #
        #
        # loop through charge states
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
            for isotope in isotopes:
                # skip invalid isotope
                if isotope.abundance <= _abundance_thres * 100:
                    continue
                #
                #
                # set mass-to-charge ratio from isotope name
                if isotope.__str__() == 'D':
                    mass_charge_ratio = 2.0 / q
                elif isotope.__str__() == 'T':
                    mass_charge_ratio = 3.0 / q
                else:
                    mass_charge_ratio = isotope.isotope / q
                #
                # append isotope dictionary to isotope list
                isotope_dict = {
                    'element':           element,
                    'charge':            q,
                    'mass_charge_ratio': mass_charge_ratio,
                    'abundance':         isotope.abundance / 100,
                    'is_max':            False
                }
                if isotope == isotope_max:
                    isotope_dict['is_max'] = True
                isotope_list.append(isotope_dict)
    #
    #
    # print all isotopes if requested
    if verbose == True:
        print("Total number of isotopes is {0:d}:".format(len(isotope_list)))
        print('\t'.join(isotope_list[0].keys()))
        print("--------" * 7 + "------")
        for isotope in isotope_list:
            print("{0:s}\t{1:d}\t{2:.2f}\t\t\t{3:.6f}\t{4!r}".
                  format(*isotope.values()))
        print("")
    #
    #
    # return isotope list
    return isotope_list
#
#
#
#
def spectrum(x, params, isotope_list, element_list = None):
    """
    Calculate spectrum for specified list of elements.

    Parameters
    ----------
    x : ndarray or scalar
        The position(s) where to evaluate the mass spectrum.
    params : dict
        The dictionary with the fit parameter names as keys, and best-fit values
        as values, as described in the |best_params| |model_result| attribute of
        the |lmfit| module.
    isotope_list : list of dicts
        The list of all occurring isotopes in the mass spectrum, as described in
        :ref:`isotope dictionary<apyt.massfit:Isotope dictionary>`.

    Keyword Arguments
    -----------------
    element_list : list of str
        The list specifying for which elements the mass spectrum should be
        evaluated. Defaults to ``None``, indicating to use all elements
        occurring in *isotope_list*. Note that the background is not included.

    Returns
    -------
    y : ndarray or scalar
        The cumulated spectrum value at position *x* for the list of provided
        elements. This is a scalar if *x* is a scalar.
    """
    #
    #
    # cumulate all isotope contributions for specified elements
    y = 0.0
    for isotope in isotope_list:
        if element_list is None or \
           _get_intensity_name(isotope).split('_')[1] in element_list:
            y += _isotope_spectrum(x, params, isotope)
    #
    #
    return y
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
    defined to be unity.
    """
    #
    #
    # lower bound to limit exponential growth in negative direction
    x_min = -10
    #
    #
    return np.where(x < x_min, 0.0, np.exp(-x, where = (x >= x_min)))
#
#
#
#
def _estimate_fit_parameters(data, isotope_list):
    """
    Estimate initial fit parameters.
    """
    #
    #
    # initialize fit parameter object
    params = lmfit.Parameters()
    #
    #
    # determine maximum peak to estimate decay constant
    print("Finding maximum peak to estimate general peak shape parameters…")
    peak, props = find_peaks(
        data[:, 1],
        distance = np.iinfo(np.int32).max,
        width = 0.0, rel_height = 1.0 - 1.0 / np.e)
    peak_pos = data[peak[0], 0]
    τ = props['widths'][0] * (data[2, 0] - data[1, 0]) / peak_pos
    print("Position of maximum peak is {0:.2f} amu/e.".format(peak_pos))
    print("Estimated decay constant is {0:.6f} amu/e.".format(τ))
    #
    # add decay constant parameters
    params.add('τ1', value =       τ, min = 0.0)
    params.add('τ2', value = 0.1 * τ, min = 0.0)
    params.add('φ',  value = 0.5)
    #
    #
    # estimate baseline
    peak, props = find_peaks(
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
    #
    #
    # estimate peak intensities
    for isotope in isotope_list:
        # only consider isotope with maximum abundance
        if isotope['is_max'] == False:
            continue
        #
        #
        # select peak region
        data_sel = data[
            (isotope['mass_charge_ratio'] - 0.25 <= data[:, 0]) &
            (data[:, 0] <= isotope['mass_charge_ratio'] + 0.25)
        ]
        #
        # find (maximum) peak in selected region
        peak, props = find_peaks(
            data_sel[:, 1],
            distance = np.iinfo(np.int32).max, height = 0.0)
        #
        #
        # estimate intensity (for entire isotopic peak group at hypothetical
        # position at unity)
        I = 2.0 * props['peak_heights'][0] / isotope['abundance'] * \
            isotope['mass_charge_ratio']
        params.add(_get_intensity_name(isotope), value = I, min = 0.0)
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
def _get_intensity_name(isotope):
    """
    Simple wrapper to obtain peak intensity for specific element
    """
    #
    #
    return "I_{0:s}_{1:d}".format(isotope['element'], isotope['charge'])
#
#
#
#
def _get_molecular_isotope_list(molecule):
    """
    Return (artificial) Element object with all isotopes for specified molecule.
    """
    #
    #
    print("Determining isotope combinations for molecule \"{0:s}\"…".
          format(molecule))
    #
    #
    # get molecular constituents and their isotopes
    atomic_number = 0
    mass_number_list = []
    for element, element_count in periodictable.formula(molecule).atoms.items():
        # cumulate (artificial) atomic number
        atomic_number += element_count * element.number
        #
        #
        # get list of isotopes with non-zero abundance for current element
        isotope_list, abundances = _get_nonzero_isotopes(element)
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
            range(element_count + 1), repeat = len(isotope_list)
        )
        # only use isotope combinations where all counts add up to total element
        # count
        isotope_combinations = \
            [ic for ic in isotope_combinations if sum(ic) == element_count]
        #
        #
        # calculate probabilities and mass number for all valid isotope
        # combinations
        mass_number_list_element = []
        for ic in isotope_combinations:
            # calculate probability for current isotope combination according to
            # multinomial distribution and store together with mass number as
            # tuple
            mass_number_list_element.append((
                np.sum(np.asarray(isotope_list) * np.asarray(ic)), # mass number
                multinomial.pmf(ic, element_count, p = abundances / 100)
            ))
        # sum total probability
        p_tot = sum([mn[1] for mn in mass_number_list_element])
        #
        #
        # print debug output if requested
        if _is_dbg == True:
            print("\nIsotope combinations for {0:s} ({1:d} atom(s) in "
                  "molecule):".format(element.name, element_count))
            for isotope in isotope_list:
                print("#{0:s}\t".format(element[isotope].__repr__()), end = '')
            print("mass number\tprobability\n" +
                  "--------" * len(isotope_list) +
                  "---------------------------")
            #
            for ic in zip(isotope_combinations, mass_number_list_element):
                # skip combinations with negligible contribution
                if ic[1][1] < _abundance_thres:
                    continue
                #
                print(("{:d}\t" * len(isotope_list)).format(*ic[0]), end = '')
                print("{0:d}\t\t{1:.9f}".format(*ic[1]))
            print("========" * len(isotope_list) +
                  "===========================")
            print("\t" * len(isotope_list) + "\ttotal\t{0:.9f}".format(p_tot))
        #
        #
        # test whether all valid isotope combinations add up to 100%
        if abs(p_tot - 1.0) > np.finfo(np.float32).eps:
            raise Exception("Total probability ({0:.6f}) for \"{1:s}\" differs "
                            "from unity.".format(p_tot, element.name))
        #
        #
        # append mass number list for current element to overall mass number
        # list
        mass_number_list.append(mass_number_list_element)
    #
    #
    # loop through all element combinations and calculate combined probability
    # for molecule
    mass_number_list_molecule = []
    for mn_tuple in itertools.product(*mass_number_list):
        mass_number_list_molecule.append((
            # sum of elemental mass numbers
            np.sum( [mn[0] for mn in mn_tuple]),
            # product of elemental probabilities
            np.prod([mn[1] for mn in mn_tuple])
        ))
    #
    #
    # create new Element object
    artificial_element = periodictable.core.Element(
        molecule, molecule, atomic_number, None, periodictable
    )
    if _is_dbg == True: print("")
    print("Artifical atomic number for \"{0:s}\" is {1:d}.".
          format(artificial_element.name, artificial_element.number))
    #
    #
    # add all possible isotopes to the artificial element
    for mn in mass_number_list_molecule:
        artificial_element.add_isotope(mn[0])
    #
    #
    # initialize all abundances to zero
    for isotope in artificial_element:
        isotope._abundance = 0.0
    #
    #
    # cumulate abundance for every molecular isotope from mass number list
    # (multiple mass numbers may correspond to the same isotope number)
    for mn in mass_number_list_molecule:
        artificial_element[mn[0]]._abundance += mn[1] * 100
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
        print("mass number\tprobability\n---------------------------")
        for isotope in artificial_element:
            if isotope.abundance >= _abundance_thres * 100:
                print("{0:d}\t\t{1:.9f}".
                      format(isotope.isotope, isotope.abundance / 100))
        print("===========================\n\ttotal\t{0:.9f}\n".
              format(total_abundance / 100))
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
    isotope_list = []
    abundances   = []
    for isotope in element.isotopes:
        if element[isotope].abundance > 0.0:
            isotope_list.append(isotope)
            abundances.append(element[isotope].abundance)
    #
    #
    # return list of isotopes and their abundances
    return isotope_list, np.asarray(abundances)
#
#
#
#
def _isotope_spectrum(x, params, isotope):
    """
    Get spectrum for specified isotope.
    """
    #
    #
    # parse parameters
    τ1 = params['τ1']
    τ2 = params['τ2']
    φ  = params['φ']
    I  = params[_get_intensity_name(isotope)]
    x0 = isotope['mass_charge_ratio']
    #
    # activation parameter is fixed through maximum condition at nominal
    # mass-to-charge ratio
    w = τ1 * τ2 / ((1.0 - φ) * τ1 + φ * τ2)
    #
    #
    # intensities are calculated for hypothetical peak position at unity to
    # account for the scaling with the mass-to-charge ratio
    return I / x0 * isotope['abundance'] * _activation((x / x0 - 1.0) / w) * (
               φ  * _decay((x / x0 - 1.0) / τ1) +
        (1.0 - φ) * _decay((x / x0 - 1.0) / τ2)
    )
#
#
#
#
def _model_spectrum(x, isotope_list = None, **params):
    """
    Model function to describe complete mass spectrum.
    """
    #
    #
    # cumulate all isotope contributions
    result = 0.0
    for isotope in isotope_list:
        result += _isotope_spectrum(x, params, isotope)
    #
    #
    # return function value
    return result + params['base']
