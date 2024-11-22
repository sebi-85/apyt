"""
The APyT spectrum analysis module
=================================


List of functions
-----------------

This module provides the following functions for the analysis of time-of-flight
or mass spectra, respectively:

* :func:`correlation`: Calculate correlated mass-to-charge ratios for
  multi-events.


.. sectionauthor:: Sebastian M. Eich <Sebastian.Eich@imw.uni-stuttgart.de>
.. moduleauthor::  Sebastian M. Eich <Sebastian.Eich@imw.uni-stuttgart.de>

"""
#
#
#
#
__version__ = '0.1.0'
__all__ = [
    'correlation'
]
#
#
#
#
# import modules
import numpy as np
#
# import some special functions/modules
#
#
#
#
################################################################################
#
# public functions
#
################################################################################
def correlation(mc_ratios, pulse_nums, order = 2):
    """
    Calculate correlated mass-to-charge ratios for multi-events.


    Parameters
    ----------
    mc_ratios: ndarray, shape (n,)
        The mass-to-charge ratios of the *n* input events.
    pulse_nums: ndarray, shape (n,)
        The corresponding pulse numbers of the *n* input events.
    order: int
        The order of the multi-events. Defaults to ``2``, i.e. calculate double
        events.

    Returns
    -------
    mc_ratios_corr: ndarray, shape (m, order)
        The correlated mass-to-charge ratios of the *m* multi-events.
    """
    #
    #
    # mark event as valid multi-event if subsequent event/s has/have identical
    # pulse number, accounting for the multi-event order
    corr_mask = np.full(len(pulse_nums), False)
    corr_mask[:-(order-1)] = \
        (pulse_nums[:-(order-1)] - pulse_nums[order-1:] == 0)
    #
    # set number of multi-events
    n_multi = np.count_nonzero(corr_mask)
    print(
        "Number of multi-events of order {0:d} is {1:d}.".
        format(order, n_multi)
    )
    #
    #
    # set mass-to-charge ratios of multi-events
    mc_ratios_corr = np.zeros((n_multi, order), dtype = mc_ratios.dtype)
    mc_ratios_corr[:, 0] = mc_ratios[corr_mask]
    for i in range(1, order):
        mc_ratios_corr[:, i] = mc_ratios[i:][corr_mask[:-i]]
    #
    # sort events in case the input event order is reversed
    mc_ratios_corr.sort(axis = 1)
    #
    #
    # return mass-to-charge ratios of the multi-events
    return mc_ratios_corr
