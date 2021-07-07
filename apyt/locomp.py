"""
The APyT local composition module
=================================


Introduction
------------


General procedure
-----------------


Howto
-----

The usage of this module is demonstrated in an auxiliary script
(``wrapper_scripts/apyt_locomp.py``) which basically serves as a wrapper for
this module. Detailed usage information can be obtained by invoking this script
with the ``"--help"`` option.


List of methods
---------------

This module provides some generic functions for the calculation of local
composition histograms.

The following methods are provided:

* :meth:`build_tree`: Build tree for neighbor search.
* :meth:`calc_stats`: Calculate statistics.
* :meth:`check_periodic_box`: .
* :meth:`get_composition`: .
* :meth:`get_query_points`: .
* :meth:`query_nearest_neighbors`: .


.. sectionauthor:: Sebastian M. Eich <Sebastian.Eich@imw.uni-stuttgart.de>
.. moduleauthor::  Sebastian M. Eich <Sebastian.Eich@imw.uni-stuttgart.de>

"""
#
#
#
#
__version__ = '0.1.0'
__all__ = [
    'build_tree',
    'calc_stats',
    'check_periodic_box',
    'get_composition',
    'get_query_points',
    'query_nearest_neighbors'
]
#
#
#
#
# import modules
import multiprocessing
import numpy as np
#
# import some special functions
from functools import partial
from scipy.spatial import cKDTree
#
#
#
#
################################################################################
#
# public functions
#
################################################################################
def build_tree(coords, boxsize):
    return cKDTree(coords, boxsize = boxsize)
#
#
#
#
def calc_stats(data, **kwargs):
    # get bin_width option
    bin_width = kwargs.get('bin_width', None)
    #
    #
    # set data properties
    num      = len(data)
    data_min = min(data)
    data_max = max(data)
    #
    #
    #
    if bin_width is None:
        bins = np.linspace(data_min - 0.5, data_max + 0.5, data_max - data_min + 2)
        bin_centers = np.arange(data_min, data_max + 1)
        #
        # calculate histogram data
        counts, bin_edges = np.histogram(data, bins = bins)
    elif bin_width == 'auto':
        # calculate histogram data
        counts, bin_edges = np.histogram(data, bins = 'auto')
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    #
    #
    # normalize histogram
    counts_norm = counts / (num * (bin_edges[1:] - bin_edges[:-1]))
    #
    #
    # calculate moments
    mu       = sum(counts *  bin_centers)          / num
    var      = sum(counts * (bin_centers - mu)**2) / num
    moment_4 = sum(counts * (bin_centers - mu)**4) / num
    #
    #
    # calulate statistical errors
    delta_mu    = np.sqrt(var / num)
    delta_var   = 2.0 / np.sqrt(num) * np.sqrt(moment_4);
    delta_var_r = 1.0 / (mu * np.sqrt(num)) * \
                  np.sqrt(4.0 * moment_4 + var**3 / mu**2)
    #
    #
    # return histogram data and statistics
    return (counts, bin_edges, bin_centers, counts_norm), \
           (mu, delta_mu, var, delta_var, var / mu, delta_var_r)
#
#
#
#
def check_periodic_box(comment):
    box = None
    try:
        pbc_index = comment.index('pbc')
        if comment[pbc_index + 1] == '1' and \
           comment[pbc_index + 2] == '1' and \
           comment[pbc_index + 3] == '1':
            #
            # get box dimension
            box = np.zeros(3)
            for i in range(0, 3):
                try:
                    pos = comment.index('cell_vec' + str(i + 1))
                    box[i] = comment[pos + i + 1]
                except:
                    print('Could not find box size for dimension {0:d}.'
                    .format(i + 1))
                    exit(1)
    except:
        pass
    return box
#
#
#
#
def get_composition(data, indices):
    partial_func = partial(_get_composition, types = data[:, 0].astype(int))
    
    pool = multiprocessing.Pool()
    return np.asarray(pool.map(partial_func, indices))
#
#
#
#
def get_query_points(coords, distance):
    # construct 3d grid if requested
    if distance is not None:
        if distance <= 0.0:
            print('Distance ({0:.3f}) must be positive.'.format(distance))
            exit(1)
        # get maximum positions for each direction
        max_pos = np.amax(coords, axis = 0)
        grid = []
        # construct 1d grid for each direction
        for i in range(0, 3):
            # number of grid points
            n_grid = int(max_pos[i] / distance)
            # separation between grid points
            delta = max_pos[i] / n_grid
            # construct grid
            grid.append(
                np.linspace(delta, (n_grid - 1) * delta, num = n_grid - 1))
        # construct 3d grid out of 1d grids
        return np.vstack(
                   np.meshgrid(grid[0], grid[1], grid[2], indexing = 'ij')
               ).reshape(3, -1).T
    # otherwise use all positions
    else:
        return coords
#
#
#
#
def query_nearest_neighbors(tree, query_points, neighbors):
    return tree.query(query_points, k = neighbors, n_jobs = -1)
#
#
#
#
################################################################################
#
# private module-level functions
#
################################################################################
def _get_composition(indices, types):
    # select atomic subset, then sum types (subtraction of 1 effectively
    # counts only atoms of type 2
    return sum(types[indices] - 1)
