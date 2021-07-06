"""
The APyT locomp module
======================


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
    partial_func = partial(_get_composition, data = data)
    
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
def _get_composition(indices, data):
    # select atomic subset, then sum type column (subtraction of 1 effectively
    # counts only atoms of type 2
    return sum(data[indices][:, 0] - 1)
