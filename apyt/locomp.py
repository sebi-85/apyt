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

* :meth:`calc_stats`: Calculate statistics.
* :meth:`check_periodic_box`: .
* :meth:`get_composition`: .
* :meth:`get_query_points`: .


.. sectionauthor:: Sebastian M. Eich <Sebastian.Eich@imw.uni-stuttgart.de>
.. moduleauthor::  Sebastian M. Eich <Sebastian.Eich@imw.uni-stuttgart.de>

"""
#
#
#
#
__version__ = '0.1.0'
__all__ = [
    'calc_stats',
    'check_periodic_box',
    'get_composition',
    'get_query_points'
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
from psutil import virtual_memory
from scipy.spatial import cKDTree
from sys import getsizeof, stderr
#
#
#
#
################################################################################
#
# private module-level variables
#
################################################################################
# set approximate maximum amount of available memory to use
_mem_threshold = 0.50
#
#
#
#
################################################################################
#
# public functions
#
################################################################################
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
    comment = comment.split()
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
def get_composition(data, query_points, query, **kwargs):
    # set verbosity
    verbose = kwargs.get('verbose', False)
    #
    #
    # set atomic types
    types = data[:, 0].astype(int)
    #
    #
    # build tree
    if verbose == True:
        print('Building tree ...')
    tree = cKDTree(data[:, 1:4], boxsize = kwargs.get('box', None))
    #
    #
    # we may require at least 2 GB of available memory
    mem_available = virtual_memory().available * 1e-9
    if mem_available < 2.0:
        print('Found only {0:.2f} GB of available memory. This may not be '
              'sufficient for calculations.\nExiting...'.format(mem_available))
        exit(1)
    #
    #
    # call respective wrapper for neighbor search
    if query['type'] == 'neighbor':
        return _query_nearest(tree, query_points, query, types,
                              verbose = verbose)
    elif query['type'] == 'volume':
        return _query_volume(tree, query_points, query, types,
                             verbose = verbose)
#
#
#
#
def get_query_points(coords, **kwargs):
    # get optional keyword arguments
    distance    = kwargs.get('distance', None)
    is_periodic = (kwargs.get('box', None) is not None)
    margin      = kwargs.get('margin', None)
    #
    #
    # do some error checking for non-reasonable combination of options
    if is_periodic and margin is not None:
        print('ERROR: You cannot use the "--margin" option with periodic '
              'boundary conditions.', file = stderr)
        exit(1)
    if is_periodic == False and margin is None:
        print('ERROR: You must use the "--margin" option to exclude surface '
              'artifacts. (See "--help" for details.)', file = stderr)
        exit(1)
    #
    #
    #
    #
    # filter margin region if requested
    if margin is not None:
        coords = _filter_margin(coords, margin)
    #
    #
    #
    #
    # construct 3d grid if requested
    if distance is not None:
        if distance <= 0.0:
            print('ERROR: Distance ({0:.3f}) must be positive.'
                  .format(distance), file = stderr)
            exit(1)
        #
        #
        # get minimum and maximum positions for each direction
        if is_periodic:
            min_pos = [0.0, 0.0, 0.0]
            max_pos = kwargs.get('box')
        else:
            min_pos = np.amin(coords, axis = 0)
            max_pos = np.amax(coords, axis = 0)
        #
        #
        # construct 1d grid for each direction
        grid = []
        for i in range(0, 3):
            # number of grid points
            n_grid = int((max_pos[i] - min_pos[i]) / distance) + 1
            if n_grid <= 1:
                print('ERROR: Cannot construct grid. Separation too big?',
                      file = stderr)
                exit(1)
            #
            # separation between grid points
            delta = (max_pos[i] - min_pos[i]) / (n_grid - 1)
            #
            # construct grid
            if is_periodic:
                # exclude end point
                grid.append(np.linspace(
                    min_pos[i], max_pos[i] - delta, num = n_grid - 1))
            else:
                # include start and end point
                grid.append(np.linspace(min_pos[i], max_pos[i], num = n_grid))
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
################################################################################
#
# private module-level functions
#
################################################################################
def _filter_margin(coords, w):
    """Filter coordinates which have a distance of lower than :math:`w` to the
    boundaries.

    Parameters
    ----------
    coords : ndarray, shape (n, 3)
        The *n* three-dimensional coordinates.
    w : float
        The width :math:`w` of the margin region to exclude.

    Returns
    -------
    coords_filtered : ndarray, shape (m, 3)
        The *m* three-dimensional filtered coordinates.
    """
    #
    #
    if w <= 0.0:
        print('ERROR: Margin ({0:.3f}) must be positive.'.format(w),
              file = stderr)
        exit(1)
    #
    # set minimum and maximum positions
    min_pos = np.amin(coords, axis = 0)
    max_pos = np.amax(coords, axis = 0)
    #
    # filter query points
    return coords[
        (coords[:, 0] > min_pos[0] + w) &
        (coords[:, 1] > min_pos[1] + w) &
        (coords[:, 2] > min_pos[2] + w) &
        (coords[:, 0] < max_pos[0] - w) &
        (coords[:, 1] < max_pos[1] - w) &
        (coords[:, 2] < max_pos[2] - w)]
#
#
#
#
def _get_composition(indices, types):
    """Evaluate composition within a neighbor list.

    Parameters
    ----------
    indices: ndarray, shape (m,) or list of length m
        The indices of the *m* neighbors, i.e. the neighbor list.
    types : ndarray, shape(n,)
        The *n* atomic types.

    Returns
    -------
    n_2 : int
        The number of type 2 atoms in the neighbor list.
    """
    #
    #
    # select atomic subset, then sum types (subtraction of 1 effectively
    # counts only atoms of type 2
    return sum(types[indices] - 1)
#
#
#
#
def _query(tree, query_points, query, types):
    """Query neighbors.

    Depending on the value of ``type`` in the ``query`` dictionary argument,
    either the nearest neighbors (``neighbor``) or neighbors within a fixed
    distance/volume (``volume``) will be searched.

    Parameters
    ----------
    tree : cKDTree
        The cKDTree object.
    query_points: ndarray, shape (m, 3)
        The *m* three-dimensional query points for the search.
    query : dict
        The dictionary containing the query type and the query parameter (number
        of atoms or neighbor search cutoff).
    types : ndarray, shape(n,)
        The *n* atomic types.

    Returns
    -------
    r : ndarray, shape (m,)
        The *m* sphere radii, i.e. the distance to the furthest neighbor for the
        ``neighbor`` query type.
    n_2 : ndarray, shape (m,)
        The *m* numbers of type 2 atoms in the spheres.


    or

    Returns
    -------
    n : ndarray, shape (m,)
        The *m* numbers of total atoms in the spheres for the ``volume`` query
        type.
    n_2 : ndarray, shape (m,)
        The *m* numbers of type 2 atoms in the spheres.
    """
    #
    #
    # depending on the invocation mode, we need to search neighbors differently,
    # i.e. constant number or constant volume
    if query['type'] == 'neighbor':
        # query neighbors
        dists, indices = tree.query(
            query_points, k = query['param'], n_jobs = -1)
        #
        #
        # distances are sorted, so maximum distance is last entry;
        # create copy of maximum distances to allow freeing of full distance
        # array
        r_sphere = np.copy(dists[:, -1])
        dists    = None
    elif query['type'] == 'volume':
        # query neighbors
        indices = tree.query_ball_point(
            query_points, query['param'], n_jobs = -1)
    #
    #
    #
    #
    # we need to call _get_composition with a second argument, so we create a
    # partial object
    get_composition_partial_obj = partial(_get_composition, types = types)
    #
    #
    # calculate compositions in parallel
    pool = multiprocessing.Pool()
    # (setting an explicit value for the chunk size in pool.map() may reduce
    # memory usage)
    n_2 = np.asarray(pool.map(get_composition_partial_obj, indices))
    #
    # in volume mode, we also need to evaluate the number of neighbors
    if query['type'] == 'volume':
        neighbor_counts = np.asarray(pool.map(len, indices))
    #
    pool.close()
    pool.join()
    #
    # free full index array
    indices = None
    #
    #
    #
    #
    # return maximum neighbor counts and compositions
    if query['type'] == 'volume':
        return neighbor_counts, n_2
    # return maximum radii and compositions
    elif query['type'] == 'neighbor':
        return r_sphere, n_2
#
#
#
#
def _query_nearest(tree, query_points, query, types, **kwargs):
    """Query nearest neighbors.

    Before the neighbor search is performed, this method estimates the amount of
    memory needed for the search. If insufficient free memory is available, the
    neighbor search is split into smaller chunks so that approximately only half
    of the currently available memory is used.

    Parameters
    ----------
    tree : cKDTree
        The cKDTree object.
    query_points: ndarray, shape (m, 3)
        The *m* three-dimensional query points for the search.
    query : dict
        The dictionary containing the query type (``neighbor``) and the query
        parameter (number of atoms).
    types : ndarray, shape(n,)
        The *n* atomic types.

    Keyword Arguments
    -----------------
    verbose : bool
         Whether to be verbose. Default: ``False``.

    Returns
    -------
    r : ndarray, shape (m,)
        The *m* sphere radii, i.e. the distance to the furthest neighbor.
    n_2 : ndarray, shape (m,)
        The *m* numbers of type 2 atoms in the spheres.
    """
    #
    #
    # set verbosity
    verbose = kwargs.get('verbose', False)
    #
    #
    # estimated memory in GB (factor two accounts for indices and distances)
    mem_estimated = len(query_points) * query['param'] * 2 * 8 * 1e-9
    #
    # get available memory
    mem_available = virtual_memory().available * 1e-9
    #
    #
    # test for sufficient available memory
    if mem_estimated > _mem_threshold * mem_available:
        if verbose == True:
            print('NOTE: Estimated memory usage ({0:.2f} GB) exceeds {1:d}% of '
                  'available memory ({2:.2f} GB).'.format(
                      mem_estimated, int(_mem_threshold * 100), mem_available))
        #
        # set number of chunks
        chunks = int(np.ceil(mem_estimated / (_mem_threshold * mem_available)))
        if verbose == True:
            print('      Splitting problem into {0:d} chunks. (This may cause '
                  'overhead.)'.format(chunks))
        #
        #
        # initialize empty arrays
        dists        = np.array([], dtype = float)
        compositions = np.array([], dtype = int)
        #
        # split query points into smaller chunks
        if verbose == True:
            print('Searching neighbors and evaluating compositions ',
                  end = '', flush = True)
        for query_points_partial in np.array_split(query_points, chunks):
            if verbose == True:
                print('.', end = '', flush = True)
            #
            # get partial results
            dists_partial, compositions_partial = _query(
                tree, query_points_partial, query, types)
            #
            # append partial results
            dists        = np.append(dists,        dists_partial)
            compositions = np.append(compositions, compositions_partial)
        if verbose == True:
            print('')
        #
        # return complete results
        return dists, compositions
    else:
        # search all neighbors at once
        if verbose == True:
            print('Searching neighbors and evaluating compositions ...')
        return _query(tree, query_points, query, types)
#
#
#
#
def _query_volume(tree, query_points, query, types, **kwargs):
    """Query neighbors within cutoff distance.

    Before the neighbor search is performed, this method estimates the amount of
    memory needed for the search. If insufficient free memory is available, the
    neighbor search is split into smaller chunks. However, the
    ``query_ball_point()`` method over-allocates memory in a weird way so that a
    precise estimate is difficult. Here, a rather conservative approach is used.

    Parameters
    ----------
    tree : cKDTree
        The cKDTree object.
    query_points: ndarray, shape (m, 3)
        The *m* three-dimensional query points for the search.
    query : dict
        The dictionary containing the query type (``volume``) and the query
        parameter (neighbor search cutoff).
    types : ndarray, shape(n,)
        The *n* atomic types.

    Keyword Arguments
    -----------------
    verbose : bool
         Whether to be verbose. Default: ``False``.

    Returns
    -------
    n : ndarray, shape (m,)
        The *m* numbers of total atoms in the spheres.
    n_2 : ndarray, shape (m,)
        The *m* numbers of type 2 atoms in the spheres.
    """
    #
    #
    # set verbosity
    verbose = kwargs.get('verbose', False)
    #
    #
    # in order to estimate memory usage, we have to obtain the approximate
    # number of neighbors first with a sample set
    samples = min(1000, len(query_points))
    neighbor_list = tree.query_ball_point(
         query_points[0:samples], query['param'])
    #
    # estimated memory in GB (factor eight accounts for weird memory
    # over-allocation)
    mem_estimated = 8 * 1e-9 * len(query_points) / samples * \
        (getsizeof(neighbor_list) + sum(map(getsizeof, neighbor_list)))
    samples = None
    #
    #
    # get available memory
    mem_available = virtual_memory().available * 1e-9
    #
    #
    # test for sufficient available memory
    if mem_estimated > _mem_threshold * mem_available:
        if verbose == True:
            print('NOTE: Estimated memory usage ({0:.2f} GB) exceeds {1:d}% of '
                  'available memory ({2:.2f} GB).'
                  .format(mem_estimated, int(_mem_threshold * 100),
                          mem_available))
        #
        # set number of chunks
        chunks = int(np.ceil(mem_estimated / (_mem_threshold * mem_available)))
        if verbose == True:
            print('      Splitting problem into {0:d} chunks. (This may cause '
                  'overhead.)'.format(chunks))
        #
        #
        # initialize empty arrays
        neighbors    = np.array([], dtype = int)
        compositions = np.array([], dtype = int)
        #
        # split query points into smaller chunks
        if verbose == True:
            print('Searching neighbors and evaluating compositions ',
                  end = '', flush = True)
        for query_points_partial in np.array_split(query_points, chunks):
            if verbose == True:
                print('.', end = '', flush = True)
            #
            # get partial results
            neighbors_partial, compositions_partial = _query(
                tree, query_points_partial, query, types)
            #
            # append partial results
            neighbors    = np.append(neighbors, neighbors_partial)
            compositions = np.append(compositions, compositions_partial)
        if verbose == True:
            print('')
        #
        # return complete results
        return neighbors, compositions
    else:
        # search all neighbors at once
        if verbose == True:
            print('Searching neighbors and evaluating compositions ...')
        return _query(tree, query_points, query, types)
