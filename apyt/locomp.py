"""
The APyT local composition module
=================================

This module enables the evaluation of local compositions within spherical (sub-)
volumes of any three-dimensional data set which includes additional information
on the atomic types. In principle, there are two mutually excluding methods how
to evaluate the local compositions:

1. Using spheres with a constant number of particles/neighbors.
2. Using spheres with a constant radius/volume.

Either of these methods is supported and determined through a dictionary, which
specifies the ``type`` (``"neighbor"`` or ``"volume"``) and the corresponding
neighbor search parameter ``param`` which must be set to the number of neighbors
or sphere radius, respectively.

The spherical (sub-) volumes can be placed around every atom or the sphere
centers can be arranged automatically to represent a three-dimensional regular
grid, where a minimum separation between the grid points is ensured to exclude
overlapping of the spheres (see :meth:`get_query_points` for details).

Neighbor searches are preformed with the |SciPy_cKDTree| class from the SciPy
package.


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

* :meth:`calc_stats`: Calculate histogram and statistics.
* :meth:`check_periodic_box`: Check periodic boundary conditions.
* :meth:`emulate_efficiency`: Emulate detector efficiency for simulated data.
* :meth:`get_composition`: Get local compositions for query points.
* :meth:`get_margin_filter`: Automatically filter margin region.
* :meth:`get_query_points`: Get query points for neighbor search.


.. |SciPy_cKDTree| raw:: html

    <a href="https://docs.scipy.org/doc/scipy/reference/generated/
    scipy.spatial.cKDTree.html" target="_blank">cKDTree</a>


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
    'emulate_efficiency',
    'get_composition',
    'get_margin_filter',
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
_mem_threshold = 0.50
"""float : The approximate maximum amount of available memory to use."""
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
    """Calculate histogram and evaluate histogram statistics.

    Parameters
    ----------
    data : ndarray, shape (n,)
        The data over which to calculate the histogram.

    Keyword Arguments
    -----------------
    bin_method : str
         The method to use to determine the bin width for the histogram. This
         argument will be passed through to the ``numpy.histogram()`` method.
         Default: ``None``, which evaluates to a bin width of ``1.0``.
    correlation : int
         The correlation factor of the samples. If the samples are correlated,
         the statistical error estimates will be underestimated, but can be
         corrected with this option. Default: ``1``, i.e. all samples are
         uncorrelated.

    Returns
    -------
    (hist, edges, centers, hist_norm) : tuple
        The tuple containing the respective histogram counts *hist*, the bin
        edges *edges*, the bin centers *centers*, and the normalized histogram
        counts *hist_norm*, each of type *ndarray, shape (m,)* or
        *shape (m+1,)*, respectively.
    (μ, Δμ, Var, ΔVar, Var/μ, Δ(Var/μ)) : tuple
        The statistical histogram data including error estimates, each of type
        *float*.
    """
    #
    #
    # get binning_method
    bin_method = kwargs.get('bin_method', None)
    #
    #
    # set data properties
    num      = len(data)
    data_min = min(data)
    data_max = max(data)
    #
    #
    # calculate histogram
    if bin_method is None:
        # use bin width of 1.0
        bins = np.linspace(
            data_min - 0.5, data_max + 0.5, data_max - data_min + 2)
        bin_centers = np.arange(data_min, data_max + 1)
        #
        counts, bin_edges = np.histogram(data, bins = bins)
    elif isinstance(bin_method, str):
        # pass through binning method
        counts, bin_edges = np.histogram(data, bins = bin_method)
        #
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    #
    #
    # normalize histogram
    counts_norm = counts / (num * (bin_edges[1:] - bin_edges[:-1]))
    #
    #
    # calculate central moments
    mu   = sum(counts *  bin_centers)          / num
    mu_2 = sum(counts * (bin_centers - mu)**2) / num
    mu_4 = sum(counts * (bin_centers - mu)**4) / num
    #
    #
    # set number of independent samples (may be lower than actual sample size
    # due to user-defined correlation);
    # using int type may result in integer overflow below; convert to float64
    n = np.float64(num / kwargs.get('correlation', 1))
    #
    #
    # calculate central moment estimates (correction does not matter for
    # sufficiently large n)
    # https://mathworld.wolfram.com/h-Statistic.html
    h_2 = n * mu_2 / (n - 1)
    h_4 = (3 * (3 - 2 * n) * n**2 * mu_2**2 + \
           n**2 * (n**2 - 2 * n + 3) * mu_4) / \
          ((n - 3) * (n - 2) * (n - 1) * n)
    #
    # calculate estimate of relative variance
    h_2_r = h_2 / mu
    #
    #
    # calculate statistical errors; use fabs to catch floating point errors
    # https://stats.stackexchange.com/a/157305
    Δmu    = np.sqrt(h_2 / n)
    Δh_2   = np.sqrt(np.fabs((h_4 - (n - 3) / (n - 1) * h_2**2) / n))
    Δh_2_r = np.sqrt(Δh_2**2 + h_2_r**2 * Δmu**2) / mu
    #
    #
    # return histogram data and statistics
    return (counts, bin_edges, bin_centers, counts_norm), \
           (mu, Δmu, h_2, Δh_2, h_2_r, Δh_2_r)
#
#
#
#
def check_periodic_box(comment):
    """Check periodic boundary conditions.

    This method checks an OVITO comment line for the presence of the periodic
    boundary flags. If found, this method returns the (periodic) box dimensions,
    otherwise ``None`` is returned.

    Parameters
    ----------
    comment : str
        A valid OVITO comment line from a file in *xyz* format.

    Returns
    -------
    box : ndarray, shape (3,) or None
        The box dimensions for periodic boundary conditions.
    """
    #
    #
    # split comment line into single tokens
    comment = comment.split()
    #
    # set default return value
    box = None
    #
    #
    # check pbc flag
    try:
        pbc_index = comment.index('pbc')
        if comment[pbc_index + 1] == '1' and \
           comment[pbc_index + 2] == '1' and \
           comment[pbc_index + 3] == '1':
            #
            # get box dimensions
            box = np.zeros(3)
            for i in range(0, 3):
                try:
                    # look for dimension
                    pos = comment.index('cell_vec' + str(i + 1))
                    box[i] = comment[pos + i + 1]
                except:
                    print('ERROR: Could not find box size for dimension {0:d}.'
                    .format(i + 1), file = stderr)
                    exit(1)
    except:
        pass
    return box
#
#
#
#
def emulate_efficiency(data, p):
    """Emulate detector efficiency for simulated data.

    For data prior to evaporation and reconstruction, the limited detector
    efficiency can be emulated for a simulated data set by randomly choosing
    the particles with a certain probability which corresponds to the detection
    efficiency ``p``.

    Parameters
    ----------
    data : ndarray, shape (n, 4)
        The *n* types and three-dimensional coordinates of the atoms.
    p : float
        The detector efficiency to emulate.

    Returns
    -------
    data_r : ndarray, shape (m, 4)
        The *m* types and three-dimensional coordinates of the randomly selected
        atoms.
    """
    #
    #
    # initialize random number generator with specific seed to avoid correlated
    # results for similar efficiencies
    rng = np.random.default_rng(int(p * 100))
    #
    # draw random number for each particle
    random_numbers = rng.random(len(data))
    #
    # return randomly selected particles
    return data[(random_numbers <= p)]
#
#
#
#
def get_composition(data, query_points, query, **kwargs):
    """Get local compositions for query points.

    Depending on the value of ``type`` in the **query** dictionary argument,
    either the nearest neighbors (``"neighbor"``) or neighbors within a fixed
    distance/volume (``"volume"``) will be searched for the provided query
    points. The composition in terms of number of type 2 particles will be
    returned for each query point. In addition, either the sphere radii or the
    total number of atoms in the spheres will be returned.

    Parameters
    ----------
    data : ndarray, shape (n, 4)
        The *n* types and three-dimensional coordinates of the atoms.
    query_points: ndarray, shape (m, 3)
        The *m* three-dimensional query points for the search.
    query : dict
        The dictionary containing the query type (``"neighbor"`` or
        ``"volume"``)
        and the query parameter ``param`` (number of neighbors or neighbor
        search cutoff).

    Keyword Arguments
    -----------------
    box : ndarray, shape (3,) or None
        The periodic box dimensions (if present).
    verbose : bool
        Whether to be verbose. Default: ``False``.

    Returns
    -------
    r : ndarray, shape (m,)
        The *m* sphere radii, i.e. the distance to the furthest neighbor for the
        ``"neighbor"`` query type.
    n_2 : ndarray, shape (m,)
        The *m* numbers of type 2 atoms in the spheres.


    or

    Returns
    -------
    n : ndarray, shape (m,)
        The *m* numbers of total atoms in the spheres for the ``"volume"`` query
        type.
    n_2 : ndarray, shape (m,)
        The *m* numbers of type 2 atoms in the spheres.
    """
    #
    #
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
        print('ERROR: Found only {0:.2f} GB of available memory. This may not '
              'be sufficient for calculations.\nExiting...'
              .format(mem_available), file = stderr)
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
def get_margin_filter(query_points, r, box_l, box_u, threshold = 1.1):
    """Automatically filter margin region.

    Query points close to the surface in the margin region should not be
    included in the evaluation since the nearest neighbors will typically not be
    confined in a spherical volume, but rather in a truncated sphere. Only if
    the distance of the query point to the surface is sufficiently large, the
    maximum nearest neighbor distance becomes equal to the sphere radius. The
    critical condition is reached if the distance of the query point is
    identical to the maximum nearest neighbor distance. By default, an
    additional buffer of 10% is included, which can be controlled by the
    optional ``threshold`` argument.

    Parameters
    ----------
    query_points : ndarray, shape (n, 3)
        The *n* three-dimensional query points.
    r : ndarray, shape (n,)
        The *n* sphere radii (maximum nearest neighbor distances).
    box_l : ndarray, shape (3,)
        The lower box boundary.
    box_u : ndarray, shape (3,)
        The upper box boundary.
    threshold : float
        The optional threshold used for filtering. Defaults to 1.1.

    Returns
    -------
    mask : ndarray, shape (n,)
        The boolean mask indicating which query points do **not**
        belong to the margin region.
    """
    #
    #
    # calculate distance of each query point to surface (first distance to
    # surface in each dimension, then minimum of all dimensions)
    print("Calculating filter mask for query points in margin region ...")
    d = np.amin(np.minimum(query_points - box_l, box_u - query_points),
                axis = 1)
    #
    #
    # create filter mask (only use query points where distance of 'sphere'
    # center from surface is greater than 'sphere' radius including additional
    # threshold)
    mask = (d >= threshold * r)
    print("Margin region contains {0:d} query points."
          .format(np.count_nonzero(~mask)))
    #
    #
    # return filter mask
    return mask
#
#
#
#
def get_query_points(coords, **kwargs):
    """Get query points for neighbor search.

    This method can be used to specify the query points for the neighbor search.
    If no additional argument is provided, all atomic positions will be used as
    query points. The ``margin`` keyword argument can be used to exclude surface
    artifacts. With the ``distance`` keyword argument, a minimum separation
    between the query points is ensured, which is achieved by the construction
    of a regular three-dimensional grid. For periodic boxes, the box dimensions
    should be passed using the ``box`` keyword argument.

    Parameters
    ----------
    coords : ndarray, shape (n, 3)
        The *n* three-dimensional coordinates.

    Keyword Arguments
    -----------------
    box : ndarray, shape (3,) or None
        The periodic box dimensions (if present).
    distance : float
        The (minimum) separation between the query points.
    margin : float
        The width of the margin region to exclude for the query points.

    Returns
    -------
    query_points : ndarray, shape (m, 3)
        The *m* three-dimensional query points for the neighbor search.
    """
    #
    #
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
    # count type 2 atoms
    return np.count_nonzero(types[indices] == 2)
#
#
#
#
def _query(tree, query_points, query, types):
    """Query neighbors.

    Depending on the value of ``type`` in the **query** dictionary argument,
    either the nearest neighbors (``"neighbor"``) or neighbors within a fixed
    distance/volume (``"volume"``) will be searched.

    Parameters
    ----------
    tree : cKDTree
        The cKDTree object.
    query_points: ndarray, shape (m, 3)
        The *m* three-dimensional query points for the search.
    query : dict
        The dictionary containing the query type and the query parameter (number
        of neighbors or neighbor search cutoff).
    types : ndarray, shape(n,)
        The *n* atomic types.

    Returns
    -------
    r : ndarray, shape (m,)
        The *m* sphere radii, i.e. the distance to the furthest neighbor for the
        ``"neighbor"`` query type.
    n_2 : ndarray, shape (m,)
        The *m* numbers of type 2 atoms in the spheres.


    or

    Returns
    -------
    n : ndarray, shape (m,)
        The *m* numbers of total atoms in the spheres for the ``"volume"`` query
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
            query_points, k = query['param'], workers = -1)
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
            query_points, query['param'], workers = -1, return_sorted = False)
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
        The dictionary containing the query type (``"neighbor"``) and the query
        parameter (number of neighbors).
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
        The dictionary containing the query type (``"volume"``) and the query
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
