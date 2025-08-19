"""
The APyT spatial distribution map (SDM) module
==============================================


Introduction
------------

Atom probe tomography (APT) reconstructions are known to introduce geometric
distortions, particularly affecting the fidelity of the crystal lattice. While
the resolution along the detection (:math:`z`) axis is typically sufficient to
resolve individual lattice planes—and to determine the correct
:math:`z`-scaling—the scaling of the lateral (:math:`x` and :math:`y`)
directions is less straightforward.

|Geiser| demonstrated that valuable crystallographic information can still be
extracted from lateral directions using **spatial distribution maps (SDMs)**.

An SDM is a two-dimensional histogram of interatomic distance vectors
:math:`\\Delta \\vec{r}_{ij} = (\\Delta x_{ij}, \\Delta y_{ij},
\\Delta z_{ij})^T`, where the lateral components :math:`\\Delta x_{ij}` and
:math:`\\Delta y_{ij}` are used as histogram axes. Depending on the crystal
structure and orientation, certain combinations :math:`(\\Delta x_{ij},
\\Delta y_{ij})` will occur more frequently, appearing as distinct maxima in the
histogram.

General Procedure
-----------------

As shown by |Geiser|, the accuracy of lateral spatial resolution improves when
atomic pairs are selected within a narrow separation along the :math:`z`-axis.
This requires optimal alignment of the sample such that the :math:`z`-axis
corresponds closely to a crystallographic direction. When constructed under
these conditions, SDMs reveal distinct (but distorted) maxima.

Since the ideal crystallographic positions of these maxima are known, a system
of linear equations can be formulated and solved to transform the distorted SDM
into its correct geometric representation. This transformation also allows for
the determination of the remaining :math:`z`-scaling factor.


List of functions
-----------------

This module provides several functions for generating, analyzing, and rectifying
SDMs from 3D APT data. These include tools for alignment optimization, SDM
construction, and spatial correction.

* :func:`lattice_vectors`: Get vectors which span the lattice planes.
* :func:`optimize_alignment`: Get optimal alignment in normal direction.
* :func:`rdf`: Analytic model of (one-dimensional) radial distribution function
  in normal direction.
* :func:`rdf_1d`: Create histogram of radial distribution function in normal
  direction.
* :func:`rdf_lateral`: Create histogram of radial distribution function for
  lateral directions.
* :func:`rectify_sdm`: Rectify SDMs (and atomic positions).
* :func:`rotate`: Rotate positions around two axes for alignment in normal
  direction.
* :func:`sdm`: Create SDMs.


.. |Geiser| raw:: html

   <a href="https://doi.org/10.1017/S1431927607070948" target="_blank">
   Geiser et al.</a>


.. sectionauthor:: Sebastian M. Eich <Sebastian.Eich@imw.uni-stuttgart.de>
.. codeauthor::    Sebastian M. Eich <Sebastian.Eich@imw.uni-stuttgart.de>
"""
#
#
#
#
__version__ = '0.1.0'
__all__ = [
    'lattice_vectors',
    'optimize_alignment',
    'rdf',
    'rdf_1d',
    'rdf_lateral',
    'rectify_sdm',
    'rotate',
    'sdm'
]
#
#
#
#
# import modules
import matplotlib.pyplot as plt
import numpy as np
#
# import some special functions
from lmfit import Model, Parameter
from mpl_toolkits.mplot3d import Axes3D
from numpy import exp, pi, sqrt
from scipy.ndimage import gaussian_filter
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation
from scipy.optimize import minimize
#
#
#
#
################################################################################
#
# private module-level variables
#
################################################################################
_dir_dict = {
    "x" : 0,
    "y" : 1,
    "z" : 2
}
"""dict : The dictionary mapping the Cartesian directions to their integer
          representation.
"""
#
#
_dir_lat_dict = {
    "x" : [1, 2],
    "y" : [2, 0],
    "z" : [0, 1]
}
"""dict : The dictionary mapping the Cartesian directions to their orthogonal
          directions represented as a list of integers.
"""
_dir_lat_str_dict = {
    "x" : ["y", "z"],
    "y" : ["z", "x"],
    "z" : ["x", "y"]
}
"""dict : The dictionary mapping the Cartesian directions to their orthogonal
          directions represented as a list of strings.
"""
#
#
#
#
################################################################################
#
# public functions
#
################################################################################
def rdf_1d(data, min, max, w, dir_key):
    """Calculate one-dimensional radial distribution function (RDF) for
    direction specified in *dir_key* and return histogram data points.

    Parameters
    ----------
    data : ndarray, shape (n, 3)
        The *n* three-dimensional coordinates.
    min : float
        The minimum distance which should be included in the RDF.
    max : float
        The maximum distance which should be included in the RDF.
    w : float
        The width used for binning the histogram data.
    dir_key : str
        The key indicating the Cartesian direction. Must be either ``"x"``,
        ``"y"``, or ``"z"``.

    Returns
    -------
    (x, y) : tuple
        The touple containing the bin centers *x* and the respective histogram
        counts *y*, both of type *ndarray* with *shape (m,)*.
    """
    #
    #
    # set direction from key
    dir = _dir_dict.get(dir_key)
    #
    # build one-dimensional k-d tree
    kd_tree = KDTree(np.reshape(data[:, dir], (-1, 1)))
    #
    # get pairs which are within cutoff
    pairs = kd_tree.query_pairs(max, output_type = "ndarray")
    #
    # calculate all pairwise distances
    r = np.abs(data[pairs[:, 0], dir] - data[pairs[:, 1], dir])
    #
    # calculate histogram
    return _make_hist(r, min, max, w)
#
#
#
#
def rdf_lateral(data, min, max, w, dir_key, r_0, δ):
    """Calculate lateral radial distribution function (RDF) and return histogram
    data points.

    The RDF is calculated for the directions which are orthogonal to the normal
    direction specified in *dir_key*, but the lateral RDF is restricted to all
    pairs for which the distance in normal direction :math:`r_\\mathrm n` is
    limited to the range :math:`r_0 - \\frac{\\delta}{2} \\leq r_\\mathrm n
    \\leq r_0 + \\frac{\\delta}{2}`, i.e. all pairs are chosen which fall in a
    narrow distance window centered at :math:`r_0` in normal direction.

    The lateral RDF may reveal peaks for highly precise atomic positions, but
    this is unlikely for APT data and rather applies to simulation data. The
    problem with the lateral RDF is that all angular, local in-plane information
    is averaged out, with the noise dominating the result. The idea behind the
    SDMs is the exactly the point to keep this local angular information,
    leading to distinct peaks in the SDMs, even with noisy data.

    Parameters
    ----------
    data : ndarray, shape (n, 3)
        The *n* three-dimensional coordinates.
    min : float
        The minimum distance which should be included in the RDF.
    max : float
        The maximum distance which should be included in the RDF.
    w : float
        The width used for binning the histogram data.
    dir_key : str
        The key indicating the normal Cartesian direction. Must be either
        ``"x"``, ``"y"``, or ``"z"``.
    r_0: float
        The center of the distance window in normal direction.
    δ : float
        The width of the distance window in normal direction.

    Returns
    -------
    (x, y) : tuple
        The touple containing the bin centers *x* and the respective histogram
        counts *y*, both of type *ndarray* with *shape (m,)*.
    """
    #
    #
    # set normal direction from key
    dir = _dir_dict.get(dir_key)
    #
    # set lateral directions
    dirs_lat = _dir_lat_dict.get(dir_key)
    #
    # build k-d tree for normal direction (normal direction is used for
    # filtering pairs)
    kd_tree = KDTree(np.reshape(data[:, dir], (-1, 1)))
    #
    # get pairs which are within cutoff in normal direction
    pairs = kd_tree.query_pairs(r_0 + δ / 2, output_type = "ndarray")
    #
    # calculate pairwise distances in normal direction
    r = np.abs(data[pairs[:, 0], dir] - data[pairs[:, 1], dir])
    #
    # only use pairs which are in the correct distance window in normal
    # direction
    pairs = pairs[abs(r - r_0) <= δ / 2]
    #
    # calculate lateral distances
    r_lat = np.sqrt(
        (data[pairs[:, 0], dirs_lat[0]] -
         data[pairs[:, 1], dirs_lat[0]])**2 +
        (data[pairs[:, 0], dirs_lat[1]] -
         data[pairs[:, 1], dirs_lat[1]])**2)
    #
    #
    # restrict lateral distances to cutoff
    r_lat = r_lat[(r_lat <= max)]
    #
    # calculate histogram
    return _make_hist(r_lat, min, max, w)
#
#
#
#
def rdf(r, N, n, w, sigma, d):
    """Analytic form of the radial distribution function (RDF).

    The RDF is modeled as

    .. math::
        \\mathrm{rdf}(r) = N\\, n\\, (n - 1)\\, w\\, f(r, 0, \\sigma)
               + \\sum_{i = 1}^{N - 1} (N - i)\\, n^2\\, w\\,
               f(r, i, d, \\sigma),

    where :math:`f(r, r_0, \\sigma)` is the Gaussian distribution centered at
    :math:`r_0` with standard deviation :math:`\\sigma`. The fist term
    represents the contribution from atomic pairs within the same atomic layer
    and the second term stems from the contributions of atoms in different
    atomic layers.

    Parameters
    ----------
    r : float
        The independent function argument.
    N : int
        The number of atomic layers in the evaluation volume.
    n : float
        The number of atoms per layer.
    w : float
        The width used for the histogram binning (needed for correct
        normalization).
    sigma : float
        The standard deviation of the Gaussian distributions.
    d : float
        The lattice spacing.

    Returns
    -------
    rdf(r) : float
        The RDF evaluated at *r*.
    """
    #
    #
    # contribution from atomic pairs in one single layer
    rdf = N * n * (n - 1) * w * _gaussian(r, 0, sigma)
    #
    # contribution from atomic pairs in different layers
    for i in range(1, N):
        rdf += (N - i) * n**2 * w * _gaussian(r, i * d, sigma)
    #
    return rdf
#
#
#
#
def rotate(data, angles, dir_key):
    """Rotate three-dimensional data by given Euler angles.

    Parameters
    ----------
    data : ndarray, shape (n, 3)
        The *n* three-dimensional coordinates.
    angles : ndarray, shape (2,)
        The two Euler angles for rotation around the first two axes. The last
        rotation around the normal axis is implicitly set to zero.
    dir_key : str
        The key indicating the normal Cartesian direction. The rotation around
        this axis will be the last one and is implicitly set to zero. Must be
        either ``"x"``, ``"y"``, or ``"z"``.

    Returns
    -------
    data_r : ndarray, shape (n, 3)
        The *n* three-dimensional coordinates after rotation.
    """
    #
    #
    # set sequence of axes for rotation
    axes_seq = '{0}{1}{2}'.format(*_dir_lat_str_dict.get(dir_key), dir_key)
    #
    # set rotation (rotation around normal axis is zero)
    r = Rotation.from_euler(axes_seq, np.concatenate((angles, [0])))
    #
    # return rotated data
    return r.apply(data[:])
#
#
#
#
def optimize_alignment(angles, data, rdf_params, rdf_model_params):
    """Align lattice planes in normal direction.

    For optimal evaluation of the spatial distribution maps, the lattice planes
    in normal direction should be aligned in an optimal way. This is
    accomplished with an optimization routine which minimizes the standard
    deviation of the Gaussian functions as described in :func:`rdf`, i.e. the
    lattice planes are aligned in such a way that the peaks in the radial
    distribution function (RDF) become the sharpest.

    First, the RDF will be generated with the parameters specified in
    `rdf_params` (see also :func:`rdf_1d`). Then, the RDF will be fitted with
    the analytical model using the (initial) parameters specified in
    `rdf_model_params` (see :func:`rdf`).

    A simplex algorithm is then used to find the minimal standard deviation by
    systematically varying the Euler angles around the two lateral axes (see
    also :func:`rotate`).

    Parameters
    ----------
    angles : ndarray, shape (2,)
        The initial Euler angles.
    data : ndarray, shape (n, 3)
        The *n* three-dimensional coordinates.
    rdf_params : tuple
        The parameters to generate the RDF as described in :func:`rdf_1d`
        (excluding *data*).
    rdf_model_params : tuple
        The parameters used for fitting the RDF as described in :func:`rdf`
        (excluding *r*). *N*, *n*, and *w* will be fixed and must be exact. The
        values for *sigma* and *d* should be approximate and will be used as
        initial values for fitting the RDF.

    Returns
    -------
    angles_opt : ndarray, shape (2,)
        The optimal Euler angles.
    """
    #
    #
    # print header for minimization progress
    print('Performing angular optimization...')
    print('{0:9s}   {1:9s}   {2:9s}'.format(' α', ' β', 'σ'))
    #
    #
    # perform minimization
    minimization_result = minimize(
        _get_rdf_std_dev, angles, args = (data, rdf_params, rdf_model_params),
        method = 'nelder-mead',
        options = {'xatol': 1e-2 / 180 * pi, 'disp': True, 'maxiter': 100})
    #
    #
    # return optimal angles
    return minimization_result.x
#
#
#
#
def lattice_vectors(data, angles, dir_key):
    """Get vectors which span the lattice planes **before** and **after**
    rotation.

    If the lattice planes are not perfectly aligned in normal direction, the box
    dimensions in lateral directions do not reflect the actual cross-sectional
    area covered by the lattice planes due to the inclination.

    This method accounts for this specific inclination and returns the vectors
    which span the lattice planes in lateral directions *before* and *after* the
    angular alignment (see :func:`optimize_alignment`). These vectors can then
    be used to calculate the actual cross-sectional area of the lattice planes
    with the aid of the cross product.

    Note that the value obtained by this correction may not differ significantly
    from the lateral box dimensions if the alignment of the lattice planes in
    normal direction is already sufficiently satisfied (i.e. very small Euler
    angles).

    Parameters
    ----------
    data : ndarray, shape (n, 3)
        The *n* three-dimensional coordinates **before** rotation (used to
        determine the **rectangular** box limits).
    angles : ndarray, shape (2,)
        The two Euler angles for rotation around the first two axes. The last
        rotation around the normal axis is implicitly set to zero.
    dir_key : str
        The key indicating the normal Cartesian direction. The rotation around
        this axis will be the last one and is implicitly set to zero. Must be
        either ``"x"``, ``"y"``, or ``"z"``.

    Returns
    -------
    B : ndarray, shape (3, 3)
        The vectors spanning the lattice planes **before** rotation, represented
        as a matrix with row vectors. The normal vector of the lattice planes is
        returned with unit length.
    B_r : ndarray, shape (3, 3)
        The vectors spanning the lattice planes **after** rotation, represented
        as a matrix with row vectors. The normal vector of the lattice planes is
        returned with unit length (and contains only one component in normal
        direction by definition).
    """
    #
    #
    # set normal and lateral directions
    n = _dir_dict.get(dir_key)
    l = _dir_lat_dict.get(dir_key)
    #
    # box size determined from maximum and minimum positions
    Δ = np.amax(data, axis = 0) - np.amin(data, axis = 0)
    #
    #
    # set sequence of axes for rotation
    axes_seq = '{0}{1}{2}'.format(*_dir_lat_str_dict.get(dir_key), dir_key)
    #
    # set rotation (rotation around normal axis is zero)
    r = Rotation.from_euler(axes_seq, np.concatenate((angles, [0])))
    #
    # set matrix representation of rotation
    R = r.as_matrix()
    #
    #
    # set normal unit vector of lattice planes *after* rotation/alignment
    e_n_rot = np.zeros(3)
    e_n_rot[n] = 1.0
    #
    #
    # set normal unit vector of lattice planes *before* rotation
    e_n = np.dot(np.linalg.inv(R), e_n_rot)
    #
    #
    # set vectors spanning lattice planes (i.e. lateral basis)
    # - the component in the respective lateral direction equals the box size in
    #   that direction
    # - the component of the complimentary lateral direction is zero (i.e. do
    #   not set)
    # - the normal component is defined through the dot product with the normal
    #   direction being zero
    B_lat = np.zeros((2, 3))
    for i in range(2):
        B_lat[i, l[i]] =  Δ[l[i]]
        B_lat[i, n]    = -Δ[l[i]] * e_n[l[i]] / e_n[n]
    #
    #
    # set basis vectors
    B = np.zeros((3, 3))
    B[n]    = e_n
    B[l[0]] = B_lat[0]
    B[l[1]] = B_lat[1]
    #
    #
    # print result of inclination correction
    print("Cross-section of box:           {0:.6e} Å²".
          format(Δ[l[0]] * Δ[l[1]]))
    print("Area spanned by lattice planes: {0:.6e} Å²".
          format(np.linalg.norm(np.cross(B[l[0]], B[l[1]]))))
    #
    #
    # return basis before and after rotation
    return B, np.dot(R, B.T).T
#
#
#
#
def sdm(data, max, n, dir_key, d_0, δ, plt_title, use_filter):
    """Calculate spatial distribution maps (SDMs) and return the results as a
    list of two-dimensional histogram data.

    The SDMs are calculated for the directions which are orthogonal to the
    normal direction specified in *dir_key*, but the (lateral) SDM is restricted
    to all pairs for which the distance in normal direction :math:`d` is limited
    to the range :math:`i\\, d_0 - \\frac{\\delta_j}{2} \\leq d \\leq
    i\\, d_0 + \\frac{\\delta_j}{2}`, i.e. all pairs are chosen which fall in a
    narrow distance window centered at :math:`i\\, d_0` in normal direction. The
    SDMs are created for :math:`i \\in (0,1,2,3)`, i.e. for pairs in the same
    layer (:math:`i = 0`) to pairs separated by up to three lattice spacings
    (:math:`i = 3`).

    Different window widths :math:`\\delta_j` in normal direction can be
    provided. For each :math:`\\delta_j`, a common plot is generated and shown
    for all :math:`i \\in (0,1,2,3)`.

    The two-dimensional histogram data of all processed SDMs is returned as a
    list.

    Parameters
    ----------
    data : ndarray, shape (n, 3)
        The *n* three-dimensional coordinates.
    max : float
        The maximum distance which should be included in the SDMs.
    n : int
        The number of bins used for the two-dimensional histogram data.
    dir_key : str
        The key indicating the normal Cartesian direction. Must be either
        ``"x"``, ``"y"``, or ``"z"``.
    d_0 : float
        The lattice spacing in normal direction.
    δ : list
        The list containing the different widths of the distance window in
        normal direction, each of type `float`.
    plt_title : str
        The plot title.
    use_filter : bool
        Whether to apply a filter to the SDMs.

    Returns
    -------
    [sdm_1, sdm_2, ...] : list
        The list containing the histogram data **(H, xedges, yedges)** of all
        processed SDMs.
    """
    #
    #
    # loop through slice thicknesses
    sdms = []
    for δ_cur in δ:
        print('Calculating SDMs for ΔΔ{0} = {1} Å...'.format(dir_key, δ_cur))
        #
        # set figure size (in inches)
        plt.figure(figsize = (12.8, 9.6))
        #
        # set plot title
        plt.suptitle('{0}\nΔΔ${1}$ = {2} Å'.format(plt_title, dir_key, δ_cur))
        #
        #
        # loop through slice separations
        for i in range(0, 4):
            plt.subplot(2, 2, i + 1)
            #
            # calculate SDM
            H, xedges, yedges = _sdm(data, max, n, dir_key, i * d_0, δ_cur)
            #
            # apply filter if requested
            if use_filter:
                # standard deviation for Gaussian kernel (might be tweaked)
                sigma = 1.5
                H = gaussian_filter(H, sigma)
            #
            # create plot
            plt.imshow(
                H.T, origin = 'lower',
                extent = (xedges[0], xedges[-1], yedges[0], yedges[-1])
            )
            plt.colorbar()
            plt.gca().set_title('Δ${0}$ = {1}$d$'.format(dir_key, i))
            plt.xlabel('Δ${0}$ (Å)'.format(_dir_lat_str_dict.get(dir_key)[0]))
            plt.ylabel('Δ${0}$ (Å)'.format(_dir_lat_str_dict.get(dir_key)[1]))
            #
            # append SDM
            sdms.append((H, xedges, yedges))
        #
        #
        # show plot
        plt.show()
    #
    #
    # return all processed SDMs as a list
    return sdms
#
#
#
#
def rectify_sdm(data, sdm_, dir_key, d, δ, plt_title, use_filter):
    """Use maxima of spatial distribution map (SDM) for data rectification.

    This method asks the user for the (approximate) positions of two linearly
    independent maxima in the SDM and performs two-dimensional Gaussian fits on
    these maxima to get accurate values for the centers of the maxima. By
    comparing these positions to the real crystallographic positions, a system
    of linear equations is solved to obtain the required transformation matrix
    which rectifies the SDM (and the atomic positions).

    Parameters
    ----------
    data : ndarray, shape (n, 3)
        The *n* three-dimensional coordinates.
    sdm : tuple
        The SDM histogram data **(H, xedges, yedges)**, as an element of the
        list returned by :func:`sdm`.
    dir_key : str
        The key indicating the normal Cartesian direction. Must be either
        ``"x"``, ``"y"``, or ``"z"``.
    d : float
        The current lattice spacing in normal direction which is required for
        correct scaling. Should be determined first from fitting the radial
        distribution function in :func:`rdf`.
    δ : list
        The list containing the different widths of the distance window in
        normal direction, each of type `float`. Required for the calculation of
        the rectified SDM(s) (see :func:`sdm` for details).
    plt_title : str
        The title used in generated plots.
    use_filter : bool
        Whether to apply a filter to the rectified SDMs.

    Returns
    -------
    T : ndarray, shape(3, 3)
        The transformation matrix for data rectification.
    data_rect : ndarray, shape (n, 3)
        The rectified *n* three-dimensional coordinates.
    [sdm_1, sdm_2, ...] : list
        The list containing the histogram data **(H, xedges, yedges)** of all
        rectified SDMs (see :func:`sdm` for details).
    """
    #
    #
    # set SDM cutoff and number of bins from bin edges
    max = sdm_[1][-1]
    n   = sdm_[1].shape[0] - 1
    #
    # set string for lateral directions in user input
    lat_dirs = "({0}, {1})".format(
        _dir_lat_str_dict.get(dir_key)[0],
        _dir_lat_str_dict.get(dir_key)[1])
    #
    #
    #
    #
    # fit first SDM maximum
    while True:
        print('')
        while True:
            try:
                x1_user, y1_user = [float(s) for s in input(
                    "Enter {0} of first SDM maximum (space-separated): "
                    .format(lat_dirs)).split()]
                break
            except:
                pass
        #
        while True:
            try:
                r = float(input("Enter fit radius: "))
                break
            except:
                pass
        #
        # get center of 2d Gaussian fit
        x1_fit, y1_fit = _get_gaussian_center(
            sdm_, x1_user, y1_user, r, dir_key, plt_title)
        #
        # check for re-fit
        if input("Re-fit maximum? (y/n): ") == 'n':
            break
    #
    #
    #
    #
    # fit second SDM maximum
    while True:
        print('')
        while True:
            try:
                x2_user, y2_user = [float(s) for s in input(
                    "Enter {0} of second SDM maximum (space-separated): "
                    .format(lat_dirs)).split()]
                break
            except:
                pass
        #
        while True:
            try:
                r = float(input("Enter fit radius: "))
                break
            except:
                pass
        #
        # get center of 2d Gaussian fit
        x2_fit, y2_fit = _get_gaussian_center(
            sdm_, x2_user, y2_user, r, dir_key, plt_title)
        #
        # check for re-fit
        if input("Re-fit maximum? (y/n): ") == 'n':
            break
    #
    #
    #
    #
    # get real crystallographic maxima and lattice spacing
    print('')
    while True:
        try:
            x1, y1 = [float(s) for s in input(
                "Enter {0} of first  real crystallographic maximum "
                "(space-separated): ".format(lat_dirs)).split()]
            break
        except:
            pass
    while True:
        try:
            x2, y2 = [float(s) for s in input(
                "Enter {0} of second real crystallographic maximum "
                "(space-separated): ".format(lat_dirs)).split()]
            break
        except:
            pass
    while True:
        try:
            d_0 = float(input("Enter real lattice spacing in {0}-direction: "
                  .format(dir_key)))
            break
        except:
            pass
    #
    #
    #
    #
    # set coefficient matrix for system of linear equations
    C = np.array([
        [x1_fit, y1_fit, 0,      0],
        [0,      0,      x1_fit, y1_fit],
        [x2_fit, y2_fit, 0,      0],
        [0,      0,      x2_fit, y2_fit]])
    #
    # set column vector with real crystallographic positions
    b = np.array([x1, y1, x2, y2])
    #
    # solve system of linear equations
    x = np.linalg.solve(C, b)
    #
    #
    # set transformation matrix
    M = np.array([
        [x[0], x[1], 0],
        [x[2], x[3], 0],
        [0,    0,    d_0 / d]])
    #
    # rectify positions
    data_r = np.dot(M, data.T).T
    #
    # calculate rectified SDM
    print('')
    sdms_r = sdm(data_r, max, n, dir_key, d_0, δ, plt_title, use_filter)
    #
    #
    # return transformation matrix, rectified positions, and rectified SDMs
    return M, data_r, sdms_r
#
#
#
#
################################################################################
#
# private module-level functions
#
################################################################################
def _gaussian(x, μ, σ):
    """The (normalized) Gaussian distribution.

    The Gaussian distribution is given by

    .. math::
        f(x) = \\frac{1}{\\sigma \\sqrt{2\\pi}}
               \\exp\\bigg(-\\frac 1 2\\frac{(x - \\mu)^2}{\\sigma^2}\\bigg),

    where :math:`\\mu` and :math:`\\sigma` are the expectation value and the
    standard deviation, respectively, of the Gaussian distribution.

    Parameters
    ----------
    x : float
        The independent function argument.
    μ : float
        The expectation value of the Gaussian distribution.
    σ : float
        The standard deviation of the Gaussian distribution.

    Returns
    -------
    f(x) : float
        The Gaussian distribution evaluated at *x*.
    """
    #
    #
    return 1.0 / (σ * sqrt(2.0 * pi)) * exp(-0.5 * ((x - μ) / σ)**2)
#
#
#
#
def _make_hist(data, min, max, w):
    """Wrapper for the NumPy histogram function.

    This wrapper takes the bin width *w* rather than the number of bins as an
    argument. Also, instead of returning the bin edges, the bin centers are
    returned.

    Parameters
    ----------
    data : ndarray, shape (n,)
        The *n* data samples.
    min : float
        The minimum value which should be included in the histogram.
    max : float
        The maximum value which should be included in the histogram.
    w : float
        The width used for binning the histogram data.

    Returns
    -------
    (x, y) : tuple
        The touple containing the bin centers *x* and the respective histogram
        counts *y*, both of type *ndarray* with *shape (m,)*.
    """
    #
    #
    # calculate histogram
    hist, edges = np.histogram(data, bins = round((max - min) / w),
                               range = (min, max))
    #
    # calculate bin centers
    centers = (edges[:-1] + edges[1:]) / 2
    #
    # return histogram
    return (centers, hist)
#
#
#
#
def _get_rdf_std_dev(angles, data, rdf_params, rdf_model_params):
    """Calculate standard deviation of the Gaussian functions used in the
    one-dimensional radial distribution function (RDF) for given rotation
    angles.

    Parameters
    ----------
    angles : ndarray, shape (2,)
        The Euler angles.
    data : ndarray, shape (n, 3)
        The *n* three-dimensional coordinates.
    rdf_params : tuple
        The parameters to generate the RDF as described in :func:`rdf_1d`
        (excluding *data*).
    rdf_model_params : tuple
        The parameters used for fitting the RDF as described in :func:`rdf`
        (excluding *r*). *N*, *n*, and *w* will be fixed and must be exact. The
        values for *sigma* and *d* should be approximate and will be used as
        initial values for fitting the RDF.

    Returns
    -------
    sigma : float
        The standard deviation used in the RDF.
    """
    #
    #
    # rotate data (normal direction is specified in rdf_params[-1])
    r_data = rotate(data, angles, rdf_params[-1])
    #
    # calculate RDF
    rdf_data = rdf_1d(r_data, *rdf_params)
    #
    #
    # initialize fit
    model = Model(rdf)
    params = model.make_params()
    params['N']     = Parameter(name = 'N', value = rdf_model_params[0],
                                vary = False)
    params['n']     = Parameter(name = 'n', value = rdf_model_params[1],
                                vary = False)
    params['w']     = Parameter(name = 'w', value = rdf_model_params[2],
                                vary = False)
    params['sigma'] = Parameter(name = 'sigma',value = rdf_model_params[3])
    params['d']     = Parameter(name = 'd', value = rdf_model_params[4])
    #
    # perform fit
    rdf_fit = model.fit(rdf_data[1], r = rdf_data[0], params = params)
    #
    #
    # print current parameters and function value
    print('{0:+1.6f}   {1:+1.6f}   {2:1.6f}'.format(
        angles[0] * 180 / pi, angles[1] * 180 / pi,
        rdf_fit.params['sigma'].value))
    #
    #
    # return standard deviation from fit
    return rdf_fit.params['sigma'].value
#
#
#
#
def _sdm(data, max, n, dir_key, r0, δ):
    """Calculate spatial distribution map (SDM) and return two-dimensional
    histogram data.

    The SDM is calculated for the directions which are orthogonal to the normal
    direction specified in *dir_key*, but the (lateral) SDM is restricted to all
    pairs for which the distance in normal direction :math:`r_\\mathrm n` is
    limited to the range :math:`r_0 - \\frac{\\delta}{2} \\leq r_\\mathrm n
    \\leq r_0 + \\frac{\\delta}{2}`, i.e. all pairs are chosen which fall in a
    narrow distance window centered at :math:`r_0` in normal direction.

    Parameters
    ----------
    data : ndarray, shape (n, 3)
        The *n* three-dimensional coordinates.
    max : float
        The maximum distance which should be included in the SDM.
    n : int
        The number of bins used for the two-dimensional histogram data.
    dir_key : str
        The key indicating the normal Cartesian direction. Must be either
        ``"x"``, ``"y"``, or ``"z"``.
    r0: float
        The center of the distance window in normal direction.
    δ : float
        The width of the distance window in normal direction.

    Returns
    -------
    H : ndarray, shape(n, n)
        The bi-dimensional histogram of the SDM samples.
    xedges : ndarray, shape(n+1,)
        The bin edges along the first dimension.
    yedges : ndarray, shape(n+1,)
        The bin edges along the second dimension.
    """
    #
    #
    # set normal direction from key
    dir = _dir_dict.get(dir_key)
    #
    # set lateral directions
    dirs_lat = _dir_lat_dict.get(dir_key)
    #
    # build k-d tree for normal direction (normal direction is used for
    # filtering pairs)
    kd_tree = KDTree(np.reshape(data[:, dir], (-1, 1)))
    #
    # get pairs which are within cutoff in normal direction
    pairs = kd_tree.query_pairs(r0 + δ / 2, output_type = "ndarray")
    #
    # calculate pairwise distances in normal direction
    r = np.abs(data[pairs[:, 0], dir] - data[pairs[:, 1], dir])
    #
    # only use pairs which are in the correct distance window in normal
    # direction
    pairs = pairs[abs(r - r0) <= δ / 2]
    #
    # calculate lateral (directed) distances (SDM)
    sdm = np.vstack((
        data[pairs[:, 0], dirs_lat[0]] - data[pairs[:, 1], dirs_lat[0]],
        data[pairs[:, 0], dirs_lat[1]] - data[pairs[:, 1], dirs_lat[1]])).T
    #
    # restrict SDM to lateral cutoff
    sdm = sdm[(abs(sdm[:, 0]) <= max) & (abs(sdm[:, 1]) <= max)]
    #
    # SDMs need pairs (i, j) and (j, i), so just append inverted values
    sdm = np.vstack((sdm, -sdm[:, :]))
    #
    # calculate two-dimensional histogram
    return np.histogram2d(sdm[:, 0], sdm[:, 1], bins = n,
                          range = [[-max, max], [-max, max]])
#
#
#
#
def _get_gaussian_center(sdm, x0, y0, r, dir_key, plt_title):
    """Fit two-dimensional Gaussian function to a local maximum of the spatial
    distribution map (SDM).

    The fit will be performed within radius `r` centered at the position given
    by (`x0`, `y0`).

    Parameters
    ----------
    sdm : tuple
        The SDM histogram data **(H, xedges, yedges)**, as an element of the
        list returned by :func:`sdm`.
    x0 : float
        The approximate position of the maximum along the first dimension.
    y0 : float
        The approximate position of the maximum along the second dimension.
    r : float
        The radius around the maximum used for fitting.
    dir_key : str
        The key indicating the normal Cartesian direction. Must be either
        ``"x"``, ``"y"``, or ``"z"``.
    plt_title : str
        The title used in generated plots.

    Returns
    -------
    x0 : float
        The center of the two-dimensional Gaussian along the first dimension, as
        obtained by the fit.
    y0 : float
        The center of the two-dimensional Gaussian along the second dimension,
        as obtained by the fit.
    """
    #
    #
    # set SDM cutoff and number of bins from bin edges
    max = sdm[1][-1]
    n   = sdm[1].shape[0] - 1
    #
    #
    #
    #
    # define two-dimensional Gaussian model function for fitting local maximum
    model = Model(_gaussian_2d, independent_vars = ['x', 'y'])
    #
    # initialize fitting parameters
    params = model.make_params(
        A = 10, x0 = x0, y0 = y0, a = 1.0, b = 0.0, c = 1.0, z0 = 1.0)
    #
    #
    # get local SDM data around maximum
    x, y, z = _get_local_hist_data(sdm, x0, y0, r)
    #
    #
    # perform fit and print results
    fit = model.fit(z, params, x = x, y = y)
    x0_fit = fit.params['x0'].value
    y0_fit = fit.params['y0'].value
    print('\nFit results for Gaussian centered at ({0:.2f}, {1:.2f}):'
          .format(x0_fit, y0_fit))
    print(fit.fit_report() + '\n')
    #
    #
    # set grid for wireframe
    X, Y = np.meshgrid(
        np.linspace(x0_fit - r, x0_fit + r, int(n * r / max)),
        np.linspace(y0_fit - r, y0_fit + r, int(n * r / max)))
    #
    #
    # plot data and fit
    fig = plt.figure()
    plt.suptitle('{0}'.format(plt_title))
    ax = plt.axes(projection = '3d')
    ax.scatter(x, y, z, c = z, cmap = 'viridis')
    ax.plot_wireframe(X, Y, fit.eval(x = X, y = Y), color = 'black')
    ax.set_xlabel('Δ${0}$ (Å)'.format(_dir_lat_str_dict.get(dir_key)[0]))
    ax.set_ylabel('Δ${0}$ (Å)'.format(_dir_lat_str_dict.get(dir_key)[1]))
    ax.set_zlabel('Counts')
    plt.show()
    #
    #
    #
    #
    return x0_fit, y0_fit
#
#
#
#
def _gaussian_2d(x, y, A, x0, y0, a, b, c, z0):
    """Analytic form of a general two-dimensional Gaussian function.

    The general form of the two-dimensional Gaussian function is given

    .. math::
        f(x, y) = z_0 + A \\exp\\Big(-\\big(  a (x - x_0)^2 +
                                            2 b (x - x_0) (y - y_0) +
                                              c (y - y_0)^2\\big)\\Big).

    Returns
    -------
    f(x, y) : float
        The two-dimensional Gaussian function evaluated at :math:`(x, y)`.
    """
    #
    #
    return z0 + A * exp(-(    a * (x - x0)**2 +
                          2 * b * (x - x0) * (y - y0) +
                              c * (y - y0)**2))
#
#
#
#
def _get_local_hist_data(sdm, x0, y0, r):
    """Select histogram counts centered at :math:`(x_0, y_0)` within radius
    :math:`r`.

    The three-dimensional data is returned as flattened arrays `x`, `y`, and
    `z`.

    Parameters
    ----------
    sdm : tuple
        The SDM histogram data **(H, xedges, yedges)**, as an element of the
        list returned by :func:`sdm`.
    x0 : float
        The center along the first dimension.
    y0 : float
        The center along the second dimension.
    r : float
        The radius around the center used for filtering.

    Returns
    -------
    x : ndarray, shape(n,)
        The positions along the first dimension.
    y : ndarray, shape(n,)
        The positions along the second dimension.
    z : ndarray, shape(n,)
        The local histogram counts.
    """
    #
    #
    # get histogram data and bin edges
    H, xedges, yedges = sdm
    #
    #
    # calculate bin centers
    xcenters = (xedges[:-1] + xedges[1:]) / 2.0
    ycenters = (yedges[:-1] + yedges[1:]) / 2.0
    #
    # return coordinate matrices from coordinate vectors
    x, y = np.meshgrid(xcenters, ycenters, indexing = 'ij')
    #
    #
    # flatten arrays
    x = x.flatten()
    y = y.flatten()
    H = H.flatten()
    #
    #
    # select valid histogram data within radius (other points default to zero)
    x_sel = x[((x - x0)**2 + (y - y0)**2 <= r**2)]
    y_sel = y[((x - x0)**2 + (y - y0)**2 <= r**2)]
    H_sel = H[((x - x0)**2 + (y - y0)**2 <= r**2)]
    #
    #
    # return 3d data
    return x_sel, y_sel, H_sel
