"""
The APyT basic reconstruction module
====================================

This module provides methods for performing three-dimensional reconstructions
of atom probe tomography (APT) data using the *classic* and *taper* geometry
models.

It is intended for standard reconstruction scenarios where the tip geometry
can be approximated by well-defined analytical models. Both reconstruction
approaches rely on geometric assumptions about the specimen shape, field
evaporation behavior, and projection parameters.


List of functions
-----------------

The following functions are provided for tip geometry calculation, evaporation
field estimation, and APT data reconstruction:

* :func:`align_evaporation_field`: Automatically align evaporation field for
  taper geometry.
* :func:`enable_debug`: Enable or disable debug output.
* :func:`get_evaporation_field`: Calculate evaporation field.
* :func:`get_geometry_classic`: Calculate tip geometry through 'classic' scheme.
* :func:`get_geometry_taper`: Calculate taper geometry.
* :func:`reconstruct`: Reconstruct :math:`xyz` tip positions.


.. sectionauthor:: Sebastian M. Eich <Sebastian.Eich@imw.uni-stuttgart.de>
.. codeauthor::    Sebastian M. Eich <Sebastian.Eich@imw.uni-stuttgart.de>
"""
#
#
#
#
__version__ = '0.1.0'
__all__ = [
    'align_evaporation_field',
    'enable_debug',
    'get_evaporation_field',
    'get_geometry_classic',
    'get_geometry_taper',
    'reconstruct'
]
#
#
#
#
# import modules
import numba
import numpy as np
#
# import some special functions/modules
from inspect import getframeinfo, stack
from scipy.optimize import minimize
from sys import stderr
#
#
#
#
# set Numba configuration for parallelization
#numba.config.THREADING_LAYER = 'omp'
#numba.set_num_threads(4)
#
#
#
#
################################################################################
#
# private module-level variables
#
################################################################################
_is_dbg = False
"""The global flag for debug output.

This flag can be set through the :func:`enable_debug` function."""
#
#
#
#
################################################################################
#
# public functions
#
################################################################################
def align_evaporation_field(V, U, params_in, E_0, θ_m, num_points = 10000):
    """
    Automatically align evaporation field for taper geometry.

    This function takes :math:`r_0`, :math:`θ`, and :math:`β` as variation
    parameters to align the evaporation field to the given target value through
    least squares minimization.


    Parameters
    ----------
    V : ndarray, shape (n,)
        The cumulated reconstructed volume for the *n* events.
    U : ndarray, shape (n,)
        The measured voltages of the *n* events.
    params_in : dict
        The parameters to be optimized, represented by the dictionary keys
        ``r_0``, ``θ``, and ``β``. The values are 2-tuples, representing the
        initial guess (float) and whether the parameter should be varied (bool).
    E_0 : float
        The target evaporation field for alignment.
    θ_m : float
        The (half) aperture angle (in radians).
    num_points : int
        The number of points used to sample the evaporation field over the given
        entire data set. The curve can be considered to change only slowly so
        that a small subset is sufficient for evaluation. Defaults to ``10000``
        sample points.

    Returns
    -------
    params_out : tuple
        The optimized parameters :math:`(r_0, θ, β)`.
    """
    #
    #
    def _squared_residuals(p, params, V, U, E_0, θ_m):
        """
        Least squares minimizer function.

        This function calculates the sum of squared residuals with respect to
        the evaporation field target value E_0.
        """
        #
        #
        # parse optimization parameters
        i = 0
        if params['r_0'][1] == True:
            r_0  = p[i] * 10.0
            i   += 1
        else:
            r_0  = params['r_0'][0]
        #
        if params['θ'][1] == True:
            θ  = p[i]
            i += 1
        else:
            θ  = params['θ'][0]
        #
        if params['β'][1] == True:
            β  = p[i]
            i += 1
        else:
            β  = params['β'][0]
        #
        #
        # calculate radii of curvature for taper geometry
        _, r = get_geometry_taper(V, r_0, θ, θ_m, use_numba = False)
        #
        #
        # return sum of squared residuals
        return np.sum((U / (β * r) - E_0)**2)
    #
    #
    #
    #
    # reduce number of data points
    if num_points > 0:
        print("Reducing data points for automatic evaporation field alignment…")
        sl_step = len(V) // num_points
        V = V[::sl_step]
        U = U[::sl_step]
    #
    #
    # set initial parameter values and bounds (scale r_0 to maintain same order
    # of magnitude for all parameters)
    init   = ()
    bounds = ()
    if params_in['r_0'][1] == True:
        init   += (params_in['r_0'][0] / 10.0,)
        bounds += ((1.0, 50.0),)
    #
    if params_in['θ'][1] == True:
        init   += (params_in['θ'][0],)
        bounds += ((np.deg2rad(45.0) , np.deg2rad(90.0)),)
    #
    if params_in['β'][1] == True:
        init   += (params_in['β'][0],)
        bounds += ((1.0, 10.0),)
    #
    #
    # minimize squared residuals
    print("Aligning evaporation field to {0:.1f} V/nm…".format(E_0))
    res = minimize(
        _squared_residuals, np.asarray(init),
        args = (params_in, V, U, E_0, θ_m),
        bounds = bounds, method = 'Nelder-Mead', options = {'xatol': 1e-2}
    )
    #
    #
    # parse optimized parameters
    params_out = ()
    i          = 0
    if params_in['r_0'][1] == True:
        params_out += (res.x[i] * 10.0,)
        i          += 1
    else:
        params_out += (params_in['r_0'][0],)
    #
    if params_in['θ'][1] == True:
        params_out += (res.x[i],)
        i          += 1
    else:
        params_out += (params_in['θ'][0],)
    #
    if params_in['β'][1] == True:
        params_out += (res.x[i],)
        i          += 1
    else:
        params_out += (params_in['β'][0],)
    #
    #
    # return optimized parameters
    return params_out
#
#
#
#
def enable_debug(is_dbg):
    """
    Enable or disable debug output.

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
@numba.njit([
    "f4[:](f4[:], f8[:], f8)", "f8[:](f8[:], f8[:], f8)"
    ], cache = True, parallel = True
)
def get_evaporation_field(U, r, β):
    """
    Calculate evaporation field.

    The evaporation field is calculated according to the relation
    :math:`E = \\frac{U}{\\beta r}`, where :math:`\\beta` is the field factor.


    Parameters
    ----------
    U : ndarray, shape (n,)
        The measured voltages of the *n* events.
    r : ndarray, shape (n,)
        The corresponding radii of curvature.
    β : float
        The field factor.

    Returns
    -------
    E : ndarray, shape (n,)
        The evaporation field of the *n* events. The precision is inferred from
        the data type of the *U* parameter.
    """
    #
    #
    # infer the precision of the evaporation field from the input data type
    E = np.empty_like(U)
    #
    # calculate evaporation field
    E[:] = U / (β * r)
    #
    # return evaporation field
    return E
#
#
#
#
@numba.njit(
    "Tuple((f8[:], f8[:]))(f8[:], f8[:], f8)", cache = True, parallel = True
)
def get_geometry_classic(Ω, r, θ_m):
    """
    Calculate tip geometry through 'classic' scheme.

    This function calculates the tip geometry (current :math:`z`-position of
    sphere center and radius of curvature) through the classical scheme for each
    event to be reconstructed. The reconstructed volume between two events is
    calculated exactly with the aid of a local taper geometry according to the
    two consecutive radii of curvature.


    Parameters
    ----------
    Ω : ndarray, shape (n,)
        The individual atomic volumes for the *n* events.
    r : ndarray, shape (n,)
        The calculated radii of curvature for the *n* events.
    θ_m : float
        The (half) aperture angle (in radians).

    Returns
    -------
    z : ndarray, shape (n,)
        The :math:`z`-positions of the sphere centers.
    r : ndarray, shape (n,)
        The corresponding radii of curvature.
    """
    #
    #
    # set consecutive radii of curvature for each local taper
    r_0 = r[:-1]
    r_1 = r[1:]
    #
    #
    # calculate local z-increment
    γ = 4.0 / 3.0 * np.pi * (r_0**2 + r_0 * r_1 + r_1**2) * np.sin(θ_m / 2.0)**2
    Δz     = np.empty_like(r)
    Δz[0]  = 0.0
    Δz[1:] = (Ω[:-1] / γ + (r_1 - r_0)) / np.cos(θ_m / 2.0)**2
    #
    #
    # return tip geometry (z-position of sphere center for each event to be
    # reconstructed and corresponding radius of curvature)
    return np.cumsum(Δz), r
#
#
#
#
def get_geometry_taper(V, r_0, θ, θ_m, use_numba = True):
    """
    Calculate taper geometry.

    This function solves a cubic equation according to Cardano's method to
    calculate the taper geometry (current :math:`z`-position of sphere center
    and radius of curvature) for each event to be reconstructed. Only the
    physical solution is calculated and returned while the other two complex
    solutions are dropped. This method is taken in modified form from |cardano|.


    Parameters
    ----------
    V : ndarray, shape (n,)
        The cumulated reconstructed volume for the *n* events.
    r_0 : float
        The initial radius of the hemispherical cap.
    θ : float
        The polar angle of the hemispherical cap (in radians). This angle is
        related to the taper angle :math:`\\alpha` through the relation
        :math:`\\theta = \\frac{\\pi - \\alpha}{2}`.
    θ_m : float
        The (half) aperture angle (in radians).
    use_numba : bool
        Whether to use the Numba njit'ed function. For low overall computational
        costs, calling the Numba function may cause too much overhead so that
        the native Python implementation may be faster, which can be used by
        setting the *use_numba* flag to ``False``. Defaults to ``True``, i.e.
        use the Numba function.

    Returns
    -------
    z : ndarray, shape (n,)
        The :math:`z`-positions of the sphere centers.
    r : ndarray, shape (n,)
        The corresponding radii of curvature.


    .. |cardano| raw:: html

        <a href="https://medium.com/@mephisto_Dev/
        solving-cubic-equation-using-cardanos-method-with-python-9465f0b92277"
        target="_blank">
        "Solving Cubic Equation using Cardano’s method with Python"
        </a>
    """
    #
    #
    # this decorator decides whether to use the Numba njit'ed function or pure
    # Python function depending on the value of 'use_numba'
    @numba.njit(
        "Tuple((f8[:], f8[:], f8))(f8[:], f8, f8, f8, b1)",
        cache = True, parallel = True
    ) if use_numba == True else lambda obj: obj
    def _get_geometry_taper(V, r_0, θ, θ_m, is_dbg):
        """
        Small helper function which solves the cubic equation.
        """
        #
        #
        # special case of 90° cap angle (cylinder instead of cone) needs to be
        # handled separately to avoid divergence of the solution below; simply
        # return results for a perfect cylinder
        if np.abs(θ - np.pi / 2) < np.finfo(np.float32).eps:
            return \
                V / (np.pi * (r_0 * np.sin(θ_m))**2), np.full_like(V, r_0), 0.0
        #
        #
        #
        #
        # set coefficients for cubic equation a * z^3 + b * z^2 + c * z + d = 0;
        # a is equal to unity by definition
        b = 3.0 * r_0 / np.cos(θ)
        c = 3.0 * (r_0 / np.cos(θ))**2
        d = -3.0 * V / (
            4.0 * np.pi * np.cos(θ)**2 *
            np.sin(θ_m / 2.0)**2 * (np.cos(θ_m / 2.0)**2 - np.cos(θ))
        )
        #
        #
        # calculate intermediate values (δ is positive for this specific
        # physical case)
        p      = c - b**2 / 3.0
        q      = 2.0 / 27.0 * b**3 - b * c / 3.0 + d
        δ_sqrt = np.sqrt(q**2 / 4.0 + p**3 / 27.0)
        #
        #
        # calculate (physical) solution
        u = (-0.5 * q + δ_sqrt)**(1.0/3.0)
        v = (-0.5 * q - δ_sqrt)**(1.0/3.0)
        z = u + v - b / 3.0
        #
        #
        # in debug mode, check the solutions for consistency, i.e. calculate
        # the corresponding function value which *must* be zero
        δ_max = 0.0
        if is_dbg == True:
            δ_max = np.abs(z**3 + b * z**2 + c * z + d).max()
        #
        #
        # return taper geometry (and maximum deviation of solution from zero)
        return z, r_0 + z * np.cos(θ), δ_max
    #
    #
    #
    #
    # limit aperture angle if required due to tip geometry
    # (should not occur realistically)
    if θ_m > θ:
        print("Reducing aperture angle to {0:.2f}° due to taper geometry.".
              format(np.rad2deg(θ)))
        θ_m = θ
    #
    #
    # call helper function to calculate taper geometry
    z, r, δ_max = _get_geometry_taper(V, r_0, θ, θ_m, _is_dbg)
    #
    #
    # in debug mode, check the solutions for consistency
    _debug(
        "Maximum deviation of the solution for the cubic equation is {0:.6e}.".
        format(δ_max)
    )
    #
    #
    # return taper geometry (z-position of sphere center for each event to be
    # reconstructed and corresponding radius of curvature)
    return z, r
#
#
#
#
@numba.njit([
    "f4[:,:](f4[:,:], f8[:], f8[:], f8, f8)",
    "f8[:,:](f8[:,:], f8[:], f8[:], f8, f8)"
    ], cache = True, parallel = True
)
def reconstruct(xy_det, z_0, r, L, ξ):
    """
    Reconstruct :math:`xyz` tip positions.

    This function reconstructs the :math:`xyz` tip positions using the
    conventional point projection algorithm from |point_projection|


    Parameters
    ----------
    xy_det : ndarray, shape (n, 2)
        The :math:`xy` detector positions for the *n* events.
    z_0 : ndarray, shape (n,)
        The :math:`z`-positions of the sphere centers.
    r : ndarray, shape (n,)
        The corresponding radii of curvature.
    L : float
        The distance between the tip and the detector. The unit **must match**
        the unit of the :math:`xy` detector positions.
    ξ : float
        The image compression factor which is defined as :math:`\\xi =
        \\frac{\\theta_{\\textnormal{crys}}}{\\tan\\theta_{\\textnormal{obs}}}`.

    Returns
    -------
    xyz_tip : ndarray, shape (n, 3)
        The :math:`xyz` tip positions of the *n* reconstructed events. The
        precision is inferred from the data type of the *xy_det* parameter.


    .. |point_projection| raw:: html

        <a href="https://doi.org/10.1016/0169-4332(94)00561-3" target="_blank">
        Bas et al.</a>
    """
    #
    #
    # unpack x- and y-positions for faster processing
    x = xy_det[:, 0]
    y = xy_det[:, 1]
    #
    #
    # calculate radial detector positions
    r_det = np.sqrt(x * x + y * y)
    #
    # calculate tip angles
    θ_tip = r_det / L * ξ
    #
    # handle special case of r_det = 0 (0/0 --> NaN); the exact mathematical
    # result is tmp = r * ξ / L, but we multiple by zero below, so it is safe to
    # set tmp to zero here for performance reasons (note that the np.isnan()
    # filter would erroneously also catch all other sources for NaN)
    tmp = r * np.sin(θ_tip) / r_det
    tmp[np.isnan(tmp)] = 0.0
    #
    #
    # infer the precision of the reconstructed positions from the input data
    # type; as of now, the raw measurement file is stored as float32, so it
    # wouldn't make sense to use higher precision for the reconstructed data
    # (which also reduces memory and should speed up some calculations)
    # (same units as z_0, r)
    xyz_tip = np.empty((len(xy_det), 3), dtype = xy_det.dtype)
    xyz_tip[:, 0] = tmp * x
    xyz_tip[:, 1] = tmp * y
    xyz_tip[:, 2] = z_0 - r * np.cos(θ_tip)
    #
    #
    # return reconstructed tip positions
    return xyz_tip
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
