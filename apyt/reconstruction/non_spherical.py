"""
The APyT advanced reconstruction module
=======================================

This module provides methods for the reconstruction of non-spherical tip shapes.


Howto
-----

The usage of this module is demonstrated in an auxiliary script
(``wrapper_scripts/apyt_reconstruction_advanced.py``) which basically serves as
a wrapper for this module. Detailed usage information can be obtained by
invoking this script with the ``"--help"`` option.


List of classes
---------------

This module provides some generic classes for the reconstruction of raw
measurement data without the restriction of a spherical tip shape.

The following classes are provided:

* :class:`CurvatureReconstructor`: Reconstruct tip height profile based on
  Gaussian curvature.


.. sectionauthor:: Sebastian M. Eich <Sebastian.Eich@imw.uni-stuttgart.de>
.. moduleauthor::  Sebastian M. Eich <Sebastian.Eich@imw.uni-stuttgart.de>

"""
#
#
#
#
__version__ = '0.1.0'
__all__ = ['CurvatureReconstructor']
#
#
#
#
# TODO:
#   - uncomment Numba njit'ed decorator
#   - check preconditioning for Krylov root solver
#     (https://docs.scipy.org/doc/scipy/tutorial/optimize.html#still-too-slow-preconditioning)
#   - use kernel density estimators for triangulation of detector density?
#
#
#
#
# import modules
import matplotlib.pyplot as plt
import numba
import numpy as np
import sys
import warnings
#
# import some special functions/modules
from findiff import FinDiff
from mpl_toolkits.axes_grid1 import make_axes_locatable
from numpy.polynomial.polynomial import polyval2d, polyvander2d
from scipy.interpolate import griddata, interpn
from scipy.ndimage import gaussian_filter
from scipy.optimize import root, root_scalar
from scipy.spatial import Delaunay
from scipy.stats import binned_statistic_2d
from timeit import default_timer as timer
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
class CurvatureReconstructor:
    """
    Reconstruct tip height profile based on Gaussian curvature.

    Parameters
    ----------
    xy_data: ndarray, shape (n, 2)
        The *x* and *y* detector positions of **all** events.
    ids: ndarray, shape (n,)
        The chemical IDs of **all** events used for the reconstruction.
    V_at: ndarray, shape (n,)
        The volumes of **all** events used for the reconstruction.
    R0: float
        The detector radius (in mm)
    r0: float
        The tip radius (in nm)
    L0: float
        The distance between tip and detector (in mm)
    ξ: float
        The image compression factor.
    ζ: float
        The detection efficiency.
    num_points_tip: int
        The number of points used for the construction of the tip grid. Must be
        an odd number. Defaults to ``101``.


    The following class methods are provided:

    * :meth:`debug_plot`: Show plots for debugging.
    * :meth:`reconstruct_height_profile`: Reconstruct tip
      height profile based on Gaussian curvature.
    * :meth:`reconstruct_positions`: Reconstruct three-dimensional tip
      positions.


    The following **general** instance properties can be accessed (*read-only*):

    * :attr:`ω`: The aperture angle.


    The **following tip-related** instance properties can be accessed
    (*read-only*):

    * :attr:`X_tip`: The *x* positions of the tip grid.
    * :attr:`Y_tip`: The *y* positions of the tip grid.
    * :attr:`mask_tip`: The Boolean mask specifying valid circular points.


    |


    Below is a list of all class objects with their detailed description.
    """
    #
    #
    def __init__(
        self, xy_data, ids, V_at, R0, r0, L0, ξ, ζ, num_points_tip = 101
    ):
        #
        #
        # set instance attributes
        self._xy_data = xy_data
        self._ids     = ids
        self._V_at    = V_at
        self._R0      = R0
        self._r0      = r0
        self._L0      = L0
        self._ξ       = ξ
        self._ζ       = ζ
        #
        #
        # correct volumes for detection efficiency
        self._V_at /= ζ
        #
        #
        # set aperture angle
        self._ω = np.arctan(self._R0 / self._L0) * self._ξ
        print(
            "(Half) aperture angle is {0:.2f}°.\n".format(np.rad2deg(self._ω))
        )
        #
        #
        # set up tip grid
        self._setup_tip(self._r0, num_points_tip, 0.5)
    #
    #
    #
    #
    ############################################################################
    ###                                                                      ###
    ###     General properties                                               ###
    ###                                                                      ###
    ############################################################################
    @property
    def ω(self):
        """Getter for the internal ``_ω`` attribute (*read-only*).

        Returns
        -------
        ω: float
            The (half) aperture angle (in radians).
        """
        return self._ω
    #
    #
    ############################################################################
    ###                                                                      ###
    ###     Tip-related properties                                           ###
    ###                                                                      ###
    ############################################################################
    @property
    def X_tip(self):
        """Getter for the internal ``_X_tip`` attribute (*read-only*).

        Returns
        -------
        X_tip :ndarray, shape (n, n)
            The *x* positions of the tip grid, as returned by the
            ``numpy.meshgrid()`` function.
        """
        return self._X_tip
    #
    @property
    def Y_tip(self):
        """Getter for the internal ``_Y_tip`` attribute (*read-only*).

        Returns
        -------
        Y_tip :ndarray, shape (n, n)
            The *y* positions of the tip grid, as returned by the
            ``numpy.meshgrid()`` function.
        """
        return self._Y_tip
    #
    @property
    def mask_tip(self):
        """Getter for the internal ``_mask_tip`` attribute (*read-only*).

        Returns
        -------
        mask_tip: ndarray, shape (n, n)
            The Boolean mask specifying valid circular points on the square tip
            grid.
        """
        return self._mask_tip
    #
    #
    #
    #
    ############################################################################
    ###                                                                      ###
    ###     Public class-level methods                                       ###
    ###                                                                      ###
    ############################################################################
    def debug_plot(self, results):
        """
        Show plots for debugging.

        Parameters
        ----------
        results: dict
            The dictionary containing various results for the reconstruction of
            the height profile. See :meth:`reconstruct_height_profile()` for
            details.
        """
        #
        #
        def _boundary_plotter(ax):
            """
            Simple helper to plot a circle at the boundary of the tip.
            """
            #
            ax.add_patch(
                plt.Circle(
                    (0.0, 0.0), self._r0, color = 'k',
                    fill = False, linewidth = 2.0
                )
            )
        #
        def _color_bar(obj, ax):
            """
            Simple helper to place color bar next to plot object.
            """
            #
            fig.colorbar(
                obj,
                cax = make_axes_locatable(ax).append_axes(
                    "right", size = "5%", pad = 0.10
                )
            )
        #
        def _tri_plotter(ax):
            """
            Simple helper to plot mapped detector triangulation.
            """
            ax.triplot(
                *tri_map.T, results['tri']['tri'].simplices,
                color = 'w', linewidth = 0.25
            )
        #
        #
        #
        #
        # map detector triangulation to tip grid
        tri_map = self._map_detector_triangulation_to_tip(
            results['ΔH'], results['tri']['tri']
        )
        #
        #
        #
        #
        # set up figure
        fig, axes = plt.subplots(2, 3, figsize = (3 * 4.8, 1.75 * 4.8))
        fig.subplots_adjust(
            top = 0.90, bottom = 0.07, left = 0.06, right = 0.94,
            hspace = 0.35, wspace = 0.50
        )
        fig.suptitle(
            "Debug plots for data interval represented by the object "
            "\"{0:s}\".".format(str(results['sl']))
        )
        #
        #
        # triangulated detector density
        ax = axes[0][0]
        ax.set_title(
            "Triangulated detector density "
            "($\\frac{\mathrm{nm}^3}{\mathrm{sr}}$)"
        )
        ax.set_aspect('equal')
        ax.set_xlabel('$x_\mathrm{det}$ (mm)')
        ax.set_ylabel('$y_\mathrm{det}$ (mm)')
        tric = ax.tripcolor(
            *results['tri']['tri'].points.T, results['tri']['tri'].simplices,
            facecolors = results['ρ_det'],
            edgecolors = 'w'
        )
        _color_bar(tric, ax)
        #
        #
        # triangulated Gaussian curvature (detector)
        ax = axes[0][1]
        ax.set_title("Triangulated Gaussian curvature $K$ ($\mathrm{nm}^{-2}$)")
        ax.set_aspect('equal')
        ax.set_xlabel('$x_\mathrm{det}$ (mm)')
        ax.set_ylabel('$y_\mathrm{det}$ (mm)')
        tric = ax.tripcolor(
            *results['tri']['tri'].points.T, results['tri']['tri'].simplices,
            facecolors = results['K_det'],
            edgecolors = 'w'#,
            #vmin = 0.5 / self._r0**2, vmax = 2.0 / self._r0**2
        )
        _color_bar(tric, ax)
        #
        #
        # mapped tip target Gaussian curvature
        ax = axes[0][2]
        ax.set_aspect('equal')
        ax.set_title("Mapped target tip curvature $K$ ($\mathrm{nm}^{-2}$)")
        ax.set_xlabel('$x_\mathrm{tip}$ (nm)')
        ax.set_ylabel('$y_\mathrm{tip}$ (nm)')
        pcm = ax.pcolormesh(
            self._X_tip, self._Y_tip, results['K_tip_from_det'],
            shading = 'nearest'#,
            #vmin = 0.5 / self._r0**2, vmax = 2.0 / self._r0**2
        )
        _tri_plotter(ax)
        _boundary_plotter(ax)
        _color_bar(pcm, ax)
        #
        #
        # reconstructed tip Gaussian curvature
        ax = axes[1][0]
        ax.set_aspect('equal')
        ax.set_title("Reconstructed tip curvature $K$ ($\mathrm{nm}^{-2}$)")
        ax.set_xlabel('$x_\mathrm{tip}$ (nm)')
        ax.set_ylabel('$y_\mathrm{tip}$ (nm)')
        pcm = ax.pcolormesh(
            self._X_tip, self._Y_tip, results['K_tip_from_height'],
            shading = 'nearest'#,
            #vmin = 0.5 / self._r0**2, vmax = 2.0 / self._r0**2
        )
        _tri_plotter(ax)
        _boundary_plotter(ax)
        _color_bar(pcm, ax)
        #
        #
        # relative curvature residuals
        ax = axes[1][1]
        ax.set_aspect('equal')
        ax.set_title("Relative curvature residuals")
        ax.set_xlabel('$x_\mathrm{tip}$ (nm)')
        ax.set_ylabel('$y_\mathrm{tip}$ (nm)')
        pcm = ax.pcolormesh(
            self._X_tip, self._Y_tip, results['ΔK'],
            shading = 'nearest',
            vmin = -2.0 * results['tol'], vmax = 2.0 * results['tol']
        )
        _tri_plotter(ax)
        _boundary_plotter(ax)
        _color_bar(pcm, ax)
        #
        #
        # reconstructed height profile
        ax = axes[1][2]
        ax.set_aspect('equal')
        ax.set_title("Reconstructed height profile $\Delta H$ ($\mathrm{nm}$)")
        ax.set_xlabel('$x_\mathrm{tip}$ (nm)')
        ax.set_ylabel('$y_\mathrm{tip}$ (nm)')
        pcm = ax.pcolormesh(
            self._X_tip, self._Y_tip, results['H'] - self._H_sphere,
            shading = 'nearest'
        )
        _tri_plotter(ax)
        _boundary_plotter(ax)
        _color_bar(pcm, ax)
        #
        #
        #
        #
        # show plots
        plt.show()
    #
    #
    #
    #
    def reconstruct_height_profile(
        self, i1, i2, ΔH_0 = None, N_simp = 500, maxiter = 100, tol = 1e-2,
        tri = None, verbose = False, debug_plot = False
    ):
        """
        Reconstruct tip height profile for given data interval.

        The core routine uses the |krylov| root finding algorithm from the SciPy
        |optimize| package for the reconstruction of the height profile.


        Parameters
        ----------
        i1: int
            The index of the *first* event to use (inclusive).
        i2: int
            The index of the *last* event to use (exclusive).
        ΔH_0: ndarray, shape (n, n)
            The height profile of the tip surface, expressed as the deviation
            :math:`\\Delta H` from a perfect sphere. If not provided, a perfect
            hemisphere is assumed for the height profile.
        N_simp: int
            The (approximate) target number of simplices for the detector
            triangulation. Defaults to ``500``.
        maxiter: int
            The maximum number of iterations for the root solver. See also
            :attr:`maxiter`. Defaults to ``100``.
        tol: float
            The maximum absolute relative curvature residual allowed for the
            reconstruction of the height profile. See also :attr:`tol`. Defaults
            to ``0.01``.
        tri: class
            The custom detector triangulation, as obtained by |delaunay|.
            Defaults to ``None``, i.e. use internal adaptive triangulation. This
            option may be useful if the internal triangulation fails.
        verbose: bool
            Whether to print additional information. Defaults to ``False``.
        debug_plot: bool
            Whether to show debug plots with various information. Defaults to
            ``False``. Setting this parameter to ``True`` is equivalent to
            calling :meth:`debug_plot` with the returned results.

        Returns
        -------
        results: dict
            The dictionary containing various results for the reconstruction of
            the height profile, e.g. the reconstructed height profile and
            information on the detector triangulation. Use ``results.keys()`` to
            get a complete list of the dictionary content. This dictionary can
            be passed directly to :meth:`debug_plot()`.


        .. |krylov| raw:: html

            <a href="https://docs.scipy.org/doc/scipy/reference/
            optimize.root-krylov.html" target="_blank">Krylov</a>

        .. |optimize| raw:: html

            <a href="https://docs.scipy.org/doc/scipy/reference/optimize.html"
            target="_blank">Optimization and root finding</a>
        """
        #
        #
        def _tab_write(msg):
            """
            Small helper to overload sys.stdout.write and append tabulator to
            output.
            """
            sys_stdout_write("\t" + msg)
        #
        #
        #
        #
        # start timer
        start = timer()
        #
        #
        # set slice object representing the data range to use
        sl = np.s_[i1:i2]
        print(
            "Reconstructing height profile for data interval specified by "
            "object \"{0:s}\".".format(str(sl))
        )
        #
        #
        # initialize height profile if not provided
        if ΔH_0 is None:
            ΔH_0 = np.zeros_like(self._X_tip)
        #
        #
        #
        #
        # triangulate detector events for given slice object
        T = self._triangulate_detector(sl, N_simp, "height", tri, verbose)
        if T == None:
            return {"success": False}
        #
        #
        # calculate cumulated detector volumes for each simplex (filter invalid
        # events outside triangulation), shape (N_simp,)
        V = np.bincount(
            T['s'][T['is_valid']], weights = self._V_at[sl][T['is_valid']]
        )
        #
        # calculate relative Gaussian curvature based on detector density,
        # shape (N_simp,)
        κ_det = T['Ω'] * np.cos(T['θ']) / V
        #
        #
        #
        #
        # use Krylov root finding algorithm for height profile reconstruction
        print("Optimizing height profile…")
        #
        # store and overload write function
        sys_stdout_write     = sys.__stdout__.write
        sys.__stdout__.write = _tab_write
        #
        sol = root(
            self._calculate_curvature_residuals, ΔH_0[self._mask_tip],
            args = (T, κ_det),
            method = 'krylov',
            options = {
                'maxiter': maxiter,
                'fatol'  : tol,
                'disp'   : verbose
            }
        )
        #
        # restore original write function
        sys.__stdout__.write = sys_stdout_write
        #
        #
        #
        #
        # apply solution to get optimized reconstructed height profile
        ΔH = self._extrapolate_height_profile(sol.x)
        #
        # calculate curvature from detector view
        K_from_det, Δz = \
            self._get_tip_curvature_from_detector_view(ΔH, T, κ_det)
        #
        # calculate curvature from height profile
        K_from_height = self._get_tip_curvature_from_height_profile(ΔH)
        #
        # calculate curvature residuals, shape (n, n)
        residuals = np.full_like(self._X_tip, np.nan)
        residuals[self._mask_tip] = \
            self._calculate_curvature_residuals(ΔH[self._mask_tip], T, κ_det)
        #
        #
        # collect all relevant results in a dictionary
        results = {
            'sl'                : sl,
            'tri'               : T,
            'ρ_det'             : V / T['Ω'],
            'K_det'             : κ_det * Δz,
            'K_tip_from_det'    : K_from_det,
            'K_tip_from_height' : K_from_height,
            'ΔK'                : residuals,
            'H'                 : ΔH + self._H_sphere,
            'ΔH'                : ΔH,
            'Δz'                : Δz,
            'tol'               : tol,
            'success'           : sol.success
        }
        #
        #
        #
        #
        # print convergence status
        if sol.success != True:
            warnings.warn(
                "Solver did not find a solution at the specified tolerance for "
                "data interval specified by slice object \"{0:s}\". Maximum "
                "absolute relative curvature residual is {1:.6f}.".
                format(str(sl), np.abs(sol.fun).max())
            )
        else:
            if verbose:
                print(
                    "\tSolver found a solution with maximum absolute relative "
                    "curvature residual of {0:.6f} "
                    "(mean: {1:.6f}; std: {2:.6f}).".
                    format(
                        np.abs(sol.fun).max(), np.mean(sol.fun), np.std(sol.fun)
                    )
                )
        if verbose:
            print(
                "\tΔz increment is {0:.3f} nm (spherical estimate is "
                "{1:.3f} nm).".
                format(
                    Δz, np.sum(V) / (np.pi * (self._r0 * np.sin(self._ω))**2)
                )
            )
        print(
            "Height profile reconstruction took {0:.3f} s "
            "({1:d} iterations).\n".format(timer() - start, sol.nit)
        )
        #
        #
        #
        #
        # show debug plots if requested
        if debug_plot == True:
            self.debug_plot(results)
        #
        #
        #
        #
        # return results dictionary
        return results
    #
    #
    #
    #
    def reconstruct_positions(
        self, p_list, N_simp = 500, extrapolate = False, tri_list = None,
        verbose = False
    ):
        """
        Reconstruct three-dimensional atomic positions between height profiles.

        This method takes a list of reconstructed height profiles, obtained from
        :meth:`reconstruct_height_profile()`, and iteratively reconstructs the
        three-dimensional atomic positions between each consecutive pair of
        height profiles.

        The height increment :math:`\Delta z` between consecutive height profile
        pairs is determined such that the total reconstructed volume matches
        exactly the volume spanned by the triangulated upper and lower height
        profiles. Each triangulated simplex, corresponding to both height
        profiles, forms a distorted prism.

        The volumes of these distorted prisms are calculated as follows:
        Consider two simplices, :math:`(A_1, B_1, C_1)` from the upper profile
        and :math:`(A_2, B_2, C_2)` from the lower profile. The lateral surface
        of this distorted prism is triangulated through the edges
        :math:`(A_1, B_2)`, :math:`(B_2, C_1)`, and :math:`(C_1, A_2)`.

        The volume of the prism can be divided into three tetrahedra:

        .. math::

            T_1 &= (A_1, B_1, C_1, B_2) \\\\
            T_2 &= (A_2, B_2, C_2, C_1) \\\\
            T_3 &= (A_1, C_1, A_2, B_2).


        The total volume is then calculated as the sum of the volumes of these
        three tetrahedra.


        Parameters
        ----------
        p_list: list
            The list of the reconstructed height profiles, as obtained by
            :meth:`reconstruct_height_profile()`.
        N_simp: int
            The (approximate) target number of simplices for the detector
            triangulation. Defaults to ``500``.
        extrapolate: bool
            Whether to reconstruct the positions extrapolated beyond the first
            and last height profile.
        tri_list: list
            The list of custom detector triangulations, as obtained by
            |delaunay|. One triangulation is needed for every reconstruction
            interval. Defaults to ``None``, i.e. use internal adaptive
            triangulation. The list may contain ``None`` for individual
            reconstruction intervals to switch between user-provided and
            internal triangulation. This option may be useful if the internal
            triangulation fails.
        verbose: bool
            Whether to print additional information. Defaults to ``False``.

        Returns
        -------
        ids, xyz: (structured) ndarray, shape (N,)
            The IDs and three-dimensional :math:`xyz` tip positions of the
            :math:`N` reconstructed events, given as a structured array with
            fields ``id``, ``x``, ``y``, and ``z``.
        """
        #
        #
        def _volume_residual(Δz, V_0, r_1, r_2, tri):
            """
            Helper to calculate volume residual for given height profile shift
            :math:`\Delta z`.
            """
            #
            #
            # shift second height profile
            r_2[:, 2] -= Δz
            #
            #
            # group corresponding simplex vertices for both triangulated height
            # profiles, shape(N_simp, 6, 3)
            vert = np.concatenate(
                (r_1[tri.simplices], r_2[tri.simplices]), axis = 1
            )
            #
            # calculate volumes between the two height profiles for all
            # simplices; tetrahedron volume with vertices a,b,c,d is given by
            # V = |det(a-d, b-d, c-d)| / 6
            V = 1.0 / 6.0 * (
                np.abs(np.linalg.det(np.dstack((
                    vert[:,0] - vert[:,4],
                    vert[:,1] - vert[:,4],
                    vert[:,2] - vert[:,4]
                )))) +
                np.abs(np.linalg.det(np.dstack((
                    vert[:,3] - vert[:,2],
                    vert[:,4] - vert[:,2],
                    vert[:,5] - vert[:,2]
                )))) +
                np.abs(np.linalg.det(np.dstack((
                    vert[:,0] - vert[:,4],
                    vert[:,2] - vert[:,4],
                    vert[:,3] - vert[:,4]
                ))))
            )
            #
            #
            # restore original array (would otherwise be modified iteratively)
            r_2[:, 2] += Δz
            #
            #
            # return volume residual
            return np.sum(V) - V_0
        #
        #
        def _get_position_vectors(ΔH, xy_tri, T, B):
            """
            Simple helper to obtain three-dimensional tip positions from
            barycentric coordinates with respect to detector triangulation.
            """
            #
            #
            # calculate xy tip position based on barycentric coordinates
            xy = np.einsum(
                'ijk,ij->ik',
                xy_tri[T['tri'].simplices][T['s'][T['is_valid']]], B
            )
            #
            # calculate corresponding height
            z = interpn(
                (self.X_tip[:, 0], self.Y_tip[0]), self._H_sphere + ΔH, xy
            )
            #
            #
            # return three-dimensional tip positions
            return np.c_[xy, z]
        #
        #
        def _interpolate_height(ΔH, xy):
            """
            Simple helper to interpolate position for given height profile.
            """
            #
            #
            # interpolate heights
            z = interpn(
                (self.X_tip[:, 0], self.Y_tip[0]), self._H_sphere + ΔH, xy
            )
            #
            #
            # return three-dimensional position vectors
            return np.c_[xy, z]
        #
        #
        # start timer
        start = timer()
        #
        #
        #
        #
        # extrapolate first and last height profile if requested
        if extrapolate == True:
            print("Using extrapolation beyond first and last height profile.")
            #
            #
            # extrapolation of *first* height profile
            Δ = (p_list[0]['sl'].stop - p_list[0]['sl'].start) // 2
            p_first = {
                'ΔH': p_list[0]['ΔH'],
                'sl': np.s_[
                    p_list[0]['sl'].start - Δ :
                    p_list[0]['sl'].stop  - Δ
                ]
            }
            #
            # extrapolation of *last* height profile
            Δ = (p_list[-1]['sl'].stop - p_list[-1]['sl'].start) // 2
            p_last = {
                'ΔH': p_list[-1]['ΔH'],
                'sl': np.s_[
                    p_list[-1]['sl'].start + Δ :
                    p_list[-1]['sl'].stop  + Δ
                ]
            }
            #
            #
            # join profiles
            p_list = [p_first] + p_list + [p_last]
        #
        #
        # initialize triangulation list if not provided; each defaults to None,
        # i.e. internal triangulation
        if tri_list == None:
            tri_list = [None] * (len(p_list) - 1)
        #
        #
        #
        #
        # loop through height profile pairs
        current_index = 0
        ids = np.empty(len(self._xy_data))
        xyz = np.empty((len(self._xy_data), 3))
        for p1, p2, tri in zip(p_list[:-1], p_list[1:], tri_list):
            # check order of height profiles
            if p1['sl'].start >= p2['sl'].start:
                raise Exception("Height profiles must be ordered.")
            #
            #
            # use central events from each height profile to determine
            # reconstruction range; make sure to be in data range in case of
            # extrapolation
            sl = np.s_[
                max((p1['sl'].start + p1['sl'].stop) // 2, 0) :
                min((p2['sl'].start + p2['sl'].stop) // 2, len(self._xy_data))
            ]
            print(
                "Reconstructing positions for data interval specified by "
                "object \"{0:s}\".".format(str(sl))
            )
            #
            #
            # triangulate data
            T = self._triangulate_detector(
                sl, N_simp, "positions", tri, verbose
            )
            #
            #
            # calculate cumulated volume for all *valid* reconstructed events,
            # shape (N_rec,)
            print("Reconstructing positions…")
            V_cum = np.cumsum(self._V_at[sl][T['is_valid']])
            #
            #
            #
            #
            # x and y tip positions of the detector triangulation vertices,
            # shape (N_vert, 2)
            xy_tri_1 = \
                self._map_detector_triangulation_to_tip(p1['ΔH'], T['tri'])
            xy_tri_2 = \
                self._map_detector_triangulation_to_tip(p2['ΔH'], T['tri'])
            #
            #
            # get barycentric coordinates of reconstructed events,
            # shape (N_rec, 3)
            B, _ = self._get_barycentric_coordinates(T, self._xy_data[sl])
            #
            # get three-dimensional position vectors for reconstructed events
            # for both height profiles, shape (N_rec, 3)
            r_1 = _get_position_vectors(p1['ΔH'], xy_tri_1, T, B)
            r_2 = _get_position_vectors(p2['ΔH'], xy_tri_2, T, B)
            #
            #
            #
            #
            # initial guess for height profile shift (total reconstructed volume
            # divided by total cross-sectional area)
            vertices = xy_tri_1[T['tri'].simplices]
            Δz = V_cum[-1] / (0.5 * np.sum(np.abs(np.cross(
                vertices[:, 2] - vertices[:, 0],
                vertices[:, 1] - vertices[:, 0])
            )))
            #
            # find shift for height profile to match reconstructed volume
            Δz = root_scalar(
                _volume_residual,
                args = (
                    V_cum[-1],
                    _interpolate_height(p1['ΔH'], xy_tri_1),
                    _interpolate_height(p2['ΔH'], xy_tri_2),
                    T['tri']
                ),
                method = 'secant', x0 = Δz
            ).root
            if verbose:
                print("\tHeight profile increment is {0:.3f} nm.".format(Δz))
            #
            #
            # shift second height profile
            r_2[:, 2] -= Δz
            #
            #
            #
            #
            # calculate fractional Δz-increment based on cumulated reconstructed
            # volumes, interval (0.0, 1.0]
            Δz_rel = V_cum / V_cum[-1]
            #
            #
            # calculate three-dimensional reconstructed positions for all
            # events, shape (N_rec, 3)
            xyz[current_index : current_index + len(r_1)] = \
                r_1 + Δz_rel[:, np.newaxis] * (r_2 - r_1)
            #
            #
            # set corresponding chemical IDs
            ids[current_index : current_index + len(r_1)] = \
                self._ids[sl][T['is_valid']]
            #
            #
            # increment current index
            current_index += len(r_1)
        #
        #
        #
        #
        # drop remaining array elements (events outside triangulation are not
        # reconstructed)
        xyz = xyz[0:current_index]
        ids = ids[0:current_index]
        #
        #
        # create structured array
        id_xyz = np.empty(
            len(ids),
            dtype=np.dtype(
                [('id', 'i8'), ('x', 'f8'), ('y', 'f8'), ('z', 'f8')]
            )
        )
        id_xyz['id'] = ids
        id_xyz['x'] = xyz[:, 0]
        id_xyz['y'] = xyz[:, 1]
        id_xyz['z'] = xyz[:, 2]
        #
        #
        #
        #
        # return structured array containing chemical IDs and reconstructed
        # positions
        print(
            "Reconstruction of {0:d} events took {1:.3f} s.".
            format(len(ids), timer() - start)
        )
        return id_xyz
    #
    #
    #
    #
    ############################################################################
    ###                                                                      ###
    ###     Private class-level methods                                      ###
    ###                                                                      ###
    ############################################################################
    def _calculate_curvature_residuals(self, ΔH_in, *args):
        """
        Calculate residuals of Gaussian curvature.

        Parameters
        ----------
        ΔH_in: ndarray, shape (m,)
            The height profile for the (valid) tip points.
        args: tuple of length 2
            The tuple containing the detector triangulation and the relative
            Gaussian curvature based on the detector view.

        Returns
        -------
        ΔK: shape (m,)
            The (relative) residuals of the Gaussian curvature.
        """
        #
        #
        # extrapolate height profile to allow calculation of finite differences,
        # shape (n, n)
        ΔH = self._extrapolate_height_profile(ΔH_in)
        #
        #
        # get target Gaussian curvature of tip based on detector view,
        # shape (n, n)
        K_from_det, _ = self._get_tip_curvature_from_detector_view(ΔH, *args)
        #
        #
        # get current Gaussian curvature based on height profile, shape (n, n)
        K_from_height = self._get_tip_curvature_from_height_profile(ΔH)
        #
        #
        # calculate relative curvature residuals, shape (n, n)
        res = (K_from_height - K_from_det) / K_from_det
        #
        #
        # return residuals for variation points, shape (m,)
        return res[self._mask_tip]
    #
    #
    #
    #
    def _extrapolate_curvature(self, κ, mode = "generic", apply_filter = True):
        """
        Extrapolate relative Gaussian curvature.

        Parameters
        ----------
        κ: ndarray, shape (n, n)
            The Relative Gaussian curvature
            :math:`\\kappa = \\frac{K}{\\Delta z}`. Invalid points are
            represented by ``numpy.nan``.
        mode: str
            The mode used to extrapolate the relative curvature. Supported modes
            are: ``"generic"``. Defaults to ``"generic"``.
        apply_filter: bool
            Whether to apply a Gaussian filter to the extrapolated curvature.
            Defaults to ``True``.

        Returns
        -------
        K: shape (n, n)
            The extrapolated and normalized Gaussian curvature.
        Δz: float
            The normalization factor.
        """
        #
        #
        # set mask specifying valid tip grid points, i.e. seen by the detector
        is_valid = ~np.isnan(κ)
        #
        #
        #
        #
        # generic extrapolation using fit
        if mode == "generic":
            # fit 2d surface to relative curvature
            c = self._polyfit2d(
                self._X_tip[is_valid], self._Y_tip[is_valid], κ[is_valid], 0
            )
            #
            #
            # extrapolate relative curvature using fit
            κ[~is_valid] = polyval2d(
                self._X_tip[~is_valid], self._Y_tip[~is_valid], c
            )
        #
        #
        else:
            raise Exception(
                "Unknown extrapolation mode ({0:s}) specified.".format(mode)
            )
        #
        #
        #
        #
        # apply Gaussian filter if requested
        if apply_filter:
            κ = gaussian_filter(κ, 2.5)
        #
        #
        # return relative Gaussian curvature
        return κ
    #
    #
    #
    #
    def _extrapolate_height_profile(self, ΔH_in):
        """
        Extrapolate height profile :math:`\\Delta H`.

        Finite differences require exactly one additional grid point in each
        direction outside the boundary of the tip. We simply extrapolate the
        height profile :math:`\\Delta H` in radial direction using the *nearest*
        data point using |griddata|. The most elegant way would be the use of
        |stencils| from the findiff package.


        Parameters
        ----------
        ΔH_in: ndarray, shape (m, )
            The height profile to be extrapolated.

        Returns
        -------
        ΔH: ndarray, shape (n, n)
            The extrapolated height profile.


        .. |griddata| raw:: html

            <a href="https://docs.scipy.org/doc/scipy/reference/generated/
            scipy.interpolate.griddata.html" target="_blank">griddata</a>

        .. |stencils| raw:: html

            <a href="https://findiff.readthedocs.io/en/latest/source/
            getstarted.html#stencils" target="_blank">stencils</a>
        """
        #
        #
        # initialize output height profile
        ΔH = np.full_like(self._X_tip, np.nan)
        ΔH[self._mask_tip] = ΔH_in
        #
        #
        # interpolate data in polar coordinates, then use it for extrapolation
        ΔH[self._mask_ΔH_extra] = griddata(
            (self._R_tip[self._mask_ΔH_inter], self._φ[self._mask_ΔH_inter]),
            ΔH[self._mask_ΔH_inter],
            (self._R_tip[self._mask_ΔH_extra], self._φ[self._mask_ΔH_extra]),
            method = 'nearest'
        )
        #
        #
        # set center of height profile to zero
        ΔH -= \
            ΔH[(np.isclose(self._X_tip, 0.0)) & (np.isclose(self._Y_tip, 0.0))]
        #
        #
        # return extrapolated height profile for entire grid
        return ΔH
    #
    #
    #
    #
    @staticmethod
    def _get_barycentric_coordinates(T, P):
        """
        Calculate barycentric coordinates for given points *P*.

        This vectorized version of the example given in |delaunay| is taken from
        |barycentric_vectorized|.


        Parameters
        ----------
        T: dict
            The dictionary containing the triangulation results. See
            :meth:`_triangulate_detector()` for details.
        P: ndarray, shape (N, 2)
            The :math:`x` and :math:`y` positions of the triangulated points.

        Returns
        -------
        B: ndarray, shape (N, 3)
            The barycentric coordinates for the points given by *P*. Note that
            invalid points outside the triangulation are filtered.
        T: dict
            The dictionary containing the triangulation results. See
            :meth:`_triangulate_detector()` for details.


        .. |barycentric_vectorized| raw:: html

            <a href="https://stackoverflow.com/a/57866557"
            target="_blank">How to vectorize calculation of barycentric
            coordinates in python</a>
        """
        #
        #
        # if simplex indices are not part of the triangulation dictionary, we
        # perform the search here and update the dictionary accordingly
        if 's' not in T:
            T['s']        = T['tri'].find_simplex(P)
            T['is_valid'] = (T['s'] != -1)
        #
        #
        # filter invalid points
        s = T['s'][T['is_valid']]
        P = P[T['is_valid']]
        #
        #
        # calculate barycentric coordinates
        B = np.sum(
            T['tri'].transform[s, :2].transpose([1, 0, 2]) * \
                (P - T['tri'].transform[s, 2]),
            axis = 2
        ).T
        B = np.c_[B, 1.0 - B.sum(axis = 1)]
        #
        #
        # return barycentric coordinates and (updated) triangulation dictionary
        return B, T
    #
    #
    #
    #
    def _get_tip_curvature_from_detector_view(self, ΔH, T, κ_det):
        """
        Calculate Gaussian curvature based on detector view.

        Parameters
        ----------
        ΔH: ndarray, shape (n, n)
            The height profile of the tip surface, expressed as the deviation
            :math:`\\Delta H` from a perfect sphere.
        T: dict
            The dictionary containing the detector triangulation. See
            :meth:`_triangulate_detector()` for details.
        κ_det: ndarray, shape (M,)
            The relative Gaussian curvature based on detector view.

        Returns
        -------
        K: shape (n, n)
            The corresponding Gaussian curvature of the height profile.
        Δz: float
            The normalization factor.
        """
        #
        #
        # get relative Gaussian curvature based on detector view *and* current
        # height profile, shape (n, n)
        κ_map = self._map_curvatures(ΔH, T, κ_det)
        #
        #
        # get target Gaussian curvature based on detector view, extrapolation,
        # and normalization
        κ_ext = self._extrapolate_curvature(κ_map)
        #
        #
        # normalize curvature
        K, Δz = self._normalize_curvature(
            κ_ext, ΔH + self._H_sphere, self._weights,
            self._mask_norm,
            self._X_center_norm, self._Y_center_norm, self._weights_center_norm,
            self._δ, self._r0
        )
        #
        #
        # return normalized curvature and normalization factor
        return K, Δz
    #
    #
    #
    #
    def _get_tip_curvature_from_height_profile(self, ΔH):
        """
        Calculate Gaussian curvature for given height profile.

        The Gaussian curvature :math:`K` is calculated based on the partial
        derivatives of the given height profile :math:`H` according to

        .. math::
            K = \\frac{
                    \\frac{\\partial^2 H}{\\partial x^2} \,
                    \\frac{\\partial^2 H}{\\partial y^2} -
                    \\left(
                        \\frac{\\partial^2 H}{\\partial x \\partial y}
                    \\right)^2
                }{
                    \\left(
                        1 + \\left(\\frac{\\partial H}{\\partial x}\\right)^2 +
                        \\left(\\frac{\\partial H}{\\partial y}\\right)^2
                    \\right)^2
                }.

        The height profile :math:`H` is represented as a superposition of a
        reference sphere and a deviation :math:`\\Delta H`. The derivatives of
        the reference sphere are calculated analytically, while the derivatives
        of :math:`\\Delta H` are calculated using finite differences through the
        |findiff| package.

        Parameters
        ----------
        ΔH: ndarray, shape (n, n)
            The height profile of the tip surface, expressed as the deviation
            :math:`\\Delta H` from a perfect sphere.

        Returns
        -------
        K: shape (n, n)
            The corresponding Gaussian curvature of the height profile.


        .. |findiff| raw:: html

            <a href="https://findiff.readthedocs.io/en/latest/index.html"
            target="_blank">findiff</a>
        """
        #
        #
        # calculate first derivatives
        H_x = self._d_dx(ΔH) + self._grad_sphere[0]
        H_y = self._d_dy(ΔH) + self._grad_sphere[1]
        #
        # calculate second derivatives
        H_xx = self._d2_dx2(ΔH)  + self._Hessian_sphere[0]
        H_xy = self._d2_dxdy(ΔH) + self._Hessian_sphere[1]
        H_yy = self._d2_dy2(ΔH)  + self._Hessian_sphere[2]
        #
        #
        # return Gaussian curvature
        return (H_xx * H_yy - H_xy**2) / (1.0 + H_x**2 + H_y**2)**2
    #
    #
    #
    #
    @staticmethod
    def _get_triangulated_solid_angles(vertices, L, ξ):
        """
        Calculate solid angles based on given triangulation.


        Parameters
        ----------
        vertices : ndarray, shape (n, 3, 2)
            The :math:`x` and :math:`y` coordinates of the vertices of each simplex.
        L : float
            The distance between the tip and the detector.
        ξ : float
            The image compression factor which is defined as :math:`\\xi =
            \\frac{\\theta_{\\textnormal{crys}}}{\\theta_{\\textnormal{obs}}}`.

        Returns
        -------
        Ω : ndarray, shape (n,)
            The corresponding solid angles of each simplex.
        """
        #
        #
        def unit_arc_length(θ_1, θ_2, Φ_1, Φ_2):
            """
            Calculate arc length between two points on the unit sphere expressed
            through their angular coordinates θ_i, Φ_i.
            """
            return np.arccos(
                np.sin(θ_1) * np.sin(θ_2) * (
                    np.cos(Φ_1) * np.cos(Φ_2) + np.sin(Φ_1) * np.sin(Φ_2)
                ) + np.cos(θ_1) * np.cos(θ_2)
            )
        #
        #
        #
        #
        # x and y coordinates of all vertices/simplices, shape (n, 3)
        x, y = vertices[:, :, 0], vertices[:, :, 1]
        #
        #
        # calculate tip polar angles (i.e. trajectory launching angles),
        # shape (n, 3)
        θ_tip = np.arctan(np.sqrt(x * x + y * y) / L) * ξ
        #
        # calculate azimuthal detector angles, shape (n, 3)
        Φ_det = np.arctan2(y, x)
        #
        #
        # calculate edge lengths of spherical triangles by calculating the arc
        # lengths of each vertex pair on the unit sphere, shape (n, 3)
        lengths = np.column_stack([
            unit_arc_length(θ_tip[:, i], θ_tip[:, j], Φ_det[:, i], Φ_det[:, j]) for
            i, j in [(0, 1), (1, 2), (2, 0)]
        ])
        #
        # calculate semi-perimeters of spherical triangles, shape (n,)
        s = np.sum(lengths, axis = 1) / 2.0
        #
        #
        # calculate corresponding solid angles of each triangle (L'Huilier's
        # theorem), shape (n,)
        Ω = 4.0 * np.arctan(np.sqrt(
            np.tan( s                  / 2.0) *
            np.tan((s - lengths[:, 0]) / 2.0) *
            np.tan((s - lengths[:, 1]) / 2.0) *
            np.tan((s - lengths[:, 2]) / 2.0)
        ))
        #
        #
        # return solid angles
        return Ω
    #
    #
    #
    #
    def _map_curvatures(self, ΔH, T, κ_det):
        """
        Map relative Gaussian curvature from detector view to tip.

        Parameters
        ----------
        ΔH: ndarray, shape (n, n)
            The height profile of the tip surface, expressed as the deviation
            :math:`\\Delta H` from a perfect sphere.
        T: dict
            The dictionary containing the detector triangulation. See
            :meth:`_triangulate_detector()` for details.
        κ_det: ndarray, shape (m,)
            The relative Gaussian curvature from detector view.

        Returns
        -------
        κ_map: ndarray, shape (n, n)
            The relative Gaussian curvature mapped back to the tip.
        """
        #
        #
        # set xy detector position
        xy_det = self._map_tip_grid_to_detector(ΔH)
        #
        #
        # find simplex index for each xy detector position; -1 indicates invalid
        # simplex, i.e. outside detector view of tip
        tri_indices = np.full(ΔH.shape, -1, dtype = int)
        tri_indices[self._mask_tip] = T['tri'].find_simplex(xy_det)
        #
        #
        # map simplex indices to respective curvatures; NaN indicates invalid
        # curvature, i.e. outside detector view of tip
        κ_map = np.full(ΔH.shape, np.nan)
        κ_map[tri_indices != -1] = κ_det[tri_indices[tri_indices != -1]]
        #
        #
        #
        #
        # return mapped tip curvatures
        return κ_map
    #
    #
    #
    #
    def _map_detector_triangulation_to_tip(self, ΔH, tri_det):
        """
        Map detector triangulation to tip grid.

        Parameters
        ----------
        ΔH: ndarray, shape (n, n)
            The height profile of the tip surface, expressed as the deviation
            :math:`\\Delta H` from a perfect sphere.
        tri_det:
            The detector triangulation, as obtained by |delaunay|.

        Returns
        -------
        xy: ndarray, shape (n, n)
            The x and y tip positions of the *real* detector triangulation
            points.


        .. |delaunay| raw:: html

            <a href="https://docs.scipy.org/doc/scipy/reference/generated/
            scipy.spatial.Delaunay.html"
            target="_blank">scipy.spatial.Delaunay()</a>
        """
        #
        #
        # valid tip positions, shape (m, 2)
        xy_tip = np.column_stack((
            self._X_tip[self._mask_tip], self._Y_tip[self._mask_tip]
        ))
        #
        # barycentric coordinates of the *real* detector triangulation points in
        # terms of the mapped tip grid triangulation, shape (m, 3)
        B, T = self._get_barycentric_coordinates(
            {'tri': Delaunay(self._map_tip_grid_to_detector(ΔH))},
            tri_det.points
        )
        #
        # the vertices of the *real* detector triangulation points mapped back
        # *to* the tip, shape (m, 3, 2)
        V_tip = xy_tip[T['tri'].simplices[T['s']]]
        #
        #
        # x and y tip positions of the *real* detector triangulation points,
        # shape (m, 2)
        return np.einsum('ijk,ij->ik', V_tip, B)
    #
    #
    #
    #
    def _map_tip_grid_to_detector(self, ΔH):
        """
        Map :math:`x` and :math:`y` tip grid positions to corresponding detector
        plane positions.

        Note that the mapped detector positions may be located beyond the
        physical detector size.

        Parameters
        ----------
        ΔH: ndarray, shape (n, n)
            The height profile of the tip surface, expressed as the deviation
            :math:`\\Delta H` from a perfect sphere.

        Returns
        -------
        xy_det: ndarray, shape (m, 2)
            The corresponding :math:`x` and :math:`y` detector positions of the
            tip grid.
        """
        #
        #
        # calculate gradient of tip surface (reference sphere plus deviation ΔH)
        H_x = (self._grad_sphere[0] + self._d_dx(ΔH))[self._mask_tip]
        H_y = (self._grad_sphere[1] + self._d_dy(ΔH))[self._mask_tip]
        #
        #
        # calculate detection angle and radial distance
        θ_tip = np.arctan(np.sqrt(H_x**2 + H_y**2))
        r_det = np.tan(θ_tip / self._ξ) * self._L0
        #
        # calculate detector polar angle
        Φ_det = np.arctan2(H_y, H_x)
        #
        #
        # set xy detector positions
        return np.column_stack((r_det * np.cos(Φ_det), r_det * np.sin(Φ_det)))
    #
    #
    #
    #
    @staticmethod
    # @numba.njit(
    #     "Tuple((f8[:,:], f8))"
    #     "(f8[:,:], f8[:,:], f8[:,:], b1[:], f8[:], f8[:], f8[:], f8, f8)",
    #     cache = True, parallel = True
    # )
    def _normalize_curvature(κ, H, W, mask, X_c, Y_c, W_c, δ, r0):
        """
        Normalize relative Gaussian curvature :math:`\\kappa`.

        The method only provides the relative Gaussian curvature defined as
        :math:`\\kappa = \\frac{K}{\\Delta z}`, where :math:`\\Delta z` is the
        increment in tip depth direction according to the total reconstructed
        volume. However,the following normalization conditions holds:

        .. math::
            \\Delta z =
                \\frac{\\int \\kappa \\mathrm d A}{\\int \\mathrm d \\varOmega},

        where :math:`\\mathrm d A` is the area of the surface segments and
        :math:`\\varOmega` is the total solid angle covered by the surface
        segments.
        ``mask`` specifies all valid detector segments for the calculation of
        the surface area which are located completely within the (projected)
        circular tip cross-sectional area.

        Note that a certain amount of surface area/solid angle is not taken into
        account at the boundaries due to the underlying square grid. However,
        this discretization error vanishes for :math:`\\delta \\to 0`.


        Parameters
        ----------
        κ: ndarray, shape (n, n)
            The relative Gaussian curvature
            :math:`\\kappa = \\frac{K}{\\Delta z}`.
        H: ndarray, shape (n, n)
            The current height profile.
        W: ndarray, shape (n, n)
            The transition weights for the Gaussian curvature.
        mask: ndarray, shape ((n-1)**2,)
            The mask specifying the valid grid segments. It contains *m*
            ``True`` values for the valid grid segments.
        X_c: ndarray, shape (m,)
            The *x* positions of the centers of the valid grid segments.
        Y_c: ndarray, shape (m,)
            The *y* positions of the centers of the valid grid segments.
        W_c: ndarray, shape (m,)
            The transition weights for the Gaussian curvature in the the centers
            of the grid segments.
        δ: float
            The grid spacing.
        r0: float
            The tip radius.

        Returns
        -------
        K: ndarray, shape (n, n)
            The normalized Gaussian curvature.
        Δz: float
            The normalization factor.
        """
        #
        #
        # calculate height differences along x- and y-directions (first and
        # second axis), shape (m,)
        ΔH_x = (H[1:, :-1] - H[:-1, :-1]).flatten()[mask]
        ΔH_y = (H[:-1, 1:] - H[:-1, :-1]).flatten()[mask]
        #
        # calculate surface normals of respective surface segments approximated
        # as parallelograms, shape (3, m)
        N_c = np.stack((-δ * ΔH_x, -δ * ΔH_y, np.full_like(ΔH_x, δ**2)))
        #
        # calculate areas of surface segments (parallelograms), shape (m,)
        A_c = np.sqrt(np.sum(N_c**2, axis = 0))
        #
        #
        # calculate curvatures in the center of the valid grid segments
        # (average over the four adjacent grid points), shape (m,)
        κ_c = 0.25 * \
            (κ[:-1, :-1] + κ[:-1, 1:] + κ[1:, :-1] + κ[1:, 1:]).flatten()[mask]
        #
        # calculate heights in the center of the valid grid segments
        # (average over the four adjacent grid points), shape (m,)
        H_c = 0.25 * \
            (H[:-1, :-1] + H[:-1, 1:] + H[1:, :-1] + H[1:, 1:]).flatten()[mask]
        #
        # set position vectors of the surface segments, shape (3, m)
        R_c = np.stack((X_c, Y_c, H_c))
        #
        #
        # cumulate solid angles of surface segments; the surface normals N_c may
        # point inwards depending of the orientation of the hemisphere with
        # respect to the z-axis, so we take the absolute value to correct for
        # the sign of the dot product R_c * N_c
        Ω = np.abs(np.sum(
            np.sum(R_c * N_c, axis = 0) / np.sqrt(np.sum(R_c**2, axis = 0))**3
        ))
        #
        #
        # calculate normalization factor
        Δz = (Ω - np.sum((1.0 - W_c) * A_c) / r0**2) / np.sum(κ_c * W_c * A_c)
        #
        # normalize curvatures
        K = Δz * κ * W + (1.0 - W) / r0**2
        #
        #
        # return normalized curvatures and normalization factor
        return K, Δz
    #
    #
    #
    #
    @staticmethod
    def _polyfit2d(x, y, f, deg, weights = None):
        """
        Custom 2d fitting routine to allow for optional weights.

        Parameters
        ----------
        x: ndarray, shape (n,)
            The function arguments along *x*-direction.
        y: ndarray, shape (n,)
            The function arguments along *y*-direction.
        f: ndarray, shape (n,)
            The function values used for fitting.
        deg: int
            The polynomial degree used for fitting.
        weights: ndarray, shape (n,)
            The optional weights used for fitting. Defaults to ``None``, i.e.
            use equal weights for all sample points.

        Returns
        -------
        coeffs: ndarray, shape (deg + 1, deg + 1)
            The coefficients obtained through fitting. The result can be passed
            directly to ``numpy.polynomial.polynomial.polyval2d``.
        """
        #
        #
        # check for weight argument
        if weights is None:
            weights = np.ones_like(f)
        #
        #
        # create pseudo-Vandermonde matrix
        vander = polyvander2d(x, y, [deg, deg])
        #
        #
        # perform least-squares fit (with weights)
        c = np.linalg.lstsq(vander * np.sqrt(weights[:, np.newaxis]),
                            f * np.sqrt(weights), rcond = None)[0]
        #
        #
        # return coefficient matrix which can be passed directly to polyval2d
        return c.reshape(deg + 1, deg + 1)
    #
    #
    #
    #
    def _setup_tip(self, r0, n, ν):
        """
        Set up tip grid.

        Parameters
        ----------
        r0: float
            The tip radius (in nm)
        n: int
            The number of points used for the construction of the tip grid.
        ν: float
            The extrapolation fraction ν, which controls how far the curvature
            as seen by the detector should be extrapolated to the tip boundary.
        """
        #
        #
        # construct tip grid; use 'ij' indexing so that x increases along first
        # axis, y along second axis
        print("Setting up tip grid…")
        print("Using {0:d} x {0:d} points for tip grid.".format(n))
        self._X_tip, self._Y_tip = np.meshgrid(
            *[np.linspace(-r0, r0, n) for i in range(2)],
            indexing = 'ij'
        )
        #
        # set grid spacing
        self._δ = 2.0 * r0 / (n - 1)
        #
        # set radial distances and polar angles
        self._R_tip = np.sqrt(self._X_tip**2 + self._Y_tip**2)
        self._φ     = np.arctan2(self._Y_tip, self._X_tip)
        #
        #
        # set mask specifying data points close to or exact at boundary with
        # r = r0 (conservative choice)
        self._boundary_mask = np.isclose(self._R_tip, r0)
        #
        # set circular tip mask specifying grid points in the interior
        self._mask_tip = np.logical_and(self._R_tip < r0, ~self._boundary_mask)
        print(
            "Number of variation points for height profile is {0:d}.\n".
            format(np.count_nonzero(self._mask_tip))
        )
        #
        #
        #
        #
        # spherical reference height profile, shape (n, n)
        # the tip direction derived from the detector coordinate system points
        # into -z direction, so we need to use the negative height profile
        self._H_sphere = np.full_like(self._X_tip, np.nan)
        self._H_sphere[self._mask_tip] = \
            -np.sqrt(r0**2 - self._R_tip[self._mask_tip]**2)
        #
        #
        # set gradient of sphere (first derivatives)
        self._grad_sphere = (
            -self._X_tip / self._H_sphere,
            -self._Y_tip / self._H_sphere
        )
        #
        # calculate sparse (upper triangle) Hessian matrix of sphere (second
        # derivatives)
        self._Hessian_sphere = (
            (self._Y_tip**2 - r0**2)   / self._H_sphere**3,
            -self._X_tip * self._Y_tip / self._H_sphere**3,
            (self._X_tip**2 - r0**2)   / self._H_sphere**3
        )
        #
        # set height at the boundary after calculation of the derivatives due to
        # their divergence
        self._H_sphere[self._boundary_mask] = 0.0
        #
        #
        #
        #
        # transition width is calculated based on aperture angle and
        # extrapolation fraction ν, which controls how far the curvature as seen
        # by the detector should be extrapolated to the tip boundary
        w0 = (1.0 - ν) * self._r0 * (1.0 - np.sin(self._ω))
        #
        # transition weights for the Gaussian curvature, shape (n, n)
        # w(r = r0)      = 0 at the boundary
        # w(r = r0 - w0) = 1 in the interior
        trans_mask = np.logical_and(self._mask_tip, self._R_tip >= r0 - w0)
        self._weights = np.full_like(self._R_tip, np.nan)
        self._weights[trans_mask] = \
            np.sqrt(w0**2 - (self._R_tip[trans_mask] - r0 + w0)**2) / w0
        self._weights[self._R_tip < r0 - w0] = 1.0
        self._weights[self._boundary_mask]   = 0.0
        #
        #
        #
        #
        # interior points used for the extrapolation of height profile ΔH
        self._mask_ΔH_inter = np.logical_and(
            self._mask_tip, self._R_tip > self._r0 - np.sqrt(2.0) * self._δ
        )
        self._mask_ΔH_inter = np.logical_or(
            self._mask_ΔH_inter,
            np.isclose(self._R_tip, self._r0 - np.sqrt(2.0) * self._δ)
        )
        #
        # exterior points at which to extrapolate height profile ΔH
        self._mask_ΔH_extra = np.logical_and(
            ~self._mask_tip, self._R_tip < self._r0 + np.sqrt(2.0) * self._δ
        )
        self._mask_ΔH_extra = np.logical_or(
            self._mask_ΔH_extra,
            np.isclose(self._R_tip, self._r0 + np.sqrt(2.0) * self._δ)
        )
        #
        #
        #
        #
        # set mask specifying valid grid segments for curvature normalization,
        # containing m True values, shape ((n-1)**2,)
        self._mask_norm = np.logical_and.reduce((
            ~np.isnan(self._H_sphere[ :-1,  :-1]),
            ~np.isnan(self._H_sphere[ :-1, 1:]),
            ~np.isnan(self._H_sphere[1:,    :-1]),
            ~np.isnan(self._H_sphere[1:,   1:])
        )).flatten()
        #
        #
        # set grid center positions, shape (m,)
        self._X_center_norm = \
            (self._X_tip[:-1, :-1] + 0.5 * self._δ).flatten()[self._mask_norm]
        self._Y_center_norm = \
            (self._Y_tip[:-1, :-1] + 0.5 * self._δ).flatten()[self._mask_norm]
        #
        # set transition weights in grid centers, shape (m,)
        self._weights_center_norm = 0.25 * (
            self._weights[ :-1, :-1] + self._weights[ :-1, 1:] +
            self._weights[1:,   :-1] + self._weights[1:,   1:]
        ).flatten()[self._mask_norm]
        #
        #
        #
        #
        # initialize finite differences
        acc = 2
        self._d_dx    = FinDiff(0, self._δ, 1, acc = acc)
        self._d_dy    = FinDiff(1, self._δ, 1, acc = acc)
        self._d2_dx2  = FinDiff(0, self._δ, 2, acc = acc)
        self._d2_dy2  = FinDiff(1, self._δ, 2, acc = acc)
        self._d2_dxdy = FinDiff((0, self._δ, 1), (1, self._δ, 1), acc = acc)
    #
    #
    #
    #
    def _triangulate_detector(self, sl, N_simp, mode, tri, verbose):
        """
        Triangulate detector events.

        Parameters
        ----------
        sl: NumPy IndexExpression object
            The NumPy slice representing the data range to use, as obtained by
            ``numpy.s_[]``.
        N_simp: int
            The (approximate) target number of simplices for the detector
            triangulation.
        mode: str
            The triangulation mode. Must be either ``height`` or ``positions``
            for the reconstruction of the height profile or three-dimensional
            atomic positions, respectively.
        tri: class
            The custom detector triangulation.
        verbose: bool
            Whether to print additional information.

        Returns
        -------
        tri: dict
            The dictionary for the detector triangulation. If an internal error
            occurred, ``None`` is returned.
        """
        #
        #
        def _printv(msg):
            """
            Small helper for printing messages only in verbose mode.
            """
            if verbose: print(msg)
        #
        #
        def _triangulate(N_simp):
            """
            Helper for the adaptive triangulation of the detector.
            """
            #
            #
            # set target number for events per simplex
            N_ref = len(self._xy_data[sl]) // N_simp
            _printv(
                "\tTarget number of events per simplex is {0:d}.".format(N_ref)
            )
            #
            #
            # empirical guess for the number of grid points; the number of
            # (square) grid segments amounts to approximately 6 times the number
            # of requested simplices
            N_grid = int(np.sqrt(N_simp * 6.0 * 4.0 / np.pi))
            _printv(
                "\tUsing square grid with {0:d} points for estimation of "
                "detector density.".format(N_grid)
            )
            #
            # set grid spacing
            δ = 2.0 * self._R0 / (N_grid - 1.0)
            #
            #
            #
            #
            # calculate 2d histogram data and bin centers
            counts, x_edge, y_edge, _ = binned_statistic_2d(
                *self._xy_data[sl].T, values = None,
                statistic = 'count', bins = N_grid,
                range = ([(-self._R0 - δ / 2, self._R0 + δ / 2)] * 2)
            )
            x_c = (x_edge[1:] + x_edge[:-1]) / 2
            y_c = (y_edge[1:] + y_edge[:-1]) / 2
            #
            #
            # calculate corresponding detector densities
            ϱ = counts / δ**2
            #
            #
            #
            #
            # create initial (coarse) detector triangulation
            #
            #
            # polar angles for points on circumference
            n_φ = 24
            φ   = np.linspace(0.0, 2.0 * np.pi, num = n_φ, endpoint = False)
            #
            # inner ring
            R1 = self._R0 * (1.0 - 2.0 * np.pi / n_φ)
            #
            # inner square grid
            X, Y = np.meshgrid(
                *[np.linspace(-R1 / 2, R1 / 2, 3) for i in range(2)]
            )
            #
            # stack all points
            P = np.vstack((
                # outer polar grid
                np.column_stack((self._R0 * np.cos(φ), self._R0 * np.sin(φ))),
                # inner polar grid
                np.column_stack((R1 * np.cos(φ), R1 * np.sin(φ))),
                # inner square grid
                np.column_stack((X.ravel(), Y.ravel()))
            ))
            #
            #
            # triangulate detector grid
            tri = Delaunay(P, incremental = True)
            #
            #
            #
            #
            # iteratively sub-divide simplices to obtain approximately the same
            # number of events in each simplex
            for i in range(15):
                # set vertices, shape (n, 3, 2)
                vertices = tri.points[tri.simplices]
                #
                #
                # interpolate density at vertices, shape (n, 3)
                ϱ_vertices = np.column_stack(
                    [interpn((x_c, y_c), ϱ, vertices[:, i]) for i in range(3)]
                )
                #
                # calculate average density, shape (n,)
                ϱ_tri = np.mean(ϱ_vertices, axis = 1)
                #
                #
                # approximate events in each simplex, shape (n,)
                N_tri = ϱ_tri * 0.5 * np.abs(np.cross(
                    vertices[:, 2] - vertices[:, 0],
                    vertices[:, 1] - vertices[:, 0]
                ))
                #
                #
                # set mask specifying simplices to be further divided
                is_divide = (N_tri / N_ref > 2.0)
                if np.count_nonzero(is_divide) == 0:
                    break
                #
                # add points to existing triangulation
                tri.add_points(np.mean(vertices, axis = 1)[is_divide])
            #
            #
            # finalize triangulation
            tri.close()
            _printv(
                "\tNumber of triangulated vertices/simplices is {0:d}/{1:d}."
                .format(len(tri.points), len(tri.simplices))
            )
            #
            #
            # return detector triangulation
            return tri
        #
        #
        #
        #
        # start timer for triangulation
        print("Triangulating events…")
        start = timer()
        #
        #
        # check mode for internal consistency
        if mode != "height" and mode != "positions":
            raise Exception(
                "Internal error: Invalid mode \"{0:s}\" specified.".format(mode)
            )
        #
        #
        #
        #
        # use adaptive triangulation if not provided
        if tri is None:
            _printv("\tUsing internal adaptive triangulation.")
            tri = _triangulate(N_simp)
        else:
            _printv("\tUsing user-provided triangulation.")
        #
        #
        #
        #
        # triangulate data
        simplex_indices = tri.find_simplex(self._xy_data[sl])
        #
        # set mask specifying valid events inside triangulation
        is_valid = (simplex_indices != -1)
        #
        #
        # get number of events in each simplex
        counts = np.bincount(
            simplex_indices[is_valid], minlength = len(tri.simplices)
        )
        _printv(
            "\tNumber of triangulated events is {0:d} (outside: {1:d})."
            .format(np.sum(counts), len(simplex_indices) - np.sum(counts))
        )
        _printv(
            "\tAverage number of events per simplex is {0:.1f} "
            "(min/max: {1:d}/{2:d}; std: {3:.1f}).".
            format(np.mean(counts), counts.min(), counts.max(), np.std(counts))
        )
        #
        #
        # check for empty segments in "height" mode; detector density would
        # diverge
        if mode == "height" and np.count_nonzero(counts == 0) > 0:
            warnings.warn(
                "Empty detector segment detected. You may want to decrease the "
                "number of simplices using the \"N_simp\" argument or provide "
                "a custom triangulation using the \"tri\" argument."
            )
            return None
        #
        #
        #
        #
        # for "positions" mode, we do not need the solid angles
        if mode == "positions":
            _printv("\tTriangulation took {0:.3f} s.".format(timer() - start))
            return {
                'tri'     : tri,
                's'       : simplex_indices,
                'is_valid': is_valid,
            }
        #
        #
        #
        #
        # calculate geometric center of each simplex
        C = np.mean(tri.points[tri.simplices], axis = 1)
        #
        # calculate corresponding detection angle of each simplex
        θ_det = np.arctan(np.linalg.norm(C, axis = 1) / self._L0) * self._ξ
        #
        #
        # calculate solid angles
        Ω = self._get_triangulated_solid_angles(
            tri.points[tri.simplices], self._L0, self._ξ
        )
        _printv(
            "\tExpected solid angle based on aperture is {0:.3f} sr "
            "({1:.1f}% of hemisphere).\n"
            "\tTotal solid angle based on triangulation is {2:.3f} sr.".format(
                2.0 * np.pi * (1.0 - np.cos(self._ω)),
                (1.0 - np.cos(self._ω)) * 100,
                np.sum(Ω)
            )
        )
        #
        #
        # return detector triangulation dictionary
        _printv("\tTriangulation took {0:.3f} s.".format(timer() - start))
        return {
            'tri'     : tri,
            's'       : simplex_indices,
            'is_valid': is_valid,
            'θ'       : θ_det,
            'Ω'       : Ω
        }
