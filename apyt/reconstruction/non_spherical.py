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
#
#
#
#
# import modules
import matplotlib.pyplot as plt
import numba
import numpy as np
import warnings
#
# import some special functions/modules
from findiff import FinDiff
from mpl_toolkits.axes_grid1 import make_axes_locatable
from numpy.polynomial.polynomial import polyval2d, polyvander2d
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from scipy.optimize import root
from scipy.spatial import ConvexHull, Delaunay
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
    num_points_det: int
        The number of points used for the construction of the detector grid.
        Must be an odd number. Defaults to ``25``.
    num_points_tip: int
        The number of points used for the construction of the tip grid. Must be
        an odd number. Defaults to ``101``.
    maxiter: int
        The maximum number of iterations for the root solver. See also
        :attr:`maxiter`. Defaults to ``100``.
    tol: float
        The maximum absolute relative curvature residual allowed for the
        reconstruction of the height profile. See also :attr:`tol`. Defaults to
        ``0.01``.


    The following instance attributes can be accessed and modified:

    Attributes
    ----------
    maxiter: int
        The maximum number of iterations for the root solver. Defaults to
        ``100``.
    tol: float
        The maximum absolute relative curvature residual allowed for the
        reconstruction of the height profile. Defaults to ``0.01``.


    The following class methods are provided:

    * :meth:`debug_plot`: Show plots for debugging.
    * :meth:`reconstruct_height_profile`: Reconstruct tip
      height profile based on Gaussian curvature.


    The following **general** instance properties can be accessed (*read-only*):

    * :attr:`ω`: The aperture angle.


    The following **detector-related** instance properties can be accessed
    (*read-only*):

    * :attr:`X_det`: The *x* positions of the detector grid.
    * :attr:`Y_det`: The *y* positions of the detector grid.
    * :attr:`mask_det`: The Boolean mask specifying valid circular points.
    * :attr:`tri`: The detector triangulation.
    * :attr:`Ω`: The triangulated solid angles.


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
        self, xy_data, V_at, R0, r0, L0, ξ,
        num_points_det = 25, num_points_tip = 101, maxiter = 100, tol = 1e-2
    ):
        #
        #
        # set instance attributes
        self._V_at   = V_at
        self._r0     = r0
        self._L0     = L0
        self._ξ      = ξ
        self.maxiter = maxiter
        self.tol     = tol
        #
        #
        # set aperture angle
        self._ω = np.arctan(R0 / self._L0) * self._ξ
        print("(Half) aperture angle is {0:.2f}°.\n".format(np.rad2deg(self._ω)))
        #
        #
        # set up detector grid
        self._setup_detector(xy_data, R0, num_points_det, L0, ξ)
        #
        #
        # set up tip grid
        self._setup_tip(r0, num_points_tip, 0.5)
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
    ###     Detector-related properties                                      ###
    ###                                                                      ###
    ############################################################################
    @property
    def X_det(self):
        """Getter for the internal ``_X_det`` attribute (*read-only*).

        Returns
        -------
        X_det: ndarray, shape (N, N)
            The *x* positions of the detector grid, as returned by the
            ``numpy.meshgrid()`` function.
        """
        return self._X_det
    #
    @property
    def Y_det(self):
        """Getter for the internal ``_Y_det`` attribute (*read-only*).

        Returns
        -------
        Y_det: ndarray, shape (N, N)
            The *y* positions of the detector grid, as returned by the
            ``numpy.meshgrid()`` function.
        """
        return self._Y_det
    #
    @property
    def mask_det(self):
        """Getter for the internal ``_mask_det`` attribute (*read-only*).

        Returns
        -------
        mask_det: ndarray, shape (N, N)
            The Boolean mask specifying valid circular points on the square
            detector grid.
        """
        return self._mask_det
    #
    @property
    def tri(self):
        """Getter for the internal ``_tri`` attribute (*read-only*).

        Returns
        -------
        tri: class
            The Delaunay tessellation of the detector, as returned by
            |delaunay|. See :meth:`CurvatureReconstructor.debug_plot` on how to
            use this in plots.


        .. |delaunay| raw:: html

            <a href="https://docs.scipy.org/doc/scipy/reference/generated/
            scipy.spatial.Delaunay.html"
            target="_blank">scipy.spatial.Delaunay()</a>
        """
        return self._tri
    #
    @property
    def Ω(self):
        """Getter for the internal ``_Ω`` attribute (*read-only*).

        Returns
        -------
        Ω :ndarray, shape (M,)
            The triangulated solid angles. See
            :meth:`CurvatureReconstructor.debug_plot` on how to use this in
            plots.
        """
        return self._Ω
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
    def debug_plot(self, sl, H, Δz, results):
        """
        Show plots for debugging.

        Parameters
        ----------
        sl: NumPy IndexExpression object
            The NumPy slice representing the data range to use.
        ΔH: ndarray, shape (n, n)
            The reconstructed tip height profile.
        Δz: float
            The corresponding :math:`\\Delta z` increment for the given data
            range (in nm).
        results: dict
            A dictionary containing various results for the reconstruction of
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
        #
        def _hull_plotter(ax):
            """
            Simple helper to plot the convex hull of the tip boundary as seen by
            the detector.
            """
            #
            for simplex in hull.simplices:
                ax.plot(
                    mapped_points[simplex, 0], mapped_points[simplex, 1], 'k-'
                )
        #
        #
        #
        #
        # stack mapped tip points, shape (m, 2)
        mapped_points = np.column_stack((
                self._X_tip[results['mask_mapped']],
                self._Y_tip[results['mask_mapped']]
        ))
        #
        # create convex hull for mapped tip points
        hull = ConvexHull(mapped_points)
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
            "\"{0:s}\".".format(str(sl))
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
            self._X_det[self._mask_det],
            self._Y_det[self._mask_det],
            self._tri.simplices,
            facecolors = results['det_dens']
        )
        fig.colorbar(
            tric,
            cax = make_axes_locatable(ax).append_axes(
                "right", size = "5%", pad = 0.10
            )
        )
        #
        #
        # triangulated Gaussian curvature (detector)
        ax = axes[0][1]
        ax.set_title("Triangulated Gaussian curvature $K$ ($\mathrm{nm}^{-2}$)")
        ax.set_aspect('equal')
        ax.set_xlabel('$x_\mathrm{det}$ (mm)')
        ax.set_ylabel('$y_\mathrm{det}$ (mm)')
        tric = ax.tripcolor(
            self._X_det[self._mask_det],
            self._Y_det[self._mask_det],
            self._tri.simplices,
            facecolors = results['det_curv'],
            vmin = 0.5 / self._r0**2, vmax = 2.0 / self._r0**2
        )
        fig.colorbar(
            tric,
            cax = make_axes_locatable(ax).append_axes(
                "right", size = "5%", pad = 0.10
            )
        )
        #
        #
        # mapped tip target Gaussian curvature
        ax = axes[0][2]
        ax.set_aspect('equal')
        ax.set_title("Mapped target tip curvature $K$ ($\mathrm{nm}^{-2}$)")
        ax.set_xlabel('$x_\mathrm{tip}$ (nm)')
        ax.set_ylabel('$y_\mathrm{tip}$ (nm)')
        pcm = ax.pcolormesh(
            self._X_tip, self._Y_tip, results['tip_curv_from_det'],
            shading = 'nearest',
            vmin = 0.5 / self._r0**2, vmax = 2.0 / self._r0**2
        )
        _hull_plotter(ax)
        _boundary_plotter(ax)
        fig.colorbar(
            pcm,
            cax = make_axes_locatable(ax).append_axes(
                "right", size = "5%", pad = 0.10
            )
        )
        #
        #
        # reconstructed tip Gaussian curvature
        ax = axes[1][0]
        ax.set_aspect('equal')
        ax.set_title("Reconstructed tip curvature $K$ ($\mathrm{nm}^{-2}$)")
        ax.set_xlabel('$x_\mathrm{tip}$ (nm)')
        ax.set_ylabel('$y_\mathrm{tip}$ (nm)')
        pcm = ax.pcolormesh(
            self._X_tip, self._Y_tip, results['tip_curv_from_height'],
            shading = 'nearest',
            vmin = 0.5 / self._r0**2, vmax = 2.0 / self._r0**2
        )
        _hull_plotter(ax)
        _boundary_plotter(ax)
        fig.colorbar(
            pcm,
            cax = make_axes_locatable(ax).append_axes(
                "right", size = "5%", pad = 0.10
            )
        )
        #
        #
        # relative curvature residuals
        ax = axes[1][1]
        ax.set_aspect('equal')
        ax.set_title("Relative curvature residuals")
        ax.set_xlabel('$x_\mathrm{tip}$ (nm)')
        ax.set_ylabel('$y_\mathrm{tip}$ (nm)')
        pcm = ax.pcolormesh(
            self._X_tip, self._Y_tip, results['curv_residuals'],
            shading = 'nearest',
            vmin = -2.0 * self.tol, vmax = 2.0 * self.tol
        )
        _hull_plotter(ax)
        _boundary_plotter(ax)
        fig.colorbar(
            pcm,
            cax = make_axes_locatable(ax).append_axes(
                "right", size = "5%", pad = 0.10
            )
        )
        #
        #
        # reconstructed height profile
        ax = axes[1][2]
        ax.set_aspect('equal')
        ax.set_title("Reconstructed height profile $\Delta H$ ($\mathrm{nm}$)")
        ax.set_xlabel('$x_\mathrm{tip}$ (nm)')
        ax.set_ylabel('$y_\mathrm{tip}$ (nm)')
        pcm = ax.pcolormesh(
            self._X_tip, self._Y_tip, H - self._H_sphere, shading = 'nearest'
        )
        _hull_plotter(ax)
        _boundary_plotter(ax)
        fig.colorbar(
            pcm,
            cax = make_axes_locatable(ax).append_axes(
                "right", size = "5%", pad = 0.10
            )
        )
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
        self, sl, H_0 = None, verbose = False, debug_plot = False
    ):
        """
        Reconstruct tip height profile for given data interval.

        The core routine uses the |krylov| root finding algorithm from the SciPy
        |optimize| package for the reconstruction of the height profile.


        Parameters
        ----------
        sl: NumPy IndexExpression object
            The NumPy slice representing the data range to use, as obtained by
            ``numpy.s_[]``.
        H_0: ndarray, shape (n, n)
            The initial guess of the height profile. Defaults to a perfect
            hemisphere if not provided.
        verbose: bool
            Whether to print the status after each iteration of the root solver.
            Defaults to ``False``.
        debug_plot: bool
            Whether to show debug plots with various information. Defaults to
            ``False``. Setting this parameter to ``True`` is equivalent to
            calling :meth:`debug_plot` with the returned results.

        Returns
        -------
        H: ndarray, shape (n, n)
            The reconstructed height profile.
        Δz: float
            The corresponding :math:`\\Delta z` increment for the given data
            range.
        results: dict
            A dictionary containing various results for the reconstruction of
            the height profile. Use ``results.keys()`` to get a list of the
            dictionary content.


        .. |krylov| raw:: html

            <a href="https://docs.scipy.org/doc/scipy/reference/
            optimize.root-krylov.html" target="_blank">Krylov</a>

        .. |optimize| raw:: html

            <a href="https://docs.scipy.org/doc/scipy/reference/optimize.html"
            target="_blank">Optimization and root finding</a>
        """
        #
        #
        start = timer()
        print(
            "Processing data interval specified by object \"{0:s}\"…".
            format(str(sl))
        )
        #
        #
        # initialize height profile if not provided
        if H_0 is None:
            ΔH_0 = np.zeros_like(self._X_tip)
        else:
            ΔH_0 = H_0 - self._H_sphere
        #
        #
        #
        #
        # calculate cumulated detector volumes
        V = self._get_detector_volume(sl)
        #
        # calculate relative Gaussian curvature based on detector density,
        # shape (M,); indexing is based on detector triangulation
        κ_det = self._Ω * np.cos(self._θ_det) / V
        #
        #
        #
        #
        # use Krylov root finding algorithm for height profile reconstruction
        sol = root(
            self._calculate_curvature_residuals, ΔH_0[self._mask_tip],
            args = (κ_det,),
            method = 'krylov',
            options = {
                'maxiter': self.maxiter,
                'fatol'  : self.tol,
                'disp'   : verbose
            }
        )
        #
        #
        #
        #
        # apply solution to get optimized reconstructed height profile
        ΔH = self._extrapolate_height_profile(sol.x)
        #
        # calculate curvature from detector view
        K_from_det, Δz, mask_mapped = \
            self._get_tip_curvature_from_detector_view(κ_det, ΔH)
        #
        # calculate curvature from height profile
        K_from_height = self._get_tip_curvature_from_height_profile(ΔH)
        #
        # calculate curvature residuals, shape (n, n)
        residuals = np.full_like(self._X_tip, np.nan)
        residuals[self._mask_tip] = \
            self._calculate_curvature_residuals(ΔH[self._mask_tip], (κ_det))
        #
        #
        # collect all relevant results in a dictionary
        results_dict = {
            'det_dens'            : self._get_detector_volume(sl) / self._Ω,
            'det_curv'            : κ_det * Δz,
            'tip_curv_from_det'   : K_from_det,
            'tip_curv_from_height': K_from_height,
            'curv_residuals'      : residuals,
            'mask_mapped'         : mask_mapped
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
            print(
                "Solver found a solution with maximum absolute relative "
                "curvature residual of {0:.6f} "
                "(mean: {1:.6f}; standard deviation: {2:.6f}).".
                format(np.abs(sol.fun).max(), np.mean(sol.fun), np.std(sol.fun))
            )
        print(
            "Δz increment is {0:.3f} nm (spherical estimate is {1:.3f} nm).".
            format(Δz, np.sum(V) / (np.pi * (self._r0 * np.sin(self._ω))**2)))
        print(
            "Surface reconstruction took {0:.3f} s ({1:d} iterations).\n".
            format(timer() - start, sol.nit)
        )
        #
        #
        #
        #
        # show debug plots if requested
        if debug_plot == True:
            self.debug_plot(sl, self._H_sphere + ΔH, Δz, results_dict)
        #
        #
        #
        #
        # return height profile, Δz increment, and results dictionary
        return self._H_sphere + ΔH, Δz, results_dict
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
        Calculate residuals of Gaussian curvature

        Parameters
        ----------
        ΔH_in: ndarray, shape (m,)
            A NumPy array representing the surface.
        args: tuple of length 1
            The one-tuple containing the relative Gaussian curvature based on
            the detector view.

        Returns
        -------
        ΔK: shape (m,)
            The (relative) residuals of the Gaussian curvature.
        """
        #
        #
        # set relative Gaussian curvature based on detector view
        κ_det = args[0]
        #
        #
        # extrapolate height profile to allow calculation of finite differences,
        # shape (n, n)
        ΔH = self._extrapolate_height_profile(ΔH_in)
        #
        #
        # get target Gaussian curvature of tip based on detector view
        K_from_det, _, _ = self._get_tip_curvature_from_detector_view(κ_det, ΔH)
        #
        #
        # get current Gaussian curvature based on height profile
        K_from_height = self._get_tip_curvature_from_height_profile(ΔH)
        #
        #
        # calculate relative curvature residuals
        res = (K_from_height - K_from_det) / K_from_det
        #
        #
        # return residuals for variation points
        return res[self._mask_tip]
    #
    #
    #
    #
    def _extrapolate_curvature(
        self, κ, mask_mapped, mode = "generic", apply_filter = True
    ):
        """
        Extrapolate relative Gaussian curvature.

        Parameters
        ----------
        κ: ndarray, shape (n, n)
            The Relative Gaussian curvature
            :math:`\\kappa = \\frac{K}{\\Delta z}`. Invalid points are
            represented by ``numpy.nan``.
        mask_mapped: ndarray, shape (n, n)
            The mask specifying the mapped positions on the tip.
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
        # generic extrapolation using fit
        if mode == "generic":
            # fit 2d surface to relative curvature
            c = self._polyfit2d(
                self._X_tip[mask_mapped], self._Y_tip[mask_mapped],
                κ[mask_mapped], 1
            )
            #
            #
            # extrapolate relative curvature using fit
            κ[~mask_mapped] = polyval2d(
                self._X_tip[~mask_mapped], self._Y_tip[~mask_mapped], c
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
        # return extrapolated height profile for entire grid
        return ΔH
    #
    #
    #
    #
    def _get_detector_volume(self, sl):
        """
        Calculate detector volume.

        Parameters
        ----------
        sl: NumPy IndexExpression object
            The NumPy slice representing the data range to use.

        Returns
        -------
        V_det: shape (m,)
            The detector volumes, indexed by the triangulation of the detector.
        """
        #
        #
        simplex_indices = self._simplex_indices[sl]
        #
        # filter invalid measurement data not covered by triangulation
        # (close to detector edge)
        mask = (simplex_indices != -1)
        simplex_indices = simplex_indices[mask]
        #
        #
        # cumulate atomic volumes in each simplex
        V = np.bincount(simplex_indices, weights = self._V_at[sl][mask])
        #
        #
        # return cumulated atomic volumes
        return V
    #
    #
    #
    #
    def _get_tip_curvature_from_detector_view(self, κ_det, ΔH):
        """
        Calculate Gaussian curvature based on detector view.

        Parameters
        ----------
        κ_det: ndarray, shape (M,)
            The relative Gaussian curvature based on detector view.
        ΔH: ndarray, shape (n, n)
            The height profile of the tip surface, expressed as the deviation
            :math:`\\Delta H` from a perfect sphere.

        Returns
        -------
        K: shape (n, n)
            The corresponding Gaussian curvature of the height profile.
        Δz: float
            The normalization factor.
        mask_mapped: ndarray, shape (n, n)
            The mask specifying the mapped positions on the tip.
        """
        #
        #
        # get relative Gaussian curvature based on detector view *and* current
        # height profile, shape (n, n)
        κ_map, mask_mapped = self._map_curvatures(κ_det, ΔH)
        #
        #
        # get target Gaussian curvature based on detector view, extrapolation, and
        # normalization
        κ_ext = self._extrapolate_curvature(κ_map, mask_mapped)
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
        # return normalized curvature,normalization factor, and mapping mask
        return K, Δz, mask_mapped
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
    def _map_curvatures(self, κ_det, ΔH):
        """
        Map relative Gaussian curvature from detector view to tip.

        Parameters
        ----------
        κ_det: ndarray, shape (m,)
            The relative Gaussian curvature from detector view.
        ΔH: ndarray, shape (n, n)
            The height profile of the tip surface, expressed as the deviation
            :math:`\\Delta H` from a perfect sphere.

        Returns
        -------
        κ_map: ndarray, shape (n, n)
            The relative Gaussian curvature mapped back to the tip.
        mask_mapped: ndarray, shape (n, n)
            The mask specifying the mapped positions on the tip.
        """
        #
        #
        # calculate gradient of tip surface (reference sphere plus deviation ΔH)
        H_x = (self._grad_sphere[0] + self._d_dx(ΔH))[self._mask_tip]
        H_y = (self._grad_sphere[1] + self._d_dy(ΔH))[self._mask_tip]
        #
        #
        #
        #
        # get detection angle and radial distance
        θ_tip = np.arctan(np.sqrt(H_x**2 + H_y**2))
        r_det = np.tan(θ_tip / self._ξ) * self._L0
        #
        # detector coordinate system is defined from view towards tip; this
        # translates into a rotation of 180° with respect to tip coordinate system
        Φ_det = np.arctan2(H_y, H_x)# + np.pi
        #
        #
        # set xy detector position
        x_det = r_det * np.cos(Φ_det)
        y_det = r_det * np.sin(Φ_det)
        #
        #
        #
        #
        # find simplex index for each xy detector position; -1 indicates invalid
        # simplex, i.e. outside detector view of tip
        tri_indices = np.full(ΔH.shape, -1, dtype = int)
        tri_indices[self._mask_tip] = self._tri.find_simplex(
            np.column_stack((x_det, y_det))
        )
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
        return κ_map, ~np.isnan(κ_map)
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
    def _setup_detector(self, xy_data, R0, N, L0, ξ):
        """
        Set up detector grid.

        Parameters
        ----------
        xy_data: ndarray, shape (n, 2)
            The *x* and *y* detector positions of **all** events.
        R0: float
            The detector radius (in mm)
        N: int
            The number of points used for the construction of the detector grid.
        L0: float
            The distance between tip and detector (mm)
        ξ: float
            The image compression factor.
        """
        #
        #
        # start timer for triangulation
        print("Setting up detector grid …")
        start = timer()
        #
        #
        # construct detector grid
        print("Using {0:d} x {0:d} points for detector grid.".format(N))
        self._X_det, self._Y_det = np.meshgrid(
            *[np.linspace(-R0, R0, N) for i in range(2)],
            indexing = 'ij'
        )
        #
        # set circular detector mask
        self._mask_det = (self._X_det**2 + self._Y_det**2 <= R0**2)
        #
        #
        # convert numpy.meshgrid to points of shape (M, 2)
        P = np.column_stack(
            (self._X_det[self._mask_det], self._Y_det[self._mask_det])
        )
        #
        # triangulate detector grid
        self._tri = Delaunay(P)
        #
        #
        # calculate geometric center of each simplex
        C = np.sum(P[self._tri.simplices], axis = 1) / 3.0
        #
        # calculate corresponding detection angle of each simplex
        self._θ_det = np.arctan(np.linalg.norm(C, axis = 1) / L0) * ξ
        #
        #
        # calculate solid angles
        self._Ω = self._get_triangulated_solid_angles(
            P[self._tri.simplices], L0, ξ
        )
        print(
            "Expected solid angle based on aperture is {0:.3f} sr "
            "({1:.1f}% of hemisphere).\n"
            "Total solid angle based on triangulation is {2:.3f} sr.".format(
                2.0 * np.pi * (1.0 - np.cos(self._ω)),
                (1.0 - np.cos(self._ω)) * 100,
                np.sum(self._Ω)
            )
        )
        #
        #
        # find simplex indices for *all* events
        print("Triangulating data …")
        self._simplex_indices = self._tri.find_simplex(xy_data)
        print(
            "Detector triangulation took {0:.3f} s.\n".
            format(timer() - start)
        )
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
        print("Setting up tip grid …")
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
