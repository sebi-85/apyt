"""
The APyT Package: From Raw Atom Probe Data to Three-Dimensional Reconstruction
==============================================================================

The APyT package is an advanced, open-source Python framework for evaluating
atom probe tomography (APT) data. It offers a suite of modules that automate key
steps in APT processing, from mass spectrum calibration to three-dimensional
sample reconstruction. Its modular architecture ensures seamless integration
with external tools through standardized input/output interfaces, supporting
both Linux and Windows environments. Users can import and use each module
independently, or integrate them in custom workflows or GUI applications.

Key features include high efficiency with NumPy and Numba, a low memory
footprint, and extensive documentation. The modules are highly automated,
requiring minimal user input to achieve accurate results. The package also
integrates SQL database management for raw measurement data and corresponding
metadata.

In addition to its core processing capabilities, APyT includes a growing
collection of specialized analysis tools. These currently support tasks such as
chemical composition analysis of reconstructed datasets and crystallographic
investigations using methods like spatial distribution maps (SDMs). This
analysis suite is designed to expand over time, with new tools and techniques
added in response to user needs.

Planned future developments include a PyQt-based graphical user interface (GUI)
that will unify access to all APyT modules within a single application. This GUI
aims to simplify the user experience, making the framework more accessible to
researchers unfamiliar with Python while maintaining the flexibility and
performance valued by advanced users.


Available subpackages
---------------------

.. toctree::
   :maxdepth: 1

   The APyT analysis subpackage (apyt.analysis)<apyt.analysis>
   The APyT file input/output subpackage (apyt.io)<apyt.io>
   The APyT reconstruction subpackage (apyt.reconstruction)<apyt.reconstruction>
   The APyT mass spectrum subpackage (apyt.spectrum)<apyt.spectrum>


.. sectionauthor:: Sebastian M. Eich <Sebastian.Eich@imw.uni-stuttgart.de>
"""
__version__ = '0.1.0'
