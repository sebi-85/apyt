"""
APyT: A modular open-source Python framework for atom probe tomography data evaluation
======================================================================================

The APyT package is an advanced, open-source Python framework for evaluating
atom probe tomography (APT) data. It offers a suite of modules that automate key
steps in APT processing, from mass spectrum calibration to three-dimensional
sample reconstruction. Its modular architecture ensures seamless integration
with external tools through standardized input/output interfaces, supporting
both Linux and Windows environments. Users can import and use each module
independently, or integrate them in custom workflows or GUI applications.

For ready-to-use scenarios or testing, a :doc:`command line interface<apyt_cli>`
(CLI) is provided. The CLI offers lightweight wrappers around the core modules,
enabling users to process raw measurement data without writing additional Python
code. It is particularly useful for exploratory analysis, quick prototyping, and
stand-alone usage.

Key features include high efficiency with |numpy| and |numba|, a low memory
footprint, and extensive documentation. The modules are highly automated,
requiring minimal user input to achieve accurate results. The package also
integrates :doc:`SQL<apyt.io.sql>` or :doc:`local<apyt.io.localdb>` database
management for raw measurement data and corresponding metadata.

In addition to its core processing capabilities, APyT includes a growing
collection of specialized analysis tools. These currently support tasks such as
chemical composition analysis of reconstructed datasets and crystallographic
investigations using methods like
:doc:`spatial distribution maps<apyt.analysis.crystallography.sdm>` (SDMs). This
analysis suite is designed to expand over time, with new tools and techniques
added in response to user needs.

Planned future developments include a PyQt-based graphical user interface (GUI)
that will unify access to all APyT modules within a single application. This GUI
aims to simplify the user experience, making the framework more accessible to
researchers unfamiliar with Python while maintaining the flexibility and
performance valued by advanced users.


.. |numpy| raw:: html

        <a href="https://numpy.org/" target="_blank">NumPy</a>

.. |numba| raw:: html

        <a href="https://numba.pydata.org/" target="_blank">Numba</a>


.. sectionauthor:: Sebastian M. Eich <Sebastian.Eich@imw.uni-stuttgart.de>
"""
__version__ = '0.1.0'
