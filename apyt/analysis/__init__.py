"""
The APyT analysis subpackage
============================

The `apyt.analysis` subpackage provides a suite of tools for the quantitative
and qualitative analysis of atom probe tomography (APT) data. It enables users
to process and interpret:

- **Raw measurement data** from atom probe experiments;
- **Calibrated mass spectra**, for element and isotope identification;
- **Reconstructed 3D tip coordinates**, to analyze spatial distributions and
  structural information.

This subpackage is designed to support common analysis workflows in APT
research, including chemical composition profiling, crystallographic analysis,
and spectrum interpretation.


Available subpackages
---------------------

These specialized subpackages provide domain-specific tools for deeper analysis:

.. toctree::
   :maxdepth: 1

   The APyT chemistry analysis subpackage \
      (apyt.analysis.chemistry)<apyt.analysis.chemistry>
   The APyT crystallography analysis subpackage \
      (apyt.analysis.crystallography)<apyt.analysis.crystallography>


Available modules
-----------------

Standalone modules that provide core functionality for spectrum and data
analysis:

.. toctree::
   :maxdepth: 1

   The APyT spectrum analysis module \
      (apyt.analysis.spectrum)<apyt.analysis.spectrum>


.. sectionauthor::  Sebastian M. Eich <Sebastian.Eich@imw.uni-stuttgart.de>
"""
__version__ = '0.1.0'
__all__ = ['spectrum']
