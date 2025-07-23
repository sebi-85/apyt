"""
The APyT mass spectrum sub-package
==================================

This sub-package provides tools for aligning high-quality mass spectra from raw
atom probe measurement data. It also supports analytical fitting of the
processed spectra to assist in chemical identification.


Package modules
---------------

.. toctree::
   :maxdepth: 1

   The APyT mass spectrum alignment module \
   (apyt.spectrum.align)<apyt.spectrum.align>
   The APyT mass spectrum fitting module (apyt.spectrum.fit)<apyt.spectrum.fit>


.. sectionauthor:: Sebastian M. Eich <Sebastian.Eich@imw.uni-stuttgart.de>
.. moduleauthor::  Sebastian M. Eich <Sebastian.Eich@imw.uni-stuttgart.de>
"""

__version__ = '0.1.0'
__all__ = ['align', 'fit']
