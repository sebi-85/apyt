"""
The APyT package
================

APyT is a Python package which provides modules for the evaluation of atom probe
tomography (APT) data.


Sub-packages
------------

.. toctree::
   :maxdepth: 1

   The APyT analysis sub-package (apyt.analysis)<apyt.analysis>
   The APyT file input/output sub-package (apyt.io)<apyt.io>
   The APyT reconstruction sub-package \
   (apyt.reconstruction)<apyt.reconstruction>
   The APyT mass spectrum sub-package (apyt.spectrum)<apyt.spectrum>


Package modules
---------------

.. toctree::
   :maxdepth: 1

   The APyT local composition module (apyt.locomp)<apyt.locomp>
   The APyT SDM module (apyt.sdm)<apyt.sdm>


.. sectionauthor:: Sebastian M. Eich <Sebastian.Eich@imw.uni-stuttgart.de>
.. moduleauthor::  Sebastian M. Eich <Sebastian.Eich@imw.uni-stuttgart.de>
"""

__version__ = '0.1.0'
__all__ = ['locomp', 'sdm']
