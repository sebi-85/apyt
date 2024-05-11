"""
The APyT package
================

APyT is a Python package which provides modules for the evaluation of atom probe
tomography (APT) data.


Package modules
---------------

.. toctree::
   :maxdepth: 1

   The APyT file format conversion module (apyt.conv)<apyt.conv>
   The APyT local composition module (apyt.locomp)<apyt.locomp>
   The APyT mass spectrum fitting module (apyt.masssfit)<apyt.massfit>
   The APyT mass spectrum module (apyt.massspec)<apyt.massspec>
   The APyT reconstruction module (apyt.reconstruction)<apyt.reconstruction>
   The APyT SDM module (apyt.sdm)<apyt.sdm>


.. sectionauthor:: Sebastian M. Eich <Sebastian.Eich@imw.uni-stuttgart.de>
.. moduleauthor::  Sebastian M. Eich <Sebastian.Eich@imw.uni-stuttgart.de>
"""

__version__ = '0.1.0'
__all__ = ['conv', 'locomp', 'massfit', 'massspec', 'reconstruction', 'sdm']
