"""
The APyT mass spectrum subpackage
=================================

The `apyt.spectrum` subpackage provides tools for processing and analyzing mass
spectra derived from raw atom probe tomography (APT) data.

It includes modules for:

- **Mass spectrum alignment**, ensuring accurate calibration and peak
  positioning.
- **Analytical peak fitting**, aiding in the identification and quantification
  of chemical species.

These tools are essential for achieving high-resolution mass spectral analysis
and reliable chemical interpretation of APT datasets.


Available modules
-----------------

The following modules are available within this subpackage:

.. toctree::
   :maxdepth: 1

   The APyT mass spectrum alignment module \
   (apyt.spectrum.align)<apyt.spectrum.align>
   The APyT mass spectrum fitting module (apyt.spectrum.fit)<apyt.spectrum.fit>


.. sectionauthor:: Sebastian M. Eich <Sebastian.Eich@imw.uni-stuttgart.de>
"""
__version__ = '0.1.0'
__all__ = ['align', 'fit']
