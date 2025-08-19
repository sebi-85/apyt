"""
The APyT reconstruction subpackage
==================================

The `apyt.reconstruction` subpackage provides tools for reconstructing
three-dimensional atom probe tomography (APT) datasets from raw measurement
data.

It includes routines for both basic and advanced reconstructions, supporting
semi-automated workflows and optimization techniques to extract spatial tip
coordinates and chemical identities of atoms.

These tools aim to enhance the accuracy and reproducibility of APT
reconstructions, addressing different tip geometries and reconstruction models.


Available modules
-----------------

The following modules are available in this subpackage:

.. toctree::
   :maxdepth: 1

   The APyT basic reconstruction module \
   (apyt.reconstruction.basic)<apyt.reconstruction.basic>
   The APyT advanced reconstruction module \
   (apyt.reconstruction.non_spherical)<apyt.reconstruction.non_spherical>


.. sectionauthor:: Sebastian M. Eich <Sebastian.Eich@imw.uni-stuttgart.de>
"""
__version__ = '0.1.0'
__all__ = ['basic', 'non_spherical']
