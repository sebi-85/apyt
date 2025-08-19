"""
The APyT crystallography analysis subpackage
============================================

The `apyt.analysis.crystallography` subpackage provides tools for the
crystallographic analysis of reconstructed atom probe tomography (APT) data.

It supports methods for identifying crystallographic features such as
atomic planes, orientation relationships, and lattice periodicities, based on
spatial distribution maps of atoms in the reconstructed 3D dataset.


Available modules
-----------------

The following modules are available in this subpackage:

.. toctree::
   :maxdepth: 1

   The APyT SDM module \
   (apyt.analysis.crystallography.sdm)<apyt.analysis.crystallography.sdm>


.. sectionauthor:: Sebastian M. Eich <Sebastian.Eich@imw.uni-stuttgart.de>
"""
__version__ = '0.1.0'
__all__ = ['sdm']
