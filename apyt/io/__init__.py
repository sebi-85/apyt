"""
The APyT file input/output subpackage
=====================================

The `apyt.io` subpackage provides functions for reading, writing, converting,
and managing raw measurement data and related files used in atom probe
tomography (APT).

In addition to file format conversion and metadata handling, this subpackage
also incorporates basic SQL database management for structured data storage,
indexing, and efficient querying of large datasets.

These tools help facilitate interoperability between file formats and enable
streamlined data workflows within the APyT framework.


Available modules
-----------------

The following modules are available in this subpackage:

.. toctree::
   :maxdepth: 1

   The APyT configuration module (apyt.io.config)<apyt.io.config>
   The APyT file format conversion module (apyt.io.conv)<apyt.io.conv>


.. sectionauthor:: Sebastian M. Eich <Sebastian.Eich@imw.uni-stuttgart.de>
"""
__version__ = "0.1.0"
__all__ = [
   "config",
   "conv"
]
