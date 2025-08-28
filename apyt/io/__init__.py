"""
The APyT file input/output subpackage
=====================================

The ``apyt.io`` subpackage provides tools for reading, writing, converting, and
managing raw measurement data and related files used in atom probe tomography
(APT).

Beyond file format conversion and metadata handling, this subpackage also
supports structured data management via both SQL and local YAML databases,
enabling efficient storage, indexing, and querying of experimental data.

Together, these tools facilitate interoperability between file formats and
provide streamlined data workflows within the APyT framework.


Available modules
-----------------

The following modules are available in this subpackage:

.. toctree::
   :maxdepth: 1

   The APyT configuration module (apyt.io.config)<apyt.io.config>
   The APyT file format conversion module (apyt.io.conv)<apyt.io.conv>
   The APyT local database module (apyt.io.localdb)<apyt.io.localdb>
   The APyT SQL module (apyt.io.sql)<apyt.io.sql>


.. sectionauthor:: Sebastian M. Eich <Sebastian.Eich@imw.uni-stuttgart.de>
"""
__version__ = "0.1.0"
__all__ = ["config", "conv", "sql"]
