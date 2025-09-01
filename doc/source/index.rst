.. include:: apyt.rst


User Guide
==========

.. toctree::
   :maxdepth: 1
   :caption: User Guide
   :hidden:

   installation
   apyt_cli


Installation
------------

Follow the :doc:`installation instructions<installation>` to set up the APyT
package on your system and ensure all required dependencies are installed.


Command line interface
----------------------

The recommended way to interact with the APyT package is through the provided
:doc:`command line scripts<apyt_cli>`. Each script is documented in detail,
including available options, expected input/output formats, and usage examples.


Developer Guide
===============

.. toctree::
   :maxdepth: 1
   :caption: Package Documentation
   :hidden:

   The APyT analysis subpackage (apyt.analysis)<apyt.analysis>
   The APyT GUI subpackage (apyt.gui)<apyt.gui>
   The APyT file input/output subpackage (apyt.io)<apyt.io>
   The APyT reconstruction subpackage (apyt.reconstruction)<apyt.reconstruction>
   The APyT mass spectrum subpackage (apyt.spectrum)<apyt.spectrum>

While the :doc:`command line interface<apyt_cli>` provides convenient,
ready-to-use scripts for exploring the capabilities of the APyT package, its
modular design allows developers to integrate individual subpackages and modules
into custom workflows. Each subpackage is fully documented, enabling
programmatic access to core functionality such as file I/O, mass spectrum
processing, reconstruction algorithms, and data analysis.

The following list provides an overview of the main APyT subpackages:

- :doc:`The APyT analysis subpackage (apyt.analysis)<apyt.analysis>`
- :doc:`The APyT GUI subpackage (apyt.gui)<apyt.gui>`
- :doc:`The APyT file input/output subpackage (apyt.io)<apyt.io>`
- :doc:`The APyT reconstruction subpackage (apyt.reconstruction)
  <apyt.reconstruction>`
- :doc:`The APyT mass spectrum subpackage (apyt.spectrum)<apyt.spectrum>`


Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
