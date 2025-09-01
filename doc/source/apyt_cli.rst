The APyT command line interface
===============================

The **APyT** package is designed in a modular way to provide maximum
flexibility. Its core modules can be integrated directly into custom workflows,
while still offering ready-to-use command line tools for standard analysis
tasks.

APyT is built with **SQL database integration** in mind:

- **Measurement data** are stored centrally in a database.
- **Metadata** (e.g. experimental parameters, calibration results) are linked to
  the measurements and stored in the SQL database.

For users who do not want to set up or manage an
:doc:`SQL database<apyt.io.sql>`, APyT also provides a
:doc:`local database module<apyt.io.localdb>`. This enables you to store
measurement files locally while keeping all associated metadata in a lightweight
YAML-based database.

Configuration of both SQL and local databases is managed through the global
configuration system. See the
:doc:`global configuration options<apyt.io.config>` for details.


Why use the command line interface?
-----------------------------------

While APyT modules can be imported and used directly in Python, the command line
interface (CLI) provides **ready-made wrappers** that:

- Streamline the most common analysis steps.
- Require minimal to no setup.
- Make it easy to test APyT without writing additional Python code.

Typical workflows covered by the CLI range from processing raw measurement data
to performing a full **3D reconstruction** of atom probe data.


Available command line scripts
------------------------------

The following command line tools are included with **APyT**:

.. toctree::
   :maxdepth: 1

   The APyT mass spectrum alignment script<apyt_cli.spectrum_align>

Each script is documented individually, including its available options,
expected input/output formats, and usage examples. For convenience, a
lightweight graphical user interface is provided through |matplotlib| plots and
widgets.

.. note::

   The Matplotlib interface is designed primarily for ease of use and producing
   high-quality plots rather than raw performance. As a result, interactions may
   feel somewhat laggy depending on your hardware.


.. |matplotlib| raw:: html

    <a href="https://matplotlib.org/" target="_blank">Matplotlib</a>


Example workflow
----------------

A typical CLI-based workflow might look like this:

1. **Prepare your measurement files** and ensure they are registered in either
   the SQL or local database.

   .. note::

      Measurement files are expected in the
      :ref:`raw file format <apyt.io.conv:Raw file format>`. If your data is in
      *ePOS* format, you can easily :func:`convert<apyt.io.conv.epos_to_raw>` it
      to raw format using a one-liner:

      .. code-block:: bash

         python3 -c "from apyt.io.conv import epos_to_raw; epos_to_raw('<epos_file>')"

2. **Align the mass spectrum** using the ``apyt_spectrum_align`` script.
3. **Fit the peaks** with the ``apyt_spectrum_fit`` script to identify species.
4. **Perform a 3D reconstruction** of the dataset with the
   ``apyt_reconstruction`` script.

This provides a fully reproducible analysis path from raw input files to
processed, interpretable scientific data.


.. sectionauthor:: Sebastian M. Eich <Sebastian.Eich@imw.uni-stuttgart.de>
