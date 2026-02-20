# APyT: A modular open-source Python framework for atom probe tomography data evaluation

The APyT package is an advanced, open-source Python framework for evaluating
atom probe tomography (APT) data. It offers a suite of modules that automate key
steps in APT processing, from mass spectrum calibration to three-dimensional
sample reconstruction. Its modular architecture ensures seamless integration
with external tools through standardized input/output interfaces, supporting
both Linux and Windows environments. Users can import and use each module
independently, or integrate them in custom workflows or GUI applications.

For ready-to-use scenarios or testing, a command line interface (CLI) is
provided. The CLI offers lightweight wrappers around the core modules, enabling
users to process raw measurement data without writing additional Python code. It
is particularly useful for exploratory analysis, quick prototyping, and
stand-alone usage.

Key features include high efficiency with [NumPy][numpy] and [Numba][numba], a
low memory footprint, and extensive documentation. The modules are highly
automated, requiring minimal user input to achieve accurate results. The package
also integrates SQL or local database management for raw measurement data and
corresponding metadata.


# User Guide

For detailed installation instructions, configuration options, and usage
examples, please refer to the official APyT documentation available through the
homepage (see project links in the navigation section on this page).


# License Notice – Use in Modifications and Derivative Works

This software is licensed under the
[GNU Affero General Public License v3.0 (AGPLv3)][agpl].

Under the AGPLv3, any software that incorporates, imports, links to, or depends
on this project to perform essential functionality may be considered a
derivative work. In such cases:

  - The complete source code of the resulting software must be made available
    under the AGPLv3 (or a compatible license).
  - This requirement applies not only to traditional distribution but also when
    the software is made available as a network service (e.g., SaaS or web
    applications).

## Personal/Academic Use

You are free to use, study, and modify this software for personal purposes
without any obligation to publish your changes, provided the software is not
distributed or made accessible to others (including over a network).

This includes, but is not limited to:

  - Personal research and experimentation
  - Academic or offline analysis
  - Internal use within a lab or private project
  - Non-public prototypes or development tools

**Important:** If you share this software or make it accessible to others—such
as through collaborative work, academic publications, network-based services, or
cloud deployments—you must comply with the AGPLv3 by releasing the complete
corresponding source code.

If you're unsure whether your use qualifies as personal or requires source
disclosure, feel free to contact the project maintainer for clarification.




[numpy]: https://numpy.org/
[numba]: https://numba.pydata.org/
[agpl]:  https://www.gnu.org/licenses/agpl-3.0.html
