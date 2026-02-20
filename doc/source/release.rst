Release history
===============

v0.1.0.dev3 (2026-02-20)
------------------------

- Add optional width scaling for individual peaks


v0.1.0.dev2 (2025-10-07)
------------------------

- Fixed Unicode encoding issue when writing the local database on Windows


v0.1.0.dev1 (2025-09-08)
------------------------

Initial public release of **APyT**.

Features
^^^^^^^^

- Modular Python framework for evaluating atom probe tomography (APT) data.
- Full support for SQL and local database backends.
- Core modules for:

  * Mass spectrum alignment (:mod:`apyt.spectrum.align`)
  * Mass spectrum fitting (:mod:`apyt.spectrum.fit`)
  * Reconstruction of three-dimensional tip geometry
    (:mod:`apyt.reconstruction`)
  * Input/output handling (:mod:`apyt.io`)
  * Analysis utilities (:mod:`apyt.analysis`)
  * Lightweight GUI components (:mod:`apyt.gui`)

- Ready-to-use :doc:`command line interface<apyt_cli>` wrappers for alignment,
  fitting, and reconstruction workflows.
- Example dataset (``apyt_W_calibration_tap_01_trimmed.raw``) included for quick
  testing and demonstration.
- Documentation with user and developer guides.


.. sectionauthor:: Sebastian M. Eich <Sebastian.Eich@imw.uni-stuttgart.de>
