"""
The APyT GUI subpackage
=======================

This subpackage provides graphical user interface (GUI) components for APyT. It
offers lightweight, cross-platform dialogs and forms built on Tkinter, designed
to complement the command-line interface by enabling interactive user input when
needed.

The GUI is intentionally minimalistic, focusing on essential workflows (such as
authentication) while remaining easy to extend for additional forms or dialogs
in the future.


Available modules
-----------------

The following modules are available in this subpackage:

.. toctree::
   :maxdepth: 1

   The APyT GUI forms module (apyt.gui.forms)<apyt.gui.forms>


.. sectionauthor:: Sebastian M. Eich <Sebastian.Eich@imw.uni-stuttgart.de>
"""
__version__ = "0.1.0"
__all__ = ["forms"]
