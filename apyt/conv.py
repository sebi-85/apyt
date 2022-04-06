"""
The APyT file format conversion module
======================================

Howto
-----

The usage of this module is demonstrated in an auxiliary script
(``wrapper_scripts/apyt_conv.py``) which basically serves as a wrapper for this
module. Detailed usage information can be obtained by invoking this script with
the ``"--help"`` option.


List of methods
---------------

This module provides some generic functions for the conversion from and to
various file formats encountered in atom probe tomography.

The following methods are provided:

* :meth:`raw_concat`: Concatenate multiple raw files to a single one.
* :meth:`raw_to_ascii`: Convert a raw measurement file to a human-readable ASCII
  file.


.. |TAPSim| raw:: html

    <a href="https://git.mp.imw.uni-stuttgart.de/cgit.cgi/tapsim.git"
    target="_blank">TAPSim</a>


.. sectionauthor:: Jianshu Zheng <zheng.jianshu@mp.imw.uni-stuttgart.de>
.. moduleauthor:: Jianshu Zheng <zheng.jianshu@mp.imw.uni-stuttgart.de>

"""
#
#
#
#
__version__ = '0.1.0'
__all__ = [
    "raw_concat",
    "raw_to_ascii"
]
#
#
#
#
# import modules
import numpy as np
#
# import some special functions
from struct import pack, unpack
#
#
#
#
################################################################################
#
# private module-level variables
#
################################################################################
_bin_fmt = "<ffffffiI"
"""str : The format of the binary data per measured event."""
#
#
#
#
################################################################################
#
# public functions
#
################################################################################
def raw_concat(raw_files, out_file):
    """Concatenate multiple raw files to a single one.

    This function combines several raw files from a continuous measurement by
    concatenating.

    Parameters
    ----------
    raw_files : list
        The list of the raw file names, each of type `str`.
    out_file: str
        The name of the output file.
    """
    #
    #
    # open output file for binary write
    print("Concatenating files {0:s} ...".format(str(raw_files)))
    with open(out_file, 'wb') as f_out:
        # loop through list of raw files
        for i in raw_files:
            # open input file for binary read
            with open(i, 'rb') as f_in:
                # append input file to output file
                f_out.write(f_in.read())
#
#
#
#
def raw_to_ascii(raw_file, ascii_file):
    """Convert a raw measurement file to a human-readable ASCII file.

    This function enables the conversion from a raw measurement file to a
    human-readable ASCII file. The binary file is read in chunks of 32 bytes,
    (representing one evaporation event), decoded into the respective data
    types, and written to an ASCII text file.

    Parameters
    ----------
    raw_file : str
        The name of the raw file
    ascii_file : str
        The name of ASCII file.
    """
    #
    #
    # get binary data from file
    print("Reading binary file \"{0:s}\" ...".format(raw_file))
    data = np.fromfile(raw_file, dtype = np.dtype('V32')).tolist()
    #
    #
    # open file for output
    print("Writing ASCII file \"{0:s}\" ...".format(ascii_file))
    with open(ascii_file, 'w') as f:
        # write header
        f.write("# U_base (V)\tU_pulse (V)\tU_ref (V)\t" \
                "x_det (mm)\ty_det (mm)\ttof (ns)\tepoch\t\tpulse_num\n")
        #
        # set format string
        fmt = "%9.3f\t%8.3f\t%7.1f\t\t%+10.6f\t%+10.6f\t%8.3f\t%d\t" \
              "%10d\n"
        #
        # convert binary data and write to file
        [f.write(fmt % unpack(_bin_fmt, i)) for i in data]
