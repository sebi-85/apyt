"""
The APyT file format conversion module
======================================

APT data can exist in multiple file formats—often representing the same
measurement dataset but stored differently (e.g., as raw binary or decoded
ASCII). This module provides easy-to-use
:ref:`functions<apyt.io.conv:List of functions>` to convert between various file
formats commonly encountered in atom probe tomography (APT) workflows.

It enables standardized preprocessing and ensures compatibility across software
tools within the APyT ecosystem.


Raw file format
---------------

The APT group at the University of Stuttgart uses a binary file format to record
APT measurements. Each file entry corresponds to a single evaporation event and
follows **Little Endian** byte ordering.

The binary format includes the following fields:

============  =========  =================================
Field         Data type  Description
============  =========  =================================
U_base        float32    base voltage (V)
U_pulse       float32    pulse voltage (V)
U_reflectron  float32    reflectron voltage (V)
x_det         float32    `x` detector position (mm)
y_det         float32    `y` detector position (mm)
tof           float32    time of flight (ns)
epoch         int32      epoch of evaporation event
pulse_num     uint32     pulse number of evaporation event
============  =========  =================================


List of functions
-----------------

The following functions are available for format conversion:

* :func:`raw_concat`: Concatenate multiple raw files to a single one.
* :func:`raw_to_ascii`: Convert a raw measurement file to a human-readable ASCII
  file.
* :func:`tapsim_to_raw`: Convert |TAPSim| ASCII file to raw file.


.. |TAPSim| raw:: html

    <a href="https://git.mp.imw.uni-stuttgart.de/cgit.cgi/tapsim.git"
    target="_blank">TAPSim</a>


.. sectionauthor:: Sebastian M. Eich <Sebastian.Eich@imw.uni-stuttgart.de>
.. codeauthor::    Jianshu Zheng <zheng.jianshu@mp.imw.uni-stuttgart.de>
.. codeauthor::    Sebastian M. Eich <Sebastian.Eich@imw.uni-stuttgart.de>
"""
#
#
#
#
__version__ = '0.1.0'
__all__ = [
    "raw_concat",
    "raw_to_ascii",
    "tapsim_to_raw"
]
#
#
#
#
# import modules
import numpy as np
import warnings
#
# import some special functions
from datetime import datetime
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
        The name of the raw file.
    ascii_file : str
        The name of the ASCII file.
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
        f.write("# U_base (V)\tU_pulse (V)\tU_reflectron (V)\t"
                "x_det (mm)\ty_det (mm)\ttof (ns)\tepoch\t\tpulse_num\n")
        #
        # set format string
        fmt = "%9.3f\t%8.3f\t%7.1f\t\t\t%+11.6f\t%+11.6f\t%8.3f\t%d\t" \
              "%10d\n"
        #
        # convert binary data and write to file
        [f.write(fmt % unpack(_bin_fmt, i)) for i in data]
#
#
#
#
def tapsim_to_raw(tapsim_file, raw_file, id_range_list):
    """Convert TAPSim ASCII file to raw file.

    This function enables the conversion from a |TAPSim| ASCII file to a raw
    file for further processing (e.g. reconstruction). A certain subset of
    columns is imported from the TAPSim file, manipulated accordingly to match
    the :ref:`raw file format<apyt.io.conv:Raw file format>`, and eventually
    written to a binary file. A constant base voltage is used for all events and
    the time of flight is arranged such that it is constant for one distinct
    species. The epoch is set to a constant time plus 1 event/s, the pulse
    number corresponds to the evaporation event.

    The conversion is illustrated in the following table:

    ============  =========  ====================  =============================
    Raw file      Data type  TAPSim file           Comment
    ============  =========  ====================  =============================
    U_base        float32    5000 V                constant
    U_pulse       float32    0                     zero
    U_reflectron  float32    0                     zero
    x_det         float32    col. 7                conversion from meter to
                                                   millimeter
    y_det         float32    col. 8                conversion from meter to
                                                   millimeter
    tof           float32    constant per species  constant for one species,
                                                   separation 50 ns
    epoch         int32      946681200 + event     (2000-01-01 00:00:00) +
                                                   1 event/s
    pulse_num     uint32     0, 1, 2, ...          corresponds to evaporation
                                                   event
    ============  =========  ====================  =============================

    Parameters
    ----------
    tapsim_file: str
        The name of the TAPSim file.
    raw_file: str
        The name of the raw file.
    id_range_list: list
        The list of id ranges used for mapping the atomic species, each of type
        `tuple` of length 2, specifying the respective minimum and maximum id.
    """
    #
    #
    # load evaporation index, atomic id, and detector xy-position from TAPSim
    # file
    print("Reading TAPSim file \"{0:s}\" ...".format(tapsim_file))
    data = np.loadtxt(tapsim_file, skiprows = 46, usecols = (0, 1, 7, 8))
    #
    #
    # filter entries with nan values for detector position
    length_init = len(data)
    data = data[~(np.isnan(data[:, 2]) | np.isnan(data[:, 3]))]
    if length_init != len(data):
        warnings.warn("{0:d} events with invalid detector positions (nan) have "
                      "been removed.".format(length_init - len(data)))
    #
    #
    # initialize empty array for mapped atomic ids
    id = np.full(len(data), -1, dtype = int)
    #
    # loop through id ranges
    for id_range in id_range_list:
        # set index of current id range
        i = id_range_list.index(id_range)
        #
        # map atomic id if in current range
        id = np.where(
            (id_range[0] <= data[:, 1]) & (data[:, 1] <= id_range[-1]), i, id)
    #
    # check whether all ids have been mapped
    if np.count_nonzero(id == -1) > 0:
        raise Exception("Unspecified id detected. Please check your id ranges "
                        "to cover all occurring ids ({0:d}, {1:d}).".format(
                            int(data[:, 1].min()), int(data[:, 1].max())))
    #
    #
    # set arbitrary timestamp required in raw file
    epoch = datetime(2000, 1, 1, 0, 0, 0).timestamp()
    #
    #
    # set voltages
    voltage = (5000.0, 0.0, 0.0)
    #
    #
    # set data types for structured array
    dt = np.dtype([
        ('x_det', np.float32), ('y_det', np.float32), ('tof', np.float32),
        ('epoch', np.int32),   ('pulse_num', np.uint32)])
    #
    # create and fill structured array
    data_str = np.empty((len(data)), dtype = dt)
    data_str['x_det']     = data[:, 2] * 1000  # m to mm
    data_str['y_det']     = data[:, 3] * 1000  # m to mm
    data_str['tof']       = id * 50.0 + 50.0   # tof grouped by atomic id
    data_str['epoch']     = data[:, 0] + epoch # event id plus time offset
    data_str['pulse_num'] = data[:, 0]         # event id
    #
    # convert structured array to list for faster iterator
    data_l = data_str.tolist()
    #
    #
    # open output file for writing
    print("Writing binary file \"{0:s}\" ...".format(raw_file))
    with open(raw_file, 'wb') as f:
        # loop through events
        [f.write(pack(_bin_fmt, *voltage, *i)) for i in data_l]
