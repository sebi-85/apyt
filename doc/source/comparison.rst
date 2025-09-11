Comparison with other APT software
==================================

Commercial software
-------------------

Commercial software such as |cameca|’s **AP Suite** (and its predecessor
**IVAS**) provides a fully integrated, end-to-end workflow for atom probe
tomography (APT) data. These tools are tightly coupled to CAMECA hardware,
featuring graphical user interfaces, project databases, and streamlined
reconstruction pipelines. While powerful and widely used, they are proprietary
and barely extensible, which limits flexibility for customized workflows or
integration with third-party tools.


Open-source APT software
------------------------

In contrast to commercial packages, several **open-source projects** address
specific aspects of APT data handling. They are often modular and
community-driven, focusing on individual tasks rather than providing a fully
integrated workflow. For example:

- |3depict| specializes in 3D visualization of atom probe datasets for
  qualitative and quantitative inspection.
- |apav| offers spectrum quantification, isotopic distributions, and
  visualization.
- |apttools| provides utilities for data processing, visualization, and analysis
  in a modular workflow.
- |atomprobetoolbox| covers multiple possibilities to analyze atom probe
  datasets and is provided as a Matlab toolbox.
- |paraprobetoolbox| provides advanced statistical and geometric analysis of
  reconstructed point clouds.
- |pyccapt| includes calibration and control routines for experimental setups.
- |pynxtoolsapm| focuses on standardized file I/O and FAIR data principles.


While these tools are valuable, their specialization often means they lack a
unified, end-to-end workflow.


APyT: Bridging the gap
----------------------

**APyT** occupies a complementary niche between commercial and open-source
software, combining high performance with flexibility and accessibility. Its key
strengths include:


High performance and automation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- **High efficiency and speed** --- optimized algorithms using |numpy| and
  |numba| enable rapid data processing even for large datasets.
- **Accurate and reliable** --- ensures high-quality reconstructions and precise
  analysis results.
- **Highly automated** --- built-in automation reduces manual intervention and
  supports batch processing.


Complete and modular workflow
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- **Complete workflow** --- provides a full pipeline from raw data to
  three-dimensional reconstruction.
- **Modular** --- Python subpackages for alignment, fitting, reconstruction, and
  analysis can be used independently or combined into custom workflows.


Accessible and transparent
^^^^^^^^^^^^^^^^^^^^^^^^^^

- **Accessible** --- lightweight :doc:`command line interface <apyt_cli>` for
  ready-to-use scenarios or exploratory testing, with a |matplotlib|-based
  graphical interface.
- **Transparent** --- implemented entirely in Python, encouraging
  reproducibility and integration with external scientific Python tools.
- **Educational** --- includes a small exemplary dataset to help new users
  quickly understand the APyT workflow.

In summary, while AP Suite remains the industrial standard and other open-source
projects provide specialized tools, **APyT bridges the gap** by offering an
open, extensible, and Python-native framework that supports both research
innovation and everyday data processing tasks.


.. |3depict| raw:: html

    <b><a href="https://threedepict.sourceforge.net/" target="_blank">3Depict</a>
    </b>

.. |apav| raw:: html

    <b><a href="https://apav.readthedocs.io/en/latest/index.html"
    target="_blank">APAV (Atom Probe Analysis and Visualization)</a></b>

.. |apttools| raw:: html

    <b><a href="https://apttools.sourceforge.io/" target="_blank">APTTools</a>
    </b>

.. |atomprobetoolbox| raw:: html

    <b><a href="https://github.com/peterfelfer/Atom-Probe-Toolbox"
    target="_blank">Atom Probe Toolbox</a></b>

.. |cameca| raw:: html

    <a href="https://www.cameca.com/" target="_blank">Cameca</a>

.. |matplotlib| raw:: html

    <a href="https://matplotlib.org/" target="_blank">Matplotlib</a>

.. |numba| raw:: html

    <a href="https://numba.pydata.org/" target="_blank">Numba</a>

.. |numpy| raw:: html

    <a href="https://numpy.org/" target="_blank">NumPy</a>

.. |paraprobetoolbox| raw:: html

    <b><a href="https://paraprobe-toolbox.readthedocs.io/en/main/"
    target="_blank">paraprobe-toolbox</a></b>

.. |pyccapt| raw:: html

    <b><a href="https://pyccapt.readthedocs.io/en/latest/" target="_blank">
    PyCCAPT</a></b>

.. |pynxtoolsapm| raw:: html

    <b><a href="https://fairmat-nfdi.github.io/pynxtools-apm/" target="_blank">
    pynxtools-apm</a></b>
