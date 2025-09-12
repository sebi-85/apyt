Installation instructions
=========================

The simplest way to install the |apyt| is using the official PyPI repository:

.. code-block:: bash

    pip3 install apyt

This command will also install all required dependencies.

.. note::

    It is recommended to create a Python virtual environment before installing
    APyT. This ensures that dependencies are isolated from your system Python
    installation and prevents potential version conflicts with other packages.

    You can create and activate a virtual environment as follows:

    **On Linux / macOS:**

    .. code-block:: bash

        python3 -m venv --prompt APyT apyt-env
        source apyt-env/bin/activate

    **On Windows (Command Prompt):**

    .. code-block:: bat

        python -m venv --prompt APyT apyt-env
        apyt-env\Scripts\activate

.. attention::

    Do **not** install Python from the Microsoft Store. There are known
    |ms_issues| related to sandboxing, particularly the redirection of
    ``AppData``, which may interfere with the :ref:`configuration file location
    <apyt.io.config:Configuration file location>` of the APyT package.
    Instead, install Python on Microsoft Windows using the official |python|
    installer.


.. |apyt| raw:: html

        <a href="https://pypi.org/project/apyt/" target="_blank">
        APyT package</a>

.. |ms_issues| raw:: html

        <a href="https://docs.python.org/3/using/windows.html#known-issues"
        target="_blank">issues</a>

.. |python| raw:: html

        <a href="https://www.python.org/" target="_blank">Python</a>


.. sectionauthor:: Sebastian M. Eich <Sebastian.Eich@imw.uni-stuttgart.de>
