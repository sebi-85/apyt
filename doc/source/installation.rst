Installation instructions
=========================

The simplest way to install the APyT package is using the official PyPi
(testing) repository.

.. code-block:: bash

    pip3 install --extra-index-url https://test.pypi.org/simple/ apyt

This command will also install all required dependencies.


.. note::

    It is recommended to create a Python virtual environment before installing
    APyT. This ensures that dependencies are isolated from your system Python
    installation and prevents potential version conflicts with other packages.
    You can create a virtual environment (on Linux) using:

    .. code-block:: bash

        python3 -m venv --prompt APyT apyt-env
        source apyt-env/bin/activate
