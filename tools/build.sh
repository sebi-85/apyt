#!/bin/bash -e
#
# Simple script for building and uploading the Python package.
#
#
#
#
# create and activate virtual environment
if [ ! -d "venv_build" ]; then
    echo -n "Creating virtual environment for package building…"
    python3 -m venv --prompt build venv_build
    source venv_build/bin/activate
    pip3 install -r requirements/build_requirements.txt > /dev/null
    echo " done."
else
    source venv_build/bin/activate
fi
#
#
# build package
python3 -m build
#
#
# upload package
python3 -m twine upload --repository testpypi dist/*
#
#
# deactivate virtual environment
deactivate
