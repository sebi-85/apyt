#!/bin/bash -e
#
# Simple script for building and uploading the Python package.
#
#
# usage notes
USAGE=$(cat << EOF
Usage: $(basename $0) [--help] [--upload]

Simple script for building and uploading the Python package.

Script options:
\t--help\t\tPrint this help message.
\t--upload\tWhether to upload the package to PyPi.
EOF
)
#
#
# parse optional command line parameters
for i in "$@"; do
    case $i in
        # set density
        --upload)
            upload=DEFINED
            shift
            ;;
        # print help
        --help)
            printf "$USAGE\n"
            exit 0
            ;;
        # unknown option
        *)
            ;;
    esac
done
#
#
# set script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
#
#
# store current working directory
PWD=$(pwd)
#
#
#
#
# change into root folder
cd ${SCRIPT_DIR}/../
#
#
# create and activate virtual environment
if [ ! -d "venv_build" ]; then
    echo -n "Creating virtual environment for package building…"
    python3 -m venv --prompt build venv_build
    source venv_build/bin/activate
    pip3 install -r requirements/requirements_build.txt > /dev/null
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
# upload package if requested
if [ ! -z ${upload+x} ]; then
    python3 -m twine upload --repository testpypi dist/*
fi
#
#
# deactivate virtual environment
deactivate
#
#
#
#
# change back into initial directory
cd $PWD
