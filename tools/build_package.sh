#!/bin/bash -e
#
# Simple script for building and uploading the Python package.
#
#
# usage notes
USAGE=$(cat << EOF
Usage: $(basename $0) [--help] [--official] [--upload]

Simple script for building and uploading the Python package.

Script options:
\t--help\t\tPrint this help message.
\t--official\tWhether to upload the package to the *real* PyPI index.
\t\t\tMust be given together with --upload.
\t--upload\tWhether to upload the package to PyPI.
EOF
)
#
#
# parse optional command line parameters
verbose=""
for i in "$@"; do
    case $i in
        # print help
        --help)
            printf "$USAGE\n"
            exit 0
            ;;
        # upload to official PyPI index
        --official)
            official=DEFINED
            shift
            ;;
        # optional upload
        --upload)
            upload=DEFINED
            shift
            ;;
        # verbose upload
        --verbose)
            verbose="--verbose"
            shift
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
    if [ ! -z ${official+x} ]; then
        # upload to official PyPI index
        python3 -m twine upload $verbose dist/*
    else
        # upload to PyPI testing index
        python3 -m twine upload $verbose --repository testpypi dist/*
    fi
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
