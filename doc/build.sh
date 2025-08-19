#/bin/bash -e
#
# Simple script for building the package documentation.
#
#
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
# change into doc folder
cd ${SCRIPT_DIR}/
#
#
# create and activate virtual environment
if [ ! -d "../venv_doc" ]; then
    echo -n "Creating virtual environment…"
    python3 -m venv --prompt APyT ../venv_doc
    source ../venv_doc/bin/activate
    pip3 install \
      -r ../requirements/requirements.txt \
      -r ../requirements/requirements_doc.txt \
      > /dev/null > /dev/null
    echo " done."
else
    source ../venv_doc/bin/activate
fi
#
#
# build documentation
rm -rf build/
sphinx-apidoc -o source -e -T -E ../apyt
sphinx-build source/ build/
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
