#/bin/bash
#
# activate virtual environment
source ../venv/bin/activate
pip3 install -r ../requirements.txt
#
rm -rf build/
sphinx-apidoc -o source -e -T -E ../apyt
sphinx-build source/ build/
#
# deactivate virtual environment
deactivate
