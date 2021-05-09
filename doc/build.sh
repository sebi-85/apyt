#/bin/bash
#
rm -rf build/
sphinx-apidoc -o source -e -T -E ../apyt
sphinx-build source/ build/
