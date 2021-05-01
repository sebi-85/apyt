#/bin/bash
#
rm -rf build/*
sphinx-apidoc -f -T -E -o source ..
sphinx-build source/ build/
