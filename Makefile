SHELL = /bin/bash
.PHONY: help clean install test version dist

help:
	@echo "make clean"
	@echo " clean all python build/compilation files and directories"
	@echo "make version"
	@echo " update _version.py with current version tag"
	@echo "make dist"
	@echo " build the package ready for distribution and update the version tag"

clean:
	find . -name '*.pyc' -exec rm --force {} +
	find . -name '*.pyo' -exec rm --force {} +
	find . -name '*~' -exec rm --force {} +
	rm --force .coverage
	rm --force --recursive .pytest_cache
	rm --force --recursive build/
	rm --force --recursive dist/
	rm --force --recursive *.egg-info

version:
	echo -e "__version__ = \"$(shell git describe --always --tags --abbrev=0)\"\n__commit__ = \"$(shell git rev-parse --short HEAD)\"" > src/abcraster/_version.py

dist: version
	pip3 install build twine
	python3 -m build
