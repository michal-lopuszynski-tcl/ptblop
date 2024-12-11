MODULE_NAME=ptblopgen

ZIP_DIR=build_zip

PY_DIRS=src/ptblopgen tests setup.py

PY_MYPY_FLAKE8=src/ptblopgen tests setup.py

FILES_TO_CLEAN=src/ptblopgen.egg-info $(ZIP_DIR)

include Makefile.inc

build_zip:
	rm -rf $(ZIP_DIR) && \
	mkdir $(ZIP_DIR) && \
	cd $(ZIP_DIR) && \
	cp -rp ../src/ptblopgen . && \
	cp -rp ../../ptblop/src/ptblop . && \
	printf "import sys\nimport ptblopgen.cli\n\nsys.exit(ptblopgen.cli.main())\n" > __main__.py && \
	zip -r ptblopgen.zip ptblop ptblopgen __main__.py && \
	rm -rf ptblop ptblopgen __main__.py

.PHONY: build_zip
