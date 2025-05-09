
VER_FILE=src/$(MODULE_NAME)/_version.py

ZIP_DIR=dist_zip

F=set_it_to_reformated_file_from_cli

all: build

build:
	python3 -mbuild

build_zip:
	ZIP_SUFFIX="-mod"; \
	git diff --exit-code && ZIP_SUFFIX=""; \
	ZIP_NAME="ptblopgen-$$(git rev-list --count HEAD)-$$(git rev-parse --short HEAD)$${ZIP_SUFFIX}.zip"; \
	rm -rf $(ZIP_DIR) && \
	mkdir $(ZIP_DIR) && \
	cd $(ZIP_DIR) && \
	cp -rp ../src/ptblopgen . && \
	cp -rp ../../ptblop/src/ptblop . && \
	printf "import sys\nimport ptblopgen.cli\n\nsys.exit(ptblopgen.cli.main())\n" > __main__.py && \
	zip -r $${ZIP_NAME} ptblop ptblopgen __main__.py && \
	rm -rf ptblop ptblopgen __main__.py
	ZIP_SUFFIX="-mod"; \
	git diff --exit-code && ZIP_SUFFIX=""; \
	ZIP_NAME="ptblopgen-plain-$$(git rev-list --count HEAD)-$$(git rev-parse --short HEAD)$${ZIP_SUFFIX}.zip"; \
	cd $(ZIP_DIR) && \
	cp -rp ../src/ptblopgen . && \
	printf "import sys\nimport ptblopgen.cli\n\nsys.exit(ptblopgen.cli.main())\n" > __main__.py && \
	zip -r $${ZIP_NAME} ptblop ptblopgen __main__.py && \
	rm -rf ptblop ptblopgen __main__.py


check: -check_black -check_isort -check_flake8 -check_mypy

check_fast: -check_black -check_isort -check_flake8

check_black:
	@black --color --check $(PY_DIRS) > make.black.log 2>&1 && \
		echo "PASSED black" || \
		(echo "FAILED black"; cat make.black.log; exit 1)

check_isort:
	@isort --check --profile=black $(ISORT_FLAGS) $(PY_DIRS) && echo "PASSED isort"


check_flake8:
	@(flake8  --color=always --ignore=E203,W503 --max-line-length 88 $(PY_MYPY_FLAKE8) > make.flake8.log 2>&1) && \
		echo "PASSED flake8" || \
		(echo "FAILED flake8"; cat make.flake8.log; exit 1)

check_mypy:
	@MYPY_FORCE_COLOR=1 mypy --color-output --disallow-incomplete-defs --disallow-untyped-defs $(PY_MYPY_FLAKE8) > make.mypy.log 2>&1 && \
		echo "PASSED mypy" || \
		(echo "FAILED mypy"; cat make.mypy.log; exit 1)

test_py:
	@pytest -vv tests

test_full:
	@rm -rf tests_full/tmp.*
	-@cd tests_full && ./test_llm_sample.sh
	-@cd tests_full && ./test_llm_paretofind.sh
	-@cd tests_full && ./test_llm_paretoeval.sh
	-@cd tests_full && ./test_llm_sample_blocks.sh
	-@cd tests_full && ./test_llm_paretofind_blocks.sh
	-@cd tests_full && ./test_llm_paretoeval_blocks.sh

test: -test_py -test_full

test_cpu:
	CUDA_VISIBLE_DEVICES="" pytest -vv tests

clean:
	rm -rf make.black.log make.flake8.log make.mypy.log tests_full/tmp.* \
    $(ZIP_DIR) $(FILES_TO_CLEAN)

fix:
	ISORT_FLAGS="$(ISORT_FLAGS)" ./aux/fix.sh $(F)

fixall:
	ISORT_FLAGS="$(ISORT_FLAGS)" ./aux/fix.sh $(PY_MYPY_FLAKE8)

commit:
	./aux/block_empty_commit.sh
	git commit -m "[ptblopgen] $$(cat ./gitmessage)" -e

show_version:
	@./aux/get_version.py --version-file=$(VER_FILE)

vercommit:
	git restore --staged $(VER_FILE)
	git checkout $(VER_FILE)
	./aux/block_empty_commit.sh
	./aux/update_version.py --version-segment=2 --version-file=$(VER_FILE)
	git add $(VER_FILE)
	-VER=$$(./aux/get_version.py --version-file=$(VER_FILE)) && git commit -m "[ptblopgen] $${VER} $$(cat ./gitmessage)" -e
	git restore --staged  $(VER_FILE) && git checkout $(VER_FILE) # In case of failed commit

manualvercommit:
	./aux/block_empty_commit.sh
	git add $(VER_FILE)
	VER=$$(./aux/get_version.py --version-file=$(VER_FILE)) && git commit -m "[ptblopgen] $${VER} $$(cat ./gitmessage)" -e

-%:
	-@$(MAKE) --no-print-directory $*

.PHONY: all build build_zip test clean fix fixall show_version\
	check  check_black check_isort check_flake8 check_mypy \
	commit vercommit manualvercommit \
    test_py test_full test

