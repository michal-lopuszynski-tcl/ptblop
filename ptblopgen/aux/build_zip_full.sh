#!/bin/bash

if [ $# -ne 1 ]; then
    exit 1
fi

ZIP_DIR="${1}/full"
TMP_DIR=tmp.$$

rm -rf ${ZIP_DIR}

ZIP_SUFFIX="-mod"
git diff --exit-code --quiet && ZIP_SUFFIX=""
ZIP_NAME="ptblopgen-full-$(git rev-list --count HEAD)-$(git rev-parse --short HEAD)${ZIP_SUFFIX}.zip"

mkdir -p ${ZIP_DIR}/${TMP_DIR}
cd ${ZIP_DIR}/${TMP_DIR}

python3 -mpip install --no-dependencies ../../../../ptblop --target .
python3 -mpip install --no-dependencies ../../../../ptblopgen --target .
python3 -mpip install --no-dependencies ../../../../ptblopgen_evalplus --target .

printf "import sys\nimport ptblopgen.cli\n\nsys.exit(ptblopgen.cli.main())\n" > __main__.py

find . | cut -c3- | zip -r ../${ZIP_NAME} -@

cd ..

rm -rf ${TMP_DIR}
