#!/bin/bash

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

echo -n "Started $(basename $0) "

rm -rf tmp.sample_blocks
mkdir -p tmp.sample_blocks

SECONDS=0
blop sample \
  --config data/config_blocks.yaml \
  --output-path tmp.sample_blocks \
  > tmp.sample_blocks/log 2>&1
ERRCODE=$?
DURATION=$SECONDS
DURATION_STR="$((DURATION / 60)):$((DURATION % 60))"


if [ ${ERRCODE} -ne 0 ]; then
    echo -e "[${DURATION_STR}] ... ${RED}FAILED${NC}"
else
    echo -e "[${DURATION_STR}] ... ${GREEN}PASSED${NC}"
fi

exit ${ERRCODE}
