#!/bin/bash

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

echo -n "Started $(basename $0) "

rm -rf tmp.llm_paretofind
mkdir tmp.llm_paretofind
SECONDS=0
blop paretofind --config data/llm_config.yaml \
    --output-path tmp.llm_paretofind \
    --bp-configs-path data/llm_bp_configs.json \
    > tmp.llm_paretofind/log 2>&1
ERRCODE=$?
DURATION=$SECONDS
DURATION_STR="$((DURATION / 60)):$((DURATION % 60))"


if [ ${ERRCODE} -ne 0 ]; then
    echo -e "[${DURATION_STR}] ... ${RED}FAILED${NC}"
else
    echo -e "[${DURATION_STR}] ... ${GREEN}PASSED${NC}"
fi

exit ${ERRCODE}
