#!/bin/bash

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

echo -n "Started $(basename $0) "

rm -rf tmp.llm_paretoeval_blocks
mkdir -p tmp.llm_paretoeval_blocks

SECONDS=0
cp data/llm_pareto_front_blocks_0004.json tmp.llm_paretoeval_blocks
blop paretoeval \
   --config data/llm_config_blocks.yaml \
   --pareto-path  tmp.llm_paretoeval_blocks/llm_pareto_front_blocks_0004.json \
   > tmp.llm_paretoeval_blocks/log 2>&1
ERRCODE=$?
DURATION=$SECONDS
DURATION_STR="$((DURATION / 60)):$((DURATION % 60))"

if [ ${ERRCODE} -ne 0 ]; then
    echo -e "[${DURATION_STR}] ... ${RED}FAILED${NC}"
else
    echo -e "[${DURATION_STR}] ... ${GREEN}PASSED${NC}"
fi

exit ${ERRCODE}
