#!/bin/bash

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

echo -n "Started $(basename $0) "

rm -rf tmp.llm_paretoeval
mkdir -p tmp.llm_paretoeval

SECONDS=0
cp data/llm_pareto_front_0004.json tmp.llm_paretoeval
blop paretoeval \
   --config data/llm_config.yaml \
   --pareto-path  tmp.llm_paretoeval/llm_pareto_front_0004.json \
   --min-mparams 200 \
   --max-mparams 400 \
   --pareto-level 2 \
   > tmp.llm_paretoeval/log 2>&1
ERRCODE=$?
DURATION=$SECONDS
DURATION_STR="$((DURATION / 60)):$((DURATION % 60))"

if [ ${ERRCODE} -ne 0 ]; then
    echo -e "[${DURATION_STR}] ... ${RED}FAILED${NC}"
else
    echo -e "[${DURATION_STR}] ... ${GREEN}PASSED${NC}"
fi

exit ${ERRCODE}
