#!/bin/bash

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

echo -n "Started $(basename $0) "

rm -rf tmp.paretoeval
mkdir -p tmp.paretoeval

SECONDS=0
cp data/llm_pareto_front_0004.json tmp.paretoeval
blop paretoeval \
   --config data/llm_config.yaml \
   --pareto-path  tmp.paretoeval/llm_pareto_front_0004.json \
   --min-mparams 200 \
   --max-mparams 400 \
   --pareto-level 2 \
   > tmp.paretoeval/log 2>&1
ERRCODE=$?
DURATION=$SECONDS
DURATION_STR="$((DURATION / 60)):$((DURATION % 60))"

if [ ${ERRCODE} -ne 0 ]; then
    echo -e "[${DURATION_STR}] ... ${RED}FAILED${NC}"
else
    echo -e "[${DURATION_STR}] ... ${GREEN}PASSED${NC}"
fi

exit ${ERRCODE}
