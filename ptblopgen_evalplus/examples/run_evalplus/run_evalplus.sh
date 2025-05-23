#!/bin/bash

rm -rf tmp
TOKENIZERS_PARALLELISM=false python3 run_evalplus.py \
    --model Qwen/Qwen3-0.6B \
    --dataset humaneval \
    --enable-thinking=False \
    --limit 0.122 \
    --greedy
