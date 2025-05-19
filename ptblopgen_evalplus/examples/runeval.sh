#!/bin/bash

rm -rf tmp
python3 evaluate.py \
    --model Qwen/Qwen3-0.6B \
    --dataset humaneval \
    --enable-thinking=False \
    --greedy
