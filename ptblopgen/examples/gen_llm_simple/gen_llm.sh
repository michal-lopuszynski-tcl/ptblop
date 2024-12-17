#!/bin/bash

touch tmp1
rm -rf tmp1
mkdir -p tmp1
blop gen --config gen_llm.yaml --output-path tmp1 2>&1 | tee tmp1/log
