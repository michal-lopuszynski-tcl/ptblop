#!/bin/bash

python3 run_imagenet.py \
   --imagenet-v1-path ~/Datasets/datahub/vision/imagenet/val \
   --imagenet-v2-path ~/Datasets/datahub/vision/imagenet-v2/imagenetv2-matched-frequency-format-val \
   --batch-size 64  \
   --model mobilevitv2_200.cvnets_in22k_ft_in1k
