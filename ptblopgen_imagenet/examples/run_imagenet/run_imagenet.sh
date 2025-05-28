#!/bin/bash

OUT_DIRNAME=out
#MODEL_NAME=mobilevitv2_200.cvnets_in22k_ft_in1k
MODEL_NAME=deit3_small_patch16_224.fb_in1k
RESULTS_FNAME="${OUT_DIRNAME}/${MODEL_NAME}.json"
mkdir -p ${OUT_DIRNAME}

python3 run_imagenet.py \
   --imagenet-v1-path ~/Datasets/datahub/vision/imagenet/val \
   --imagenet-v2-path ~/Datasets/datahub/vision/imagenet-v2/imagenetv2-matched-frequency-format-val \
   --batch-size 128  \
   --model ${MODEL_NAME} \
   --results ${RESULTS_FNAME}
