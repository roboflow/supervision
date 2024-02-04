#!/bin/bash

# DATASET points to the directory containing pa100k/
DATASET=$1

python main.py \
--root ${DATASET} \
-d pa100k \
-a osnet_maxpool \
--max-epoch 50 \
--stepsize 30 40 \
--batch-size 32 \
--lr 0.065 \
--optim sgd \
--weighted-bce \
--save-dir log/pa100k-osnet_maxpool