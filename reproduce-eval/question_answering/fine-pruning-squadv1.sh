#!/usr/bin/env bash

# export CUDA_VISIBLE_DEVICES=0

# OUTDIR=latest-run-bert-base-uncased-block-pruned-squadv1
# mkdir -p $OUTDIR
# nohup  nn_pruning_run_example finetune \
#     --model-name-or-path bert-base-uncased \
#     --teacher bert-large-uncased-whole-word-masking-finetuned-squad \
#     --per-device-train-batch-size 16 \
#     --regularization-final-lambda 20.0 \
#     --dense-lambda 1.0 \
#     --ampere-pruning-method disabled \
#     squadv1 \
#     $OUTDIR 2>&1 | tee $OUTDIR/run.log &


export CUDA_VISIBLE_DEVICES=1

OUTDIR=latest-run-bert-base-uncased-block-pruned-squadv1
mkdir -p $OUTDIR

nohup  nn_pruning_run_example finetune \
    --json-path golden-parameters.json \
    squadv1 $OUTDIR 2>&1 | tee $OUTDIR/run.log &