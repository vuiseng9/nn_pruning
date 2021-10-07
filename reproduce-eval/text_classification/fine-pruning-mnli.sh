#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0

OUTDIR=latest-run-bert-base-uncased-block-pruned-mnli
mkdir -p $OUTDIR
# nohup  nn_pruning_run_example finetune \
#     --json-path parameters.json \
#     mnli $OUTDIR 2>&1 | tee $OUTDIR/run.log &

# nohup  nn_pruning_run_example finetune \
#     --model-name-or-path bert-base-uncased \
#     --per-device-train-batch-size 32 \
#     --regularization-final-lambda 5.0 \
#     --dense-lambda 1.0 \
#     --ampere-pruning-method disabled \
#     mnli $OUTDIR 2>&1 | tee $OUTDIR/run.log &

nohup  nn_pruning_run_example finetune \
    --model-name-or-path bert-base-uncased \
    --teacher vuiseng9/bert-base-uncased-mnli \
    --per-device-train-batch-size 32 \
    --regularization-final-lambda 5.0 \
    --dense-lambda 1.0 \
    --ampere-pruning-method disabled \
    mnli $OUTDIR 2>&1 | tee $OUTDIR/run.log &
