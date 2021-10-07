#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=1

OUTDIR=latest-run-bert-base-uncased-block-pruned-squadv1-final-finetune
mkdir -p $OUTDIR

nohup  nn_pruning_run_example final-finetune \
    --checkpoint latest-run-bert-base-uncased-block-pruned-squadv1/checkpoint-110000 \
    --teacher bert-large-uncased-whole-word-masking-finetuned-squad \
    $OUTDIR \
    squad 2>&1 | tee $OUTDIR/run.log &
