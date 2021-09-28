#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0

OUTDIR=run-benchmark-madlag-bert-base-uncased-squadv1-x2.44-f87.7-d26-hybrid-filled-v1
mkdir -p $OUTDIR

cd ../transformers/examples/pytorch/question-answering
python benchmark_qa.py \
    --model_name_or_path madlag/bert-base-uncased-squadv1-x2.44-f87.7-d26-hybrid-filled-v1 \
    --dataset_name squad \
    --do_eval \
    --per_device_eval_batch_size 384 \
    --max_seq_length 384 \
    --doc_stride 128 \
    --output_dir $OUTDIR 2>&1 | tee $OUTDIR/run.log &