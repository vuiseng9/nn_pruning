#!/usr/bin/env bash

export WANDB_DISABLED=false # Enable wandb
export WANDB_WATCH=false # Disable gradient serialization to wandb
export WANDB_USERNAME=vchua
export WANDB_API_KEY=f8a95080288950342f1695008cd8256adc3b0778

# ---------------------------------------------------------------------------------------------
export WANDB_PROJECT="nncf-mvmt-squad ($(hostname))"
export CUDA_VISIBLE_DEVICES=0

RUNID=run10-bert-squad-hybrid # !!!!! we need to manually align run_name in parameters.json to reflect this id in wandb
OUTROOT=/data1/vchua/run/ssbs-feb/bert-squad/
WORKDIR=/data1/vchua/dev/ssbs-feb/nn_pruning/reproduce-eval/question_answering

CONDAROOT=/data1/vchua
CONDAENV=ssbs-feb
# ---------------------------------------------------------------------------------------------
OUTDIR=$OUTROOT/$RUNID
mkdir -p $OUTDIR

cd $WORKDIR

# nohup 
cmd="
nn_pruning_run_example finetune \
    --json-path golden-parameters.json \
    squadv1 $OUTDIR
"
    # 2>&1 | tee $OUTDIR/run.log &

if [[ $1 == "local" ]]; then
    echo "${cmd}" > $OUTDIR/run.log
    echo "### End of CMD ---" >> $OUTDIR/run.log
    cmd="nohup ${cmd}"
    eval $cmd >> $OUTDIR/run.log 2>&1 &
    echo "logpath: $OUTDIR/run.log"
elif [[ $1 == "dryrun" ]]; then
    echo "[INFO]: dryrun mode is unsupported in this script"
    # echo "[INFO: dryrun, add --max_steps 25 to cli"
    # cmd="${cmd} --max_steps 25"
    # echo "${cmd}" > $OUTDIR/dryrun.log
    # echo "### End of CMD ---" >> $OUTDIR/dryrun.log
    # eval $cmd >> $OUTDIR/dryrun.log 2>&1 &
    # echo "logpath: $OUTDIR/dryrun.log"
else
    source $CONDAROOT/miniconda3/etc/profile.d/conda.sh
    conda activate ${CONDAENV}
    eval $cmd
fi

# Keep the following for references
# ----------------------------------

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


# export CUDA_VISIBLE_DEVICES=1

# OUTDIR=latest-run-bert-base-uncased-block-pruned-squadv1
# mkdir -p $OUTDIR

# nohup  nn_pruning_run_example finetune \
#     --json-path golden-parameters.json \
#     squadv1 $OUTDIR 2>&1 | tee $OUTDIR/run.log &