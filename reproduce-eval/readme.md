### Objectives
This branch (reproduce-eval) aims to provide a step-by-step guide to reproduce a few results published in the paper. We used a specific transformers branching from v4.10.3 (only minor changes for better utility). 


### Setup
```bash
git clone https://github.com/vuiseng9/nn_pruning
cd nn_pruning && git checkout reproduce-evaluation

# clone patched transformers branching from v4.10.3
git submodule init
git submodule update

# Install nn-pruning
pip install -e ".[dev]"

# Install transformer
cd ../transformers
pip install -e .

# Install torch according to your system, following are ones used at my end for this documentation
# GPU
pip3 install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
# CPU
pip3 install torch==1.9.1+cpu torchvision==0.10.1+cpu torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
```

### bert-base-uncased-MNLI baseline
Gotcha! Many models in model hub are incompatible with v4.10.3, evaluation with run_glue.py for the task of MNLI shows unexpected low accuracy, at least it happened at my end. Therefore, we have fine-tuned one with 4.10.3 and shared to model hub, please checkout [vuiseng9/bert-base-uncased-mnli](https://huggingface.co/vuiseng9/bert-base-uncased-mnli). This model serves as baseline and as the teacher model of distillation.
1. Baseline Task Performance and Latency
    
    According to the paper, all benchmarks are performed with batch size of 128. See ```eval_mnli_results.json``` and ```eval_mnli-mm_results.json``` in the ```$OUTDIR``` for m/mm scores and evaluation latency. ```linear_layer_stats_total_75M.csv``` tabulates parameter breakdown per linear layer in the encoder.
    ```bash
    export CUDA_VISIBLE_DEVICES=0

    OUTDIR=baseline-bert-based-uncased-mnli
    WORKDIR=nn_pruning/transformers/examples/pytorch/text-classification
    cd $WORKDIR

    nohup python run_glue.py \
        --model_name_or_path vuiseng9/bert-base-uncased-mnli  \
        --task_name mnli  \
        --do_eval  \
        --per_device_eval_batch_size 128  \
        --max_seq_length 128  \
        --overwrite_output_dir \
        --output_dir $OUTDIR 2>&1 | tee $OUTDIR/run.log &
    ```

1. Block Fine-Pruning of bert-base-uncased for MNLI
    ```bash
    cd nn_pruning/reproduce-eval/text_classification
    ./fine-pruning-mnli.sh
    ```
    


### Benchmark Block-pruned Squad
```
cd nn_pruning/reproduce-eval
./benchmark-qa.sh
```

### Download archived checkpoint and Generate Analysis
```bash
cd nn_pruning/reproduce-eval/scripts
python generate_bash_s3_crawler.py

#├── results_mnli_download_ckpt.sh
#├── results_squadv1_download_ckpt.sh
#└── results_squadv2_download_ckpt.sh

mkdir -p /data/hf-block-pruning/mnli
cd /data/hf-block-pruning/mnli
cp nn_pruning/reproduce-eval/scripts/results_mnli_download_ckpt.sh .
./results_mnli_download_ckpt.sh

# generate analysis
cd nn_pruning/analysis
python command_line.py analyze /data/hf-block-pruning/ --task mnli --output /data/hf-block-pruning/mnli/ckpt_analysis
# ckpt_analysis_mnli.json will be generated
```
