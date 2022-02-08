## Objectives
This branch (reproduce-eval) aims to provide a step-by-step guide to reproduce a few results published in the paper. We used a specific transformers branching from v4.10.3 (only minor changes for better utility). 


### Setup
```bash
git clone https://github.com/vuiseng9/nn_pruning
cd nn_pruning && git checkout reproduce-evaluation

# Install nn-pruning
pip install -e ".[dev]"

# Install transformer
cd ../transformers
git checkout tags/v4.9.1 -b v4.9.1 # v4.10.3 was validated as well
pip install -e .

# Install torch according to your system, following are ones used at my end for this documentation
# GPU
pip3 install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
# CPU
pip3 install torch==1.9.1+cpu torchvision==0.10.1+cpu torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
```

## bert-base-uncased-MNLI
Gotcha! Many models in model hub are incompatible with v4.10.3, evaluation with run_glue.py for the task of MNLI shows unexpected low accuracy, at least it happened at my end. Therefore, we have fine-tuned one with 4.10.3 and shared to model hub, please checkout [vuiseng9/bert-base-uncased-mnli](https://huggingface.co/vuiseng9/bert-base-uncased-mnli). This model serves as baseline and as the teacher model of distillation.
1. Baseline Task Performance and Latency
    
    According to the paper, all benchmarks are performed with batch size of 128. See ```eval_mnli_results.json``` and ```eval_mnli-mm_results.json``` in the ```$OUTDIR``` for m/mm scores and evaluation latency. ```linear_layer_stats_total_86M.csv``` tabulates parameter breakdown per linear layer in the encoder.
    ```bash
    export CUDA_VISIBLE_DEVICES=0

    OUTDIR=baseline-bert-based-uncased-mnli
    WORKDIR=nn_pruning/transformers/examples/pytorch/text-classification
    cd $WORKDIR && mkdir -p $OUTDIR

    nohup python run_glue.py \
        --model_name_or_path vuiseng9/bert-base-uncased-mnli  \
        --task_name mnli  \
        --do_eval  \
        --per_device_eval_batch_size 128  \
        --max_seq_length 128  \
        --overwrite_output_dir \
        --output_dir $OUTDIR 2>&1 | tee $OUTDIR/run.log &
    ```

1. Block Fine-Pruning of bert-base-uncased for MNLI (hybrid mode w/ distillation)
    ```bash
    cd nn_pruning/reproduce-eval/text_classification
    ./fine-pruning-mnli.sh
    ```
    The run outputs to ```nn_pruning/reproduce-eval/text_classification/latest-run-bert-base-uncased-block-pruned-mnli```. The final model will be have the head pruned if any and saved as ```compiled_checkpoint```. Execute the following command with the compiled model to obtain post-optimization task performance and latency. Do note that the ```--optimize_model_before_eval``` is needed to crop the linear FFNN layers.
    ```bash
    export CUDA_VISIBLE_DEVICES=0

    OUTDIR=blk-pruned-bert-based-uncased-mnli
    WORKDIR=nn_pruning/transformers/examples/pytorch/text-classification
    cd $WORKDIR && mkdir -p $OUTDIR

    nohup python run_glue.py \
        --model_name_or_path <path/to>/nn_pruning/reproduce-eval/text_classification/latest-run-bert-base-uncased-block-pruned-mnli/compiled_checkpoint  \
        --task_name mnli  \
        --do_eval  \
        --optimize_model_before_eval \
        --per_device_eval_batch_size 128  \
        --max_seq_length 128  \
        --overwrite_output_dir \
        --output_dir $OUTDIR 2>&1 | tee $OUTDIR/run.log &
    ```


## bert-base-uncased-squadv1
```csarron/bert-base-uncased-squad-v1``` is referenced as SQuADv1-tuned model in the original implementation. We follow the same.
1. Baseline Task Performance and Latency
    Similar to MNLI above, batch size of 128 is set. See ```$OUTDIR``` for accuracy and latency reports. 
    ```bash
    export CUDA_VISIBLE_DEVICES=0

    OUTDIR=baseline-bert-based-uncased-squad1
    WORKDIR=nn_pruning/transformers/examples/pytorch/question-answering
    cd $WORKDIR && mkdir -p $OUTDIR

    nohup python run_qa.py \
        --model_name_or_path csarron/bert-base-uncased-squad-v1  \
        --dataset_name squad  \
        --do_eval  \
        --per_device_eval_batch_size 128  \
        --max_seq_length 128  \
        --doc_stride 128  \
        --overwrite_output_dir  \
        --output_dir $OUTDIR 2>&1 | tee $OUTDIR/run.log &
    ```

1. Block Fine-Pruning of bert-base-uncased for Squadv1 (hybrid mode w/ large teacher distillation)
    ```bash
    cd nn_pruning/reproduce-eval/question_answering
    ./fine-pruning-squadv1.sh
    ```
1. Final fine-tuning (Hybrid "Filled" mode)
    From landing page:
    - *"Hybrid : prune using blocks for attention and rows/columns for the two large FFNs."*
    - *"Filled : remove empty heads and empty rows/columns of the FFNs, then re-finetune the previous network, letting the zeros in non-empty attention heads evolve and so regain some accuracy while keeping the same network speed."*
    ```bash
    cd nn_pruning/reproduce-eval/question_answering
    ./final-finetune-squadv1.sh
    ```

## Official Archived Runs
Thanks to the authors, runs published in the paper have been archived in S3 and are accessible for our reference.

Reference runs for two models above:
```
# MNLI BERT-base hybrid
aws s3 sync s3://lagunas-sparsity-experiments/backup/nn_pruning/output/mnli_test2/hp_od-output__mnli_test2___pdtbs32_pdebs128_nte12_ws12000_rn-output__mnli_test2___fw4_rfl5/ full_run_mnli_bert_base_hybrid_tinybert_speed_equivalent/

# Squad BERT-base hybrid run
aws s3 ls s3://lagunas-sparsity-experiments/backup/nn_pruning/output/squad_test4/hp_od-__data_2to__devel_data__nn_pruning__output__squad4___es-steps_nte20_ls250_stl50_est5000_rn-__data_2to__devel_data__nn_pruning__output__squad4___dpm-sigmoied_threshold:1d_alt_ap--17cd29ad8a563746/

# Squad BERT-base hybrid-filled LT run (checkpoint 110000 above was used)
aws s3 sync s3://lagunas-sparsity-experiments/backup/nn_pruning/output/squad_test_final_fine_tune/fine_tuned_hp_od-__data_2to__devel_data__nn_pruning__output__squad4___es-steps_nte20_ls250_stl50_est5000_rn-__data_2to__devel_data__nn_pruning__output__squad4___dpm-sigmoied_threshold:1d_alt_ap--17cd29ad8a563746/ /data/hf-block-pruning/squadv1/full_run_squadv1_bert_base_hybrid_filled_lt_f188p3_2p31x
```
#### Download archived checkpoint and Generate Analysis
AWS CLI is required.
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

<!-- ### Benchmark Block-pruned Squad

# quantization in pytest requires this version
pip3 install torch==1.8.2+cu111 torchvision==0.9.2+cu111 torchaudio===0.8.2 -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html
``` -->


