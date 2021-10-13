from transformers import pipeline
from nn_pruning.inference_model_patcher import optimize_model
from datetime import datetime
import time

PROF_ITER=100
class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# BERT-base uncased finetuned on SQuAD, speedup is 2.44, F1=87.7, 26% of linear layer parameters remaining,
# with hybrid-pruning + final fill -> dense matrices
# MODEL_NAME = "madlag/bert-base-uncased-squadv1-x2.44-f87.7-d26-hybrid-filled-v1"
# MODEL_NAME = "/home/vchua/blk-prune/nn_pruning/reproduce-eval/question_answering/run-bert-based-block-pruned-squadv1/checkpoint-95000"
MODEL_NAME = "/tmp/vscode-runs/dbg-bert-based-block-pruned-squadv1/compiled_checkpoint"

qa_pipeline = pipeline(
    "question-answering",
    model=MODEL_NAME,
)

orig_lat = AverageMeter()
for i in range(PROF_ITER):
    end = time.time()
    predictions = qa_pipeline({
        'context': "Frédéric François Chopin, born Fryderyk Franciszek Chopin (1 March 1810 – 17 October 1849), was a Polish composer and virtuoso pianist of the Romantic era who wrote primarily for solo piano.",
        'question': "Who is Frederic Chopin?",
    })
    orig_lat.update(time.time() - end)
print("Original Latency: {:.3f}s".format(orig_lat.avg))
print("Predictions", predictions)
print()

# Original BERT-base size
original_bert = 110E6
print(f"BERT-base parameters: {original_bert/1E6:0.1f}M")

# Optimize the model: this just removes the empty parts of the model (lines/columns), as we
# cannot currently store the shrunk version on disk in a huggingface transformers compatible format
qa_pipeline.model = optimize_model(qa_pipeline.model, "dense")

# Check the new size
new_count = int(qa_pipeline.model.num_parameters())
print(f"Parameters count after optimization={new_count / 1E6:0.1f}M")
print(f"Reduction of the total number of parameters compared to BERT-base:{original_bert / new_count:0.2f}X")

opt_lat = AverageMeter()
for i in range(PROF_ITER):
    end = time.time()
    predictions = qa_pipeline({
        'context': "Frédéric François Chopin, born Fryderyk Franciszek Chopin (1 March 1810 – 17 October 1849), was a Polish composer and virtuoso pianist of the Romantic era who wrote primarily for solo piano.",
        'question': "Who is Frederic Chopin?",
    })
    opt_lat.update(time.time() - end)
print("Post-pruned Latency: {:.3f}".format(opt_lat.avg))
print("Predictions", predictions)
print()
print("Speedup: {:2.2f}".format(orig_lat.avg/opt_lat.avg))