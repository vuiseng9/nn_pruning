### Setup
```bash
git clone https://github.com/vuiseng9/nn_pruning
cd nn_pruning && git checkout reproduce-evaluation
git submodule init
git submodule update

# Install nn-pruning
pip install -e ".[dev]"

# Install transformer
cd ../transformers
pip install -e .

# Install torch
# GPU
pip3 install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
# CPU
pip3 install torch==1.9.1+cpu torchvision==0.10.1+cpu torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
```

### Benchmark Block-pruned Squad
```
cd nn_pruning/reproduce-eval
./benchmark-qa.sh
```