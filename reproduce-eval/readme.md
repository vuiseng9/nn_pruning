```bash
git clone https://github.com/vuiseng9/nn_pruning
cd nn_pruning && git checkoutout reproduce-evaluation
git submodule init
git submodule update

# Install nn-pruning
pip install -e ".[dev]"

# Install transformer
cd ../transformers
pip install -e .
```