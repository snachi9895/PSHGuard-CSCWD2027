# Recorded software environments

The experiments were run under Python 3.10.20. The principal recorded environments were:

## Graph branch

- NumPy 2.2.6
- pandas 2.3.3
- PyTorch 2.12.0.dev20260407+cu128
- PyTorch Geometric 2.7.0
- scikit-learn 1.7.2
- tqdm 4.67.3

## Sequence branch

- NumPy 2.2.6
- pandas 2.3.3
- PyTorch 2.11.0+cu128
- transformers 4.57.6
- scikit-learn 1.7.2

The development PyTorch build for the graph branch may not be available from a standard package index. `requirements.txt` therefore records portable compatibility ranges, while this file preserves the exact observed versions. CUDA and PyTorch builds should be selected together for the review machine.

