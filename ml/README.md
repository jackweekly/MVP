# Machine Learning Scaffold

This folder contains the initial scaffolding for experimenting with learning-to-optimize
strategies on top of the Rust VRP solver. The goal is to support two workflows:

1. **Data collection** – ingest JSONL logs emitted by the solver (via `RunRecorder`).
2. **Policy experimentation** – build and train prototype models that score moves or
   predict solver hyper-parameters.

## Layout

```
ml/
  README.md                – you are here
  requirements.txt         – python dependencies for experimentation
  src/
    __init__.py            – package marker
    dataset.py             – utilities for loading solver logs
    train_policy.py        – baseline training loop for a move-scoring policy
```

## Getting Started

1. Create a Python environment (conda or venv) and install dependencies:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. Collect solver logs (JSONL) using the new CLI helper:

   ```bash
   cargo run --bin vrp-dataset -- collect --input data/problems.json --output logs/
   ```

3. Run the baseline policy training script:

   ```bash
   python -m ml.src.train_policy --logs logs/run_*.jsonl --epochs 10
   ```

The baseline training loop is intentionally lightweight; replace the simple
feed-forward network with a GNN or reinforcement learning agent as you iterate on
feature engineering.
