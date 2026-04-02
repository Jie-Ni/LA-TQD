# LA-TQD

Quality-Diversity optimization framework for constrained peptide generation using frozen LLMs as mutation operators.

## Overview

LA-TQD combines a MAP-Elites behavioral archive with a structured prompt-guided LLM mutation engine. A deterministic oracle filters generated sequences using two lightweight composition rules before inserting them into a 10×10 charge/hydropathy grid.

**Key design choices:**
- Training-free: the LLM is never fine-tuned
- Two composition priors prevent sequence degeneracy (max AA frequency ≤4, ≥1 aromatic residue)
- ESM-2 (650M) pseudo-perplexity as a sequence plausibility score
- Supports any HuggingFace causal LM (Qwen, Mistral, Llama, etc.)

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Quick test (no GPU)
python main.py --smoke-test

# Full experiment
python main.py --model Qwen/Qwen2.5-7B-Instruct --rounds 30 --runs 5

# 4-bit quantization (low VRAM)
python main.py --model Qwen/Qwen2.5-72B-Instruct --rounds 30 --load-in-4bit

# Skip ESM-2 scoring (fast mode)
python main.py --model mock --rounds 30 --no-esm
```

## Project Structure

```
├── main.py              # Entry point (CLI)
├── src/
│   ├── config.py        # All hyperparameters and constants
│   ├── biophysics.py    # Hydropathy, net charge, Shannon entropy
│   ├── oracle.py        # Composition filters + fitness scoring
│   ├── archive.py       # MAP-Elites 10×10 grid
│   ├── llm_mutator.py   # LLM mutation engine + prompt builder
│   ├── search.py        # Main QD search loop
│   ├── baselines.py     # GA (BLOSUM62) and Vanilla LLM baselines
│   ├── metrics.py       # Coverage, hacking rate, Levenshtein distance
│   └── esm_scorer.py    # ESM-2 pseudo-perplexity loader
├── scripts/
│   ├── run_ablation.py  # Ablation: with/without composition priors
│   ├── run_scaling.py   # Scaling across model families and sizes
│   └── run_relaxed.py   # Generalization to variable-length sequences
└── data/
    └── seed_sequences.json
```

## Experiments

```bash
# Ablation study (effect of composition priors)
python scripts/run_ablation.py --model Qwen/Qwen2.5-72B-Instruct --rounds 30

# Model scaling experiment
python scripts/run_scaling.py --rounds 30 --load-in-4bit

# Relaxed length constraint (12–18 residues)
python scripts/run_relaxed.py --model Qwen/Qwen2.5-72B-Instruct --rounds 30
```

## Core Algorithm

The fitness function balances sequence naturalness and biophysical relevance:

```
E(S) = 0.7 × pplx_ESM2(S) + 0.3 × |charge(S) − charge_target|
```

Sequences are only admitted to the archive if they pass both composition priors:
1. No single amino acid appears more than 4 times
2. At least one aromatic residue (F, W, or Y) is present

## Requirements

- Python 3.9+
- PyTorch 2.1+
- transformers ≥ 4.40
- NVIDIA GPU recommended (A100 for 70B models)
