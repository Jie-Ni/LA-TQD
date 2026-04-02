"""
run_ablation.py -- Ablation study: effect of composition priors.

Runs four configurations:
  1. LA-TQD Full (both priors active)
  2. w/o Aromatic Anchor prior
  3. w/o Poly Constraint prior
  4. w/o Both (no composition priors)

Usage:
    python scripts/run_ablation.py --model mock --rounds 5
    python scripts/run_ablation.py --model Qwen/Qwen2.5-72B-Instruct --rounds 30
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.archive import MapElitesArchive
from src.config import DATA_DIR, N_ROUNDS, N_RUNS, RESULTS_DIR, MAX_AA_FREQ, MIN_AROMATIC
from src.metrics import compute_all_metrics
from src.oracle import fitness, is_valid_basic
from src.biophysics import aromatic_count, max_single_aa_count

import random
import numpy as np


def _passes_priors(seq: str, use_aromatic: bool, use_poly: bool) -> bool:
    """Check composition priors based on ablation flags."""
    if use_poly and max_single_aa_count(seq) > MAX_AA_FREQ:
        return False
    if use_aromatic and aromatic_count(seq) < MIN_AROMATIC:
        return False
    return True


def run_config(mutator, seeds, n_rounds, n_runs, use_aromatic, use_poly,
               esm_model=None, esm_tokenizer=None, device="cpu", name="config") -> dict:
    per_cov, per_hack = [], []
    for run_idx in range(n_runs):
        random.seed(42 + run_idx)
        np.random.seed(42 + run_idx)
        archive = MapElitesArchive()
        all_gen = []

        for seq in seeds:
            seq = seq.upper().strip()
            if not is_valid_basic(seq):
                continue
            if not _passes_priors(seq, use_aromatic, use_poly):
                continue
            score = fitness(seq, esm_model, esm_tokenizer, device=device, enforce_priors=False)
            if score != float("inf"):
                archive.add(seq, score, generation=0)

        for rnd in range(1, n_rounds + 1):
            parent, shift = archive.sample_with_target()
            if parent is None:
                continue
            child = mutator.mutate(parent, shift)
            if child is None:
                continue
            child = child.upper().strip()
            all_gen.append(child)
            if not is_valid_basic(child):
                continue
            if not _passes_priors(child, use_aromatic, use_poly):
                continue
            score = fitness(child, esm_model, esm_tokenizer, device=device, enforce_priors=False)
            if score != float("inf"):
                archive.add(child, score, generation=rnd)

        m = compute_all_metrics(archive, all_gen)
        per_cov.append(m["coverage_pct"])
        per_hack.append(m["hacking_rate_pct"])
        print(f"  [{name}] Run {run_idx+1}: cov={m['coverage_pct']:.2f}% hack={m['hacking_rate_pct']:.2f}%")

    return {"config": name,
            "coverage_mean": float(np.mean(per_cov)), "coverage_std": float(np.std(per_cov)),
            "hacking_mean": float(np.mean(per_hack)), "hacking_std": float(np.std(per_hack))}


def main():
    parser = argparse.ArgumentParser(description="LA-TQD Ablation Study")
    parser.add_argument("--model", default="mock")
    parser.add_argument("--rounds", type=int, default=N_ROUNDS)
    parser.add_argument("--runs", type=int, default=N_RUNS)
    parser.add_argument("--no-esm", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default=os.path.join(RESULTS_DIR, "ablation"))
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(DATA_DIR, "seed_sequences.json")) as f:
        data = json.load(f)
    seeds = data if isinstance(data, list) else data["sequences"]

    if args.model.lower() == "mock":
        from src.llm_mutator import MockMutator
        mutator = MockMutator()
    else:
        from src.llm_mutator import LLMMutator
        mutator = LLMMutator(args.model, device=args.device)

    esm_model, esm_tokenizer = None, None
    if not args.no_esm:
        from src.esm_scorer import load_esm2
        esm_model, esm_tokenizer = load_esm2(device=args.device)

    configs = [
        ("Full",             True,  True),
        ("w/o Arom. Anchor", False, True),
        ("w/o Poly. Const.", True,  False),
        ("w/o Both",         False, False),
    ]

    all_results = []
    for name, arom, poly in configs:
        print(f"\n[Ablation] {name}")
        result = run_config(mutator, seeds, args.rounds, args.runs, arom, poly,
                            esm_model, esm_tokenizer, args.device, name)
        all_results.append(result)

    print("\n" + "=" * 55)
    print(f"{'Config':<25} {'Coverage (%)':>15} {'Hacking (%)':>14}")
    print("-" * 55)
    for r in all_results:
        print(f"{r['config']:<25} {r['coverage_mean']:>7.2f}+/-{r['coverage_std']:<5.2f} "
              f"{r['hacking_mean']:>7.2f}+/-{r['hacking_std']:.2f}")

    out = os.path.join(args.output_dir, "ablation_results.json")
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
