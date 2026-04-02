"""
run_relaxed.py — Evaluation under relaxed length constraints (L = 12–18).

Compares LA-TQD with strict (L=15) vs relaxed (L=12-18) sequence length.

Usage:
    python scripts/run_relaxed.py --model mock --rounds 5
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.archive import MapElitesArchive
from src.config import (DATA_DIR, MAX_AA_FREQ, MIN_AROMATIC, N_ROUNDS, N_RUNS,
                        RELAXED_MAX_LEN, RELAXED_MIN_LEN, RESULTS_DIR)
from src.biophysics import aromatic_count, max_single_aa_count
from src.oracle import check_alphabet, check_rgd_motif, fitness
from src.metrics import compute_all_metrics


def is_valid_relaxed(seq: str) -> bool:
    seq = seq.upper().strip()
    if not (RELAXED_MIN_LEN <= len(seq) <= RELAXED_MAX_LEN):
        return False
    if not check_alphabet(seq):
        return False
    if not check_rgd_motif(seq):
        return False
    if max_single_aa_count(seq) > MAX_AA_FREQ:
        return False
    if aromatic_count(seq) < MIN_AROMATIC:
        return False
    return True


def run_relaxed(mutator, seeds, n_rounds, n_runs, esm_model=None, esm_tokenizer=None, device="cpu") -> dict:
    per_cov, per_hack = [], []
    for run_idx in range(n_runs):
        random.seed(42 + run_idx)
        np.random.seed(42 + run_idx)
        archive = MapElitesArchive()
        all_gen = []

        for seq in seeds:
            seq = seq.upper().strip()
            if is_valid_relaxed(seq):
                score = fitness(seq, esm_model, esm_tokenizer, length=len(seq))
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
            if is_valid_relaxed(child):
                score = fitness(child, esm_model, esm_tokenizer, length=len(child))
                if score != float("inf"):
                    archive.add(child, score, generation=rnd)

        m = compute_all_metrics(archive, all_gen)
        per_cov.append(m["coverage_pct"])
        per_hack.append(m["hacking_rate_pct"])
        print(f"  [Relaxed] Run {run_idx+1}: cov={m['coverage_pct']:.2f}%")

    return {"coverage_mean": float(np.mean(per_cov)), "coverage_std": float(np.std(per_cov)),
            "hacking_mean": float(np.mean(per_hack)), "hacking_std": float(np.std(per_hack))}


def main():
    parser = argparse.ArgumentParser(description="LA-TQD Relaxed Length Constraint Experiment")
    parser.add_argument("--model", default="mock")
    parser.add_argument("--rounds", type=int, default=N_ROUNDS)
    parser.add_argument("--runs", type=int, default=N_RUNS)
    parser.add_argument("--no-esm", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default=os.path.join(RESULTS_DIR, "relaxed"))
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

    from src.search import run_experiment
    strict = run_experiment(mutator=mutator, seeds=seeds, n_runs=args.runs, n_rounds=args.rounds,
                            esm_model=esm_model, esm_tokenizer=esm_tokenizer, device=args.device,
                            save_dir=args.output_dir, experiment_name="strict")
    relaxed = run_relaxed(mutator, seeds, args.rounds, args.runs, esm_model, esm_tokenizer, args.device)

    results = {"strict": {"coverage_mean": strict["coverage_mean"], "coverage_std": strict["coverage_std"]},
               "relaxed": relaxed}
    out = os.path.join(args.output_dir, "relaxed_results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)

    sr, rr = results["strict"], results["relaxed"]
    print(f"\nStrict (L=15):   {sr['coverage_mean']:.2f}+/-{sr['coverage_std']:.2f}%")
    print(f"Relaxed (L=12-18):{rr['coverage_mean']:.2f}+/-{rr['coverage_std']:.2f}%")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
