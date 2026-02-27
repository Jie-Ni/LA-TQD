"""
search.py — Main QD search loop.
"""
from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.archive import MapElitesArchive
from src.config import C_TARGET, N_ROUNDS, N_RUNS, RESULTS_DIR
from src.oracle import oracle_evaluate


def _load_seeds(seed_path: str) -> List[str]:
    with open(seed_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else data.get("sequences", [])


def initialize_archive(seeds, archive, esm_model=None, esm_tokenizer=None, device="cpu"):
    for seq in seeds:
        seq = seq.upper().strip()
        valid, score = oracle_evaluate(seq, esm_model, esm_tokenizer, C_TARGET, device)
        if valid:
            archive.add(seq, score, generation=0)
    print(f"[init] {archive}")
    return archive


def run_qd_search(
    mutator,
    seeds: List[str],
    n_rounds: int = N_ROUNDS,
    esm_model=None,
    esm_tokenizer=None,
    c_target: float = C_TARGET,
    device: str = "cpu",
    verbose: bool = True,
    strict_length: bool = True,
) -> Tuple[MapElitesArchive, List[float], List[str]]:
    """One complete MAP-Elites QD search run.

    Each round:
      1. Sample parent + shift from archive
      2. LLM proposes a mutated sequence
      3. Oracle evaluates validity + fitness
      4. Insert into archive if it improves the cell
    """
    archive = MapElitesArchive()
    initialize_archive(seeds, archive, esm_model, esm_tokenizer, device)

    coverage_curve: List[float] = [archive.coverage_percent()]
    all_generated: List[str] = list(seeds)

    for rnd in range(1, n_rounds + 1):
        t0 = time.time()
        parent, shift = archive.sample_with_target()
        if parent is None:
            coverage_curve.append(archive.coverage_percent())
            continue

        child = mutator.mutate(parent, shift)
        if child is None:
            coverage_curve.append(archive.coverage_percent())
            continue

        child = child.upper().strip()
        all_generated.append(child)

        valid, score = oracle_evaluate(child, esm_model, esm_tokenizer, c_target, device)
        accepted = archive.add(child, score, generation=rnd) if valid else False

        coverage_curve.append(archive.coverage_percent())
        if verbose:
            status = "[OK]" if accepted else ("[valid]" if valid else "[SKIP]")
            print(
                f"[Round {rnd:02d}] {parent} -> {child} "
                f"| valid={valid} score={score:.3f} {status} "
                f"| cov={archive.coverage_percent():.1f}% t={time.time()-t0:.1f}s"
            )

    return archive, coverage_curve, all_generated


def run_experiment(
    mutator,
    seeds: List[str],
    n_runs: int = N_RUNS,
    n_rounds: int = N_ROUNDS,
    esm_model=None,
    esm_tokenizer=None,
    c_target: float = C_TARGET,
    device: str = "cpu",
    base_seed: int = 42,
    save_dir: Optional[str] = None,
    experiment_name: str = "latqd",
    strict_length: bool = True,
) -> Dict:
    """Run N independent experiments, return aggregated results."""
    from src.metrics import compute_all_metrics

    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)

    results = {
        "experiment": experiment_name,
        "model": mutator.model_name,
        "per_run_coverage": [],
        "per_run_curves": [],
        "per_run_metrics": [],
    }

    for run_idx in range(n_runs):
        seed = base_seed + run_idx
        random.seed(seed)
        np.random.seed(seed)

        print(f"\n{'='*60}\n[Run {run_idx+1}/{n_runs}] seed={seed}\n{'='*60}")

        archive, curve, all_gen = run_qd_search(
            mutator=mutator, seeds=seeds, n_rounds=n_rounds,
            esm_model=esm_model, esm_tokenizer=esm_tokenizer,
            c_target=c_target, device=device, strict_length=strict_length,
        )

        results["per_run_coverage"].append(archive.coverage_percent())
        results["per_run_curves"].append(curve)
        metrics = compute_all_metrics(archive, all_gen)
        results["per_run_metrics"].append(metrics)

        if save_dir:
            name = f"{experiment_name}_run{run_idx+1}"
            archive.save(os.path.join(save_dir, f"{name}_archive.json"))
            with open(os.path.join(save_dir, f"{name}_metrics.json"), "w") as f:
                json.dump(metrics, f, indent=2)

        print(f"  Coverage: {archive.coverage_percent():.2f}% | Metrics: {metrics}")

    covs = results["per_run_coverage"]
    results["coverage_mean"] = float(np.mean(covs))
    results["coverage_std"] = float(np.std(covs))

    print(f"\n[Summary] {experiment_name}: {results['coverage_mean']:.2f}% +/- {results['coverage_std']:.2f}%")

    if save_dir:
        with open(os.path.join(save_dir, f"{experiment_name}_summary.json"), "w") as f:
            json.dump(results, f, indent=2)

    return results
