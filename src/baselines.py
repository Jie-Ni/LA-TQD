"""
baselines.py -- Baseline algorithms for comparison.
"""
from __future__ import annotations

import random
from typing import List, Tuple

import numpy as np

from src.archive import MapElitesArchive
from src.config import ALPHABET, BLOSUM62, C_TARGET, N_ROUNDS
from src.oracle import (
    fitness, is_valid_basic, oracle_evaluate,
)


def _blosum62_weighted_residue(from_aa: str) -> str:
    row = BLOSUM62.get(from_aa, {})
    if not row:
        return random.choice(ALPHABET)
    aas = list(row.keys())
    scores = np.array([row[aa] for aa in aas], dtype=float)
    scores = scores - scores.min() + 1.0
    probs = scores / scores.sum()
    return np.random.choice(aas, p=probs)


def random_blosum62_mutation(seq: str, n_mutations: int = 1) -> str:
    seq_list = list(seq)
    rgd_pos = seq.find("RGD")
    protected = set(range(rgd_pos, rgd_pos + 3)) if rgd_pos >= 0 else set()
    mutable = [i for i in range(len(seq_list)) if i not in protected]
    for pos in random.sample(mutable, min(n_mutations, len(mutable))):
        seq_list[pos] = _blosum62_weighted_residue(seq_list[pos])
    return "".join(seq_list)


def run_ga_search(
    seeds: List[str],
    n_rounds: int = N_ROUNDS,
    esm_model=None,
    esm_tokenizer=None,
    c_target: float = C_TARGET,
    device: str = "cpu",
    use_oracle_priors: bool = True,
    verbose: bool = True,
) -> Tuple[MapElitesArchive, List[float], List[str]]:
    """Standard GA with BLOSUM62 mutations."""
    archive = MapElitesArchive()
    for seq in seeds:
        seq = seq.upper().strip()
        valid, score = oracle_evaluate(
            seq, esm_model, esm_tokenizer, c_target, device,
            enforce_priors=use_oracle_priors,
        )
        if valid:
            archive.add(seq, score, generation=0)

    coverage_curve = [archive.coverage_percent()]
    all_generated = list(seeds)

    for rnd in range(1, n_rounds + 1):
        parent = archive.sample()
        if parent is None:
            coverage_curve.append(archive.coverage_percent())
            continue

        child = random_blosum62_mutation(parent, n_mutations=random.randint(1, 3)).upper()
        all_generated.append(child)

        valid, score = oracle_evaluate(
            child, esm_model, esm_tokenizer, c_target, device,
            enforce_priors=use_oracle_priors,
        )
        accepted = archive.add(child, score, generation=rnd) if valid else False
        coverage_curve.append(archive.coverage_percent())

        if verbose:
            print(f"[GA {rnd:02d}] {'[OK]' if accepted else '[x]'} {child} cov={archive.coverage_percent():.1f}%")

    return archive, coverage_curve, all_generated


def run_vanilla_llm_search(
    mutator,
    seeds: List[str],
    n_rounds: int = N_ROUNDS,
    esm_model=None,
    esm_tokenizer=None,
    c_target: float = C_TARGET,
    device: str = "cpu",
    verbose: bool = True,
) -> Tuple[MapElitesArchive, List[float], List[str]]:
    """LLM search WITHOUT composition priors (vanilla baseline)."""
    archive = MapElitesArchive()
    for seq in seeds:
        seq = seq.upper().strip()
        if is_valid_basic(seq):
            score = fitness(seq, esm_model, esm_tokenizer, c_target, device, enforce_priors=False)
            if score != float("inf"):
                archive.add(seq, score, generation=0)

    coverage_curve = [archive.coverage_percent()]
    all_generated = list(seeds)

    for rnd in range(1, n_rounds + 1):
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

        accepted = False
        if is_valid_basic(child):
            score = fitness(child, esm_model, esm_tokenizer, c_target, device, enforce_priors=False)
            if score != float("inf"):
                accepted = archive.add(child, score, generation=rnd)

        coverage_curve.append(archive.coverage_percent())
        if verbose:
            print(f"[VanillaLLM {rnd:02d}] {'[OK]' if accepted else '[x]'} cov={archive.coverage_percent():.1f}%")

    return archive, coverage_curve, all_generated
