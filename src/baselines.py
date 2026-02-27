"""
baselines.py — Baseline algorithms for comparison.
"""
from __future__ import annotations

import random
from typing import List, Optional, Tuple

import numpy as np

from src.archive import MapElitesArchive
from src.config import ALPHABET, BLOSUM62, C_TARGET, N_ROUNDS
from src.oracle import fitness, check_length, check_alphabet, check_rgd_motif


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
    """Apply BLOSUM62-weighted substitution to n_mutations non-RGD positions."""
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
    """Standard GA with BLOSUM62 mutations. use_oracle_priors=False for random-mutation baseline."""
    from src.oracle import oracle_evaluate, check_composition_priors

    archive = MapElitesArchive()
    for seq in seeds:
        seq = seq.upper().strip()
        valid, score = oracle_evaluate(seq, esm_model, esm_tokenizer, c_target, device)
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

        if use_oracle_priors:
            valid, score = oracle_evaluate(child, esm_model, esm_tokenizer, c_target, device)
        else:
            basic = check_length(child) and check_alphabet(child) and check_rgd_motif(child)
            if basic:
                score = fitness(child, esm_model, esm_tokenizer, c_target, device)
                valid = score != float("inf")
            else:
                valid, score = False, float("inf")

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
        if check_length(seq) and check_alphabet(seq) and check_rgd_motif(seq):
            score = fitness(seq, esm_model, esm_tokenizer, c_target, device)
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
        if check_length(child) and check_alphabet(child) and check_rgd_motif(child):
            score = fitness(child, esm_model, esm_tokenizer, c_target, device)
            if score != float("inf"):
                accepted = archive.add(child, score, generation=rnd)

        coverage_curve.append(archive.coverage_percent())
        if verbose:
            print(f"[VanillaLLM {rnd:02d}] {'[OK]' if accepted else '[x]'} cov={archive.coverage_percent():.1f}%")

    return archive, coverage_curve, all_generated
