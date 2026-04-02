"""
biophysics.py — Biophysical property calculations for peptide sequences.
"""
from __future__ import annotations
import math
from collections import Counter
from typing import Sequence

from src.config import AROMATIC_RESIDUES, HACK_ENTROPY_THRESHOLD, HACK_RUN_LENGTH

# Kyte-Doolittle hydropathy scale
KD_SCALE: dict[str, float] = {
    "A":  1.8,  "R": -4.5,  "N": -3.5,  "D": -3.5,  "C":  2.5,
    "Q": -3.5,  "E": -3.5,  "G": -0.4,  "H": -3.2,  "I":  4.5,
    "L":  3.8,  "K": -3.9,  "M":  1.9,  "F":  2.8,  "P": -1.6,
    "S": -0.8,  "T": -0.7,  "W": -0.9,  "Y": -1.3,  "V":  4.2,
}

# Charge contributions at pH 7
_CHARGE: dict[str, float] = {
    "K":  1.0, "R":  1.0,
    "D": -1.0, "E": -1.0,
    "H":  0.1,
}


def net_charge(seq: str) -> float:
    return sum(_CHARGE.get(aa, 0.0) for aa in seq)


def mean_hydropathy(seq: str) -> float:
    if not seq:
        return 0.0
    return sum(KD_SCALE.get(aa, 0.0) for aa in seq) / len(seq)


def amino_acid_frequencies(seq: str) -> dict[str, float]:
    if not seq:
        return {}
    counts = Counter(seq)
    total = len(seq)
    return {aa: count / total for aa, count in counts.items()}


def shannon_entropy(seq: str) -> float:
    """Compute Shannon entropy of residue distribution (bits)."""
    if not seq:
        return 0.0
    freqs = amino_acid_frequencies(seq)
    return -sum(p * math.log2(p) for p in freqs.values() if p > 0)


def max_run_length(seq: str) -> int:
    """Length of the longest contiguous homopolymeric run."""
    if not seq:
        return 0
    max_run = curr_run = 1
    for i in range(1, len(seq)):
        curr_run = curr_run + 1 if seq[i] == seq[i - 1] else 1
        max_run = max(max_run, curr_run)
    return max_run


def has_long_repeat(seq: str, min_len: int = HACK_RUN_LENGTH) -> bool:
    return max_run_length(seq) >= min_len


def is_reward_hacked(seq: str) -> bool:
    """True if the sequence shows degeneracy: long repeat OR low Shannon entropy."""
    return has_long_repeat(seq) or shannon_entropy(seq) < HACK_ENTROPY_THRESHOLD


def max_single_aa_count(seq: str) -> int:
    if not seq:
        return 0
    return Counter(seq).most_common(1)[0][1]


def aromatic_count(seq: str) -> int:
    return sum(1 for aa in seq if aa in AROMATIC_RESIDUES)
