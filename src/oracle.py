"""
oracle.py -- Sequence validity checks and fitness scoring.
"""
from __future__ import annotations

import math
import warnings
from typing import Optional, Tuple

from src.config import (
    ALPHABET, ALPHA, BETA, C_TARGET,
    MAX_AA_FREQ, MIN_AROMATIC, MOTIF, SEQ_LEN,
)
from src.biophysics import aromatic_count, max_single_aa_count, net_charge


# ─── Validity Checks ─────────────────────────────────────────────────────────

def check_length(seq: str, length: int = SEQ_LEN) -> bool:
    return len(seq) == length


def check_alphabet(seq: str) -> bool:
    return all(aa in ALPHABET for aa in seq)


def check_rgd_motif(seq: str, motif: str = MOTIF) -> bool:
    return motif in seq


def check_composition_priors(seq: str) -> bool:
    """Two composition guards (the deterministic oracle):
    1. No amino acid appears more than MAX_AA_FREQ times.
    2. At least MIN_AROMATIC aromatic residue (F, W, Y) present.
    """
    return (
        max_single_aa_count(seq) <= MAX_AA_FREQ
        and aromatic_count(seq) >= MIN_AROMATIC
    )


def is_valid_basic(seq: str, length: int = SEQ_LEN) -> bool:
    """Basic structural validity: length + alphabet + RGD motif.
    Does NOT enforce composition priors.
    """
    seq = seq.upper().strip()
    if not check_length(seq, length):
        return False
    if not check_alphabet(seq):
        return False
    if not check_rgd_motif(seq):
        return False
    return True


def is_valid(seq: str, enforce_priors: bool = True, length: int = SEQ_LEN) -> bool:
    """Full validity check including optional composition priors."""
    seq = seq.upper().strip()
    if not is_valid_basic(seq, length):
        return False
    if enforce_priors and not check_composition_priors(seq):
        return False
    return True


# ─── ESM-2 Pseudo-Perplexity ─────────────────────────────────────────────────

def esm2_perplexity(seq: str, model, tokenizer, device: str = "cpu") -> float:
    """Masked pseudo-log-likelihood (negative mean log-prob per position).
    Lower values indicate more natural sequences.
    """
    import torch

    tokens = tokenizer(seq, return_tensors="pt").to(device)
    input_ids = tokens["input_ids"]
    aa_positions = list(range(1, input_ids.shape[1] - 1))

    total_log_prob = 0.0
    model.eval()
    with torch.no_grad():
        for pos in aa_positions:
            masked = input_ids.clone()
            masked[0, pos] = tokenizer.mask_token_id
            logits = model(masked).logits
            log_probs = torch.log_softmax(logits[0, pos], dim=-1)
            total_log_prob += log_probs[input_ids[0, pos].item()].item()

    L = len(aa_positions)
    return -total_log_prob / L if L > 0 else float("inf")


# ─── Fitness Function ─────────────────────────────────────────────────────────

def fitness(
    seq: str,
    esm_model=None,
    esm_tokenizer=None,
    c_target: float = C_TARGET,
    device: str = "cpu",
    alpha: float = ALPHA,
    beta: float = BETA,
    enforce_priors: bool = True,
    length: int = SEQ_LEN,
) -> float:
    """E(S) = alpha * pplx(S) + beta * |charge(S) - c_target|

    Returns +inf for structurally invalid sequences.
    ``enforce_priors`` controls whether composition priors are checked.
    """
    if not is_valid(seq, enforce_priors=enforce_priors, length=length):
        return float("inf")

    charge_penalty = abs(net_charge(seq) - c_target)

    if esm_model is not None and esm_tokenizer is not None:
        try:
            pplx = esm2_perplexity(seq, esm_model, esm_tokenizer, device)
        except Exception as e:
            warnings.warn(f"ESM-2 failed for '{seq}': {e}")
            return float("inf")
        return alpha * pplx + beta * charge_penalty
    else:
        return beta * charge_penalty


def oracle_evaluate(
    seq: str,
    esm_model=None,
    esm_tokenizer=None,
    c_target: float = C_TARGET,
    device: str = "cpu",
    enforce_priors: bool = True,
) -> Tuple[bool, float]:
    """Evaluate a candidate: returns (is_valid, fitness_score)."""
    seq = seq.upper().strip()
    valid = is_valid(seq, enforce_priors=enforce_priors)
    if not valid:
        return False, float("inf")
    score = fitness(
        seq, esm_model, esm_tokenizer, c_target, device,
        enforce_priors=enforce_priors,
    )
    return True, score
