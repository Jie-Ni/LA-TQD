"""
metrics.py — Evaluation metrics.
"""
from __future__ import annotations

from typing import Dict, List, Optional
from src.biophysics import is_reward_hacked


def levenshtein_distance(s1: str, s2: str) -> int:
    m, n = len(s1), len(s2)
    if m == 0: return n
    if n == 0: return m
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            temp = dp[j]
            dp[j] = prev if s1[i-1] == s2[j-1] else 1 + min(prev, dp[j], dp[j-1])
            prev = temp
    return dp[n]


def avg_pairwise_levenshtein(sequences: List[str]) -> float:
    """Average edit distance over all unique pairs."""
    n = len(sequences)
    if n < 2:
        return 0.0
    total = sum(
        levenshtein_distance(sequences[i], sequences[j])
        for i in range(n) for j in range(i + 1, n)
    )
    return total / (n * (n - 1) / 2)


def hacking_rate(sequences: List[str]) -> float:
    """Percentage of sequences that are degenerate (low entropy or long repeat)."""
    if not sequences:
        return 0.0
    return 100.0 * sum(1 for s in sequences if is_reward_hacked(s)) / len(sequences)


def coverage_percent(archive) -> float:
    return archive.coverage_percent()


def avg_plddt(plddt_scores: List[float]) -> float:
    return sum(plddt_scores) / len(plddt_scores) if plddt_scores else 0.0


def compute_all_metrics(archive, all_generated: List[str],
                         plddt_scores: Optional[List[float]] = None) -> Dict:
    elites = archive.all_sequences()
    return {
        "coverage_pct": archive.coverage_percent(),
        "hacking_rate_pct": hacking_rate(elites),
        "hacking_rate_all_pct": hacking_rate(all_generated),
        "avg_levenshtein": avg_pairwise_levenshtein(elites),
        "n_elites": len(elites),
        "n_total_generated": len(all_generated),
        "avg_plddt": avg_plddt(plddt_scores) if plddt_scores else 0.0,
    }
