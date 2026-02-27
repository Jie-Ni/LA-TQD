"""
esm_scorer.py — ESM-2 model loader and batch scorer (singleton).
"""
from __future__ import annotations

import warnings
from typing import List, Optional, Tuple

from src.config import ESM2_MODEL_NAME

_esm_model = None
_esm_tokenizer = None


def load_esm2(model_name: str = ESM2_MODEL_NAME, device: str = "auto") -> Tuple:
    """Load ESM-2 masked LM (cached after first call)."""
    global _esm_model, _esm_tokenizer
    if _esm_model is not None:
        return _esm_model, _esm_tokenizer
    try:
        from transformers import EsmForMaskedLM, EsmTokenizer
        import torch

        print(f"[ESM2] Loading: {model_name}")
        tokenizer = EsmTokenizer.from_pretrained(model_name)
        model = EsmForMaskedLM.from_pretrained(model_name)
        model.eval()
        dev = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
        model = model.to(dev)
        print(f"[ESM2] Ready on {dev}")
        _esm_model, _esm_tokenizer = model, tokenizer
        return model, tokenizer
    except Exception as e:
        warnings.warn(f"[ESM2] Could not load: {e}. Fitness will use charge-only fallback.")
        return None, None


def batch_pseudo_perplexity(
    sequences: List[str], model=None, tokenizer=None,
    device: str = "cpu", auto_load: bool = True,
) -> List[float]:
    from src.oracle import esm2_perplexity
    if model is None and auto_load:
        model, tokenizer = load_esm2(device=device)
    results = []
    for seq in sequences:
        if model is None:
            results.append(float("inf"))
            continue
        try:
            results.append(esm2_perplexity(seq, model, tokenizer, device))
        except Exception as e:
            warnings.warn(f"[ESM2] {seq}: {e}")
            results.append(float("inf"))
    return results
