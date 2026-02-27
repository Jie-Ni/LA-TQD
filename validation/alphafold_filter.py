"""
alphafold_filter.py — AlphaFold-based structural confidence filter.

Uses ESMFold (via the HuggingFace transformers library) as a rapid,
locally-runnable structure predictor to estimate AlphaFold-level pLDDT
confidence scores. As described in §3 and Appendix A:
  - AlphaFold2 v2.3.0 with Amber relaxation used for offline validation
  - ESMFold used as rapid online proxy within the evolutionary loop

Requires: transformers >= 4.31, torch
"""
from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple

PLDDT_THRESHOLD = 70.0  # Minimum acceptable pLDDT for structural confidence


def load_esmfold(device: str = "auto"):
    """Load ESMFold model for rapid structure prediction.

    ESMFold provides AlphaFold-quality predictions ~60× faster, making it
    suitable for integration within the evolutionary loop as described in §3.

    Args:
        device: PyTorch device ('auto', 'cpu', 'cuda', etc.).

    Returns:
        Tuple of (model, tokenizer), or (None, None) on import failure.
    """
    try:
        from transformers import EsmForProteinFolding, AutoTokenizer
        import torch

        print("[ESMFold] Loading model: facebook/esmfold_v1")
        tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
        model = EsmForProteinFolding.from_pretrained(
            "facebook/esmfold_v1",
            low_cpu_mem_usage=True,
        )
        model.eval()

        if device == "auto":
            dev = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            dev = device

        model = model.to(dev)
        print(f"[ESMFold] Model loaded on {dev}")
        return model, tokenizer

    except Exception as e:
        warnings.warn(f"[ESMFold] Could not load: {e}. Skipping structure prediction.")
        return None, None


def predict_structure(
    seq: str,
    model=None,
    tokenizer=None,
    save_pdb: Optional[str] = None,
) -> Dict:
    """Predict protein structure and extract pLDDT confidence score.

    Args:
        seq: Amino acid sequence.
        model: ESMFold model instance.
        tokenizer: ESMFold tokenizer.
        save_pdb: If provided, save predicted PDB to this path.

    Returns:
        Dict with keys:
          - "plddt": mean pLDDT score (0–100)
          - "plddt_per_residue": list of per-residue pLDDT
          - "pdb_string": PDB file content (if model loaded)
    """
    if model is None:
        return {"plddt": 0.0, "plddt_per_residue": [], "pdb_string": ""}

    import torch

    with torch.no_grad():
        tokenized = tokenizer(
            seq,
            return_tensors="pt",
            add_special_tokens=False,
        ).to(model.device)
        outputs = model(**tokenized)

    # Extract pLDDT from output
    plddt_per_residue = outputs.plddt[0].cpu().numpy().tolist()
    mean_plddt = float(sum(plddt_per_residue) / len(plddt_per_residue)) if plddt_per_residue else 0.0

    # Convert to PDB string
    pdb_string = ""
    try:
        from transformers.models.esm.openfold_utils.protein import to_pdb, Protein
        import numpy as np

        final_atom_positions = outputs.positions[-1, 0].cpu().numpy()
        final_atom_mask = outputs.atom37_atom_exists[0].cpu().numpy()
        pred_b_factors = outputs.plddt[0].cpu().numpy()

        protein_obj = Protein(
            aatype=tokenized["input_ids"][0].cpu().numpy(),
            atom_positions=final_atom_positions,
            atom_mask=final_atom_mask,
            residue_index=np.arange(len(seq)),
            b_factors=np.repeat(pred_b_factors[:, None], 37, axis=-1),
            chain_index=np.zeros(len(seq), dtype=int),
        )
        pdb_string = to_pdb(protein_obj)
    except Exception as e:
        warnings.warn(f"[ESMFold] PDB conversion failed: {e}")

    if save_pdb and pdb_string:
        with open(save_pdb, "w") as f:
            f.write(pdb_string)

    return {
        "plddt": mean_plddt,
        "plddt_per_residue": plddt_per_residue,
        "pdb_string": pdb_string,
    }


def filter_by_plddt(
    sequences: List[str],
    model=None,
    tokenizer=None,
    threshold: float = PLDDT_THRESHOLD,
    output_dir: Optional[str] = None,
) -> List[Tuple[str, float]]:
    """Filter sequences by AlphaFold pLDDT confidence.

    Returns only sequences with mean pLDDT ≥ threshold, along with
    their scores. Optionally saves PDB files to output_dir.

    Args:
        sequences: List of candidate sequences.
        model: ESMFold model.
        tokenizer: ESMFold tokenizer.
        threshold: Minimum pLDDT (default 70.0).
        output_dir: If provided, save PDB files here.

    Returns:
        List of (sequence, plddt) tuples for accepted candidates.
    """
    import os

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    accepted = []
    for idx, seq in enumerate(sequences):
        pdb_path = None
        if output_dir:
            pdb_path = os.path.join(output_dir, f"candidate_{idx:04d}_{seq}.pdb")

        result = predict_structure(seq, model, tokenizer, save_pdb=pdb_path)
        plddt = result["plddt"]

        print(f"[pLDDT] {seq}: {plddt:.1f} {'✓' if plddt >= threshold else '✗'}")

        if plddt >= threshold:
            accepted.append((seq, plddt))

    print(f"\n[pLDDT Filter] Accepted {len(accepted)}/{len(sequences)} sequences "
          f"(threshold={threshold})")
    return accepted
