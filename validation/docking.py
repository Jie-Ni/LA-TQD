"""
docking.py — AutoDock Vina Molecular Docking Wrapper.

Implements the docking pipeline described in §4.4:
  - AutoDock Vina v1.2.5 (Eberhardt et al., 2021)
  - Target: αvβ3 integrin receptor (PDB: 1L5G)
  - Docking scores in kcal/mol (paper results: −7.20 to −8.40)

Requires: AutoDock Vina 1.2.0+ installed and on PATH, or via Python bindings.
"""
from __future__ import annotations

import os
import subprocess
import tempfile
import warnings
from typing import Dict, List, Optional, Tuple


# ─── Vina Configuration ───────────────────────────────────────────────────────

# Default search box centered on αvβ3 integrin RGD binding site
# (PDB 1L5G, approximate centroid of the GRGDSP binding pocket)
DEFAULT_BOX_CENTER = {"x": 15.0, "y": 22.0, "z": 18.0}
DEFAULT_BOX_SIZE   = {"x": 20.0, "y": 20.0, "z": 20.0}
DEFAULT_EXHAUSTIVENESS = 8
DEFAULT_NUM_MODES = 9


def parse_vina_output(vina_log: str) -> Optional[float]:
    """Parse AutoDock Vina output log to extract the best docking score.

    Args:
        vina_log: String output from Vina (stdout or log file contents).

    Returns:
        Best docking score in kcal/mol (most negative = strongest binding),
        or None if parsing fails.
    """
    import re

    # Vina output table format:
    #   mode | affinity (kcal/mol) | dist from best mode (RMSD)
    #      1 |       -8.4          |   0.000  |   0.000
    pattern = re.compile(
        r"^\s*1\s+\|\s*([-+]?\d+\.?\d*)\s+\|", re.MULTILINE
    )
    match = pattern.search(vina_log)
    if match:
        return float(match.group(1))

    # Alternative: look for first line starting with "1" in a numbered table
    for line in vina_log.splitlines():
        parts = line.strip().split()
        if len(parts) >= 2 and parts[0] == "1":
            try:
                return float(parts[1])
            except ValueError:
                continue

    return None


def run_autodock_vina(
    ligand_pdbqt: str,
    receptor_pdbqt: str,
    output_dir: str,
    box_center: Optional[Dict[str, float]] = None,
    box_size: Optional[Dict[str, float]] = None,
    exhaustiveness: int = DEFAULT_EXHAUSTIVENESS,
    num_modes: int = DEFAULT_NUM_MODES,
    seed: int = 42,
    vina_executable: str = "vina",
) -> Tuple[Optional[float], str]:
    """Run AutoDock Vina docking for a single ligand-receptor pair.

    Args:
        ligand_pdbqt: Path to ligand PDBQT file.
        receptor_pdbqt: Path to receptor PDBQT file.
        output_dir: Directory for Vina output files.
        box_center: Dict with x, y, z center coordinates (Å).
        box_size: Dict with x, y, z search box dimensions (Å).
        exhaustiveness: Search exhaustiveness (default 8).
        num_modes: Maximum number of binding modes to output (default 9).
        seed: Random seed for reproducibility.
        vina_executable: Path to vina binary (default: 'vina' on PATH).

    Returns:
        Tuple of (best_score, output_log_string).
    """
    os.makedirs(output_dir, exist_ok=True)

    box_center = box_center or DEFAULT_BOX_CENTER
    box_size = box_size or DEFAULT_BOX_SIZE

    output_pdbqt = os.path.join(output_dir, "docked_out.pdbqt")
    log_file = os.path.join(output_dir, "vina_log.txt")

    cmd = [
        vina_executable,
        "--receptor", receptor_pdbqt,
        "--ligand",   ligand_pdbqt,
        "--out",      output_pdbqt,
        "--log",      log_file,
        "--center_x", str(box_center["x"]),
        "--center_y", str(box_center["y"]),
        "--center_z", str(box_center["z"]),
        "--size_x",   str(box_size["x"]),
        "--size_y",   str(box_size["y"]),
        "--size_z",   str(box_size["z"]),
        "--exhaustiveness", str(exhaustiveness),
        "--num_modes",      str(num_modes),
        "--seed",           str(seed),
    ]

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=300
        )
        log_text = result.stdout + result.stderr

        if result.returncode != 0:
            warnings.warn(f"[Vina] Non-zero exit code: {result.returncode}\n{log_text}")

        # Also read the written log file if it exists
        if os.path.exists(log_file):
            with open(log_file, "r") as f:
                log_text += f.read()

        best_score = parse_vina_output(log_text)
        return best_score, log_text

    except FileNotFoundError:
        warnings.warn(
            f"[Vina] '{vina_executable}' not found on PATH. "
            f"Install AutoDock Vina 1.2.0+ to enable docking."
        )
        return None, ""
    except subprocess.TimeoutExpired:
        warnings.warn("[Vina] Docking timed out after 300s.")
        return None, ""
    except Exception as e:
        warnings.warn(f"[Vina] Unexpected error: {e}")
        return None, ""


def batch_dock(
    sequences: List[str],
    receptor_pdbqt: str,
    output_dir: str,
    pdb_dir: Optional[str] = None,
    esm_model=None,
    esm_tokenizer=None,
) -> List[Dict]:
    """Batch docking pipeline: structure prediction → PDBQT conversion → Vina.

    Full pipeline as described in §4.4:
    1. Predict structure with ESMFold (or load existing PDB)
    2. Convert PDB to PDBQT (requires Open Babel or MGLTools)
    3. Dock against αvβ3 integrin receptor with AutoDock Vina

    Args:
        sequences: List of elite candidate sequences.
        receptor_pdbqt: Path to pre-prepared receptor PDBQT file.
        output_dir: Root output directory.
        pdb_dir: Directory of pre-predicted PDB files (optional).
        esm_model: ESMFold model for structure prediction.
        esm_tokenizer: ESMFold tokenizer.

    Returns:
        List of dicts: {sequence, plddt, docking_score, pdb_path}.
    """
    from validation.alphafold_filter import predict_structure

    results = []
    for idx, seq in enumerate(sequences):
        seq_dir = os.path.join(output_dir, f"seq_{idx:04d}_{seq}")
        os.makedirs(seq_dir, exist_ok=True)

        # Step 1: Structure prediction
        pdb_path = os.path.join(seq_dir, f"{seq}.pdb")
        if pdb_dir and os.path.exists(os.path.join(pdb_dir, f"{seq}.pdb")):
            pdb_path = os.path.join(pdb_dir, f"{seq}.pdb")
            plddt = 0.0  # Not recomputed from existing PDB
        else:
            struct = predict_structure(seq, esm_model, esm_tokenizer, save_pdb=pdb_path)
            plddt = struct["plddt"]

        # Step 2: PDB → PDBQT conversion via Open Babel (if available)
        pdbqt_path = pdb_path.replace(".pdb", ".pdbqt")
        try:
            subprocess.run(
                ["obabel", pdb_path, "-O", pdbqt_path, "--partialcharge", "gasteiger"],
                capture_output=True, timeout=60
            )
        except Exception:
            warnings.warn(f"[OpenBabel] Could not convert {pdb_path} to PDBQT.")

        # Step 3: Docking
        docking_score = None
        if os.path.exists(pdbqt_path) and os.path.exists(receptor_pdbqt):
            docking_score, _ = run_autodock_vina(
                ligand_pdbqt=pdbqt_path,
                receptor_pdbqt=receptor_pdbqt,
                output_dir=os.path.join(seq_dir, "vina_out"),
                seed=42 + idx,
            )

        results.append({
            "sequence": seq,
            "plddt": plddt,
            "docking_score": docking_score,
            "pdb_path": pdb_path,
        })
        print(f"[Docking] {seq}: pLDDT={plddt:.1f}, score={docking_score} kcal/mol")

    return results
