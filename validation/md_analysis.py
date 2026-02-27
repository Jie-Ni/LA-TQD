"""
md_analysis.py — Molecular Dynamics Trajectory Analysis.

Implements post-processing for GROMACS 100 ns MD simulations as described
in §4.4. Computes backbone RMSD and TM-score alignment to integrin ligands.

Requires:
  - MDTraj (pip install mdtraj) for RMSD and trajectory analysis
  - TM-score standalone binary (from Zhang lab) for TM-score computation
  - GROMACS output: .xtc trajectory + .gro topology files
"""
from __future__ import annotations

import os
import subprocess
import warnings
from typing import Dict, List, Optional, Tuple


# ─── RMSD Analysis ────────────────────────────────────────────────────────────

def compute_backbone_rmsd(
    trajectory_file: str,
    topology_file: str,
    selection: str = "backbone",
) -> Tuple[List[float], List[float]]:
    """Compute backbone RMSD over a GROMACS MD trajectory.

    As described in §4.4: RMSD stabilizes below 2.5 Å for elite candidates,
    indicating stable micro-folds over the 100 ns simulation.

    Args:
        trajectory_file: Path to GROMACS .xtc or .dcd trajectory file.
        topology_file: Path to topology (.gro, .pdb, or .psf) file.
        selection: MDTraj atom selection for RMSD computation (default: backbone).

    Returns:
        Tuple of (time_ns, rmsd_angstrom):
          - time_ns: List of time points in nanoseconds
          - rmsd_angstrom: Per-frame backbone RMSD in Ångströms
    """
    try:
        import mdtraj as md
        import numpy as np

        print(f"[MD] Loading trajectory: {trajectory_file}")
        traj = md.load(trajectory_file, top=topology_file)

        # Select backbone atoms
        backbone_idx = traj.topology.select(selection)
        if len(backbone_idx) == 0:
            warnings.warn("[MD] No backbone atoms found with selection. Using all atoms.")
            backbone_idx = traj.topology.select("protein")

        # Superpose to first frame and compute RMSD
        traj.superpose(traj, 0, atom_indices=backbone_idx)
        rmsd_nm = md.rmsd(traj, traj, 0, atom_indices=backbone_idx)

        # Convert: nm → Å, MDTraj time → ns
        rmsd_angstrom = (rmsd_nm * 10.0).tolist()
        # Estimate time from frame count (GROMACS default dt=2 ps, save every 1000 steps = 2 ns/frame)
        n_frames = traj.n_frames
        time_ns = [i * (100.0 / n_frames) for i in range(n_frames)]

        print(f"[MD] {n_frames} frames | Max RMSD: {max(rmsd_angstrom):.2f} Å | "
              f"Mean RMSD: {sum(rmsd_angstrom)/len(rmsd_angstrom):.2f} Å")

        return time_ns, rmsd_angstrom

    except ImportError:
        warnings.warn("[MD] MDTraj not installed. Run: pip install mdtraj")
        return [], []
    except Exception as e:
        warnings.warn(f"[MD] RMSD computation failed: {e}")
        return [], []


def check_rmsd_stability(
    rmsd_values: List[float],
    threshold_angstrom: float = 2.5,
    equilibration_fraction: float = 0.2,
) -> bool:
    """Check whether RMSD remains stably below threshold after equilibration.

    As reported in §4.4: backbone RMSD "swiftly plateaued and remained stably
    bounded below 2.5 Å" for elite candidates.

    Args:
        rmsd_values: Per-frame RMSD values in Ångströms.
        threshold_angstrom: Maximum acceptable RMSD after equilibration.
        equilibration_fraction: Fraction of trajectory to skip as equilibration.

    Returns:
        True if the post-equilibration RMSD is stable and below threshold.
    """
    if not rmsd_values:
        return False

    skip = int(len(rmsd_values) * equilibration_fraction)
    production = rmsd_values[skip:]

    if not production:
        return False

    max_rmsd = max(production)
    mean_rmsd = sum(production) / len(production)

    print(f"[MD Stability] Post-equil. RMSD: mean={mean_rmsd:.2f} Å, max={max_rmsd:.2f} Å")
    return max_rmsd <= threshold_angstrom


# ─── TM-Score ─────────────────────────────────────────────────────────────────

def compute_tm_score(
    mobile_pdb: str,
    reference_pdb: str,
    tmscore_binary: str = "TMscore",
) -> Optional[float]:
    """Compute TM-score between two PDB structures.

    Implements the metric from Zhang & Skolnick (2004), as used in §4.4 where
    TM-scores of 0.68–0.79 confirm high structural fidelity to native integrin
    ligands.

    Args:
        mobile_pdb: Path to query PDB structure.
        reference_pdb: Path to reference PDB structure.
        tmscore_binary: Path to TM-score executable (default: 'TMscore' on PATH).

    Returns:
        TM-score in [0, 1] (>0.5 indicates similar fold), or None on failure.
    """
    try:
        result = subprocess.run(
            [tmscore_binary, mobile_pdb, reference_pdb],
            capture_output=True, text=True, timeout=60
        )
        output = result.stdout

        # Parse TM-score line: "TM-score    = 0.7234 (d0= ..."
        import re
        match = re.search(r"TM-score\s*=\s*([\d.]+)", output)
        if match:
            tm = float(match.group(1))
            print(f"[TM-score] {os.path.basename(mobile_pdb)} vs reference: {tm:.4f}")
            return tm
        else:
            warnings.warn(f"[TM-score] Could not parse output:\n{output[:500]}")
            return None

    except FileNotFoundError:
        warnings.warn(
            f"[TM-score] '{tmscore_binary}' not found. "
            f"Download from: https://zhanggroup.org/TM-score/"
        )
        return None
    except Exception as e:
        warnings.warn(f"[TM-score] Error: {e}")
        return None


# ─── Full MD Validation Pipeline ─────────────────────────────────────────────

def validate_candidates_md(
    trajectory_files: List[str],
    topology_files: List[str],
    sequences: List[str],
    pdb_files: Optional[List[str]] = None,
    reference_pdb: Optional[str] = None,
    output_dir: str = "md_validation",
) -> List[Dict]:
    """Full MD validation: RMSD stability + TM-score for each candidate.

    Args:
        trajectory_files: List of .xtc trajectory files.
        topology_files: List of corresponding topology files.
        sequences: Sequence strings (for labeling).
        pdb_files: Predicted PDB files for TM-score comparison.
        reference_pdb: Reference integrin ligand PDB for TM-score.
        output_dir: Output directory for results.

    Returns:
        List of validation result dicts per candidate.
    """
    import json

    os.makedirs(output_dir, exist_ok=True)
    results = []

    for idx, (traj_f, top_f, seq) in enumerate(zip(trajectory_files, topology_files, sequences)):
        print(f"\n[MD Validation] {seq}")

        time_ns, rmsd = compute_backbone_rmsd(traj_f, top_f)
        stable = check_rmsd_stability(rmsd)

        tm = None
        if pdb_files and reference_pdb and idx < len(pdb_files):
            tm = compute_tm_score(pdb_files[idx], reference_pdb)

        entry = {
            "sequence": seq,
            "max_rmsd_angstrom": max(rmsd) if rmsd else None,
            "mean_rmsd_angstrom": sum(rmsd)/len(rmsd) if rmsd else None,
            "rmsd_stable": stable,
            "tm_score": tm,
        }
        results.append(entry)
        print(f"  → Stable: {stable} | TM-score: {tm}")

    # Save results
    out_path = os.path.join(output_dir, "md_validation_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[MD] Results saved to {out_path}")

    return results
