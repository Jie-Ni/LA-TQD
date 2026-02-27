"""
archive.py — MAP-Elites 10x10 behavioral grid archive.

Behavioral axes:
  - Axis 0 (rows): net charge
  - Axis 1 (cols): mean Kyte-Doolittle hydropathy
"""
from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from src.config import (
    CHARGE_MAX, CHARGE_MIN, GRID_SIZE,
    HYDROPATHY_MAX, HYDROPATHY_MIN,
)
from src.biophysics import net_charge, mean_hydropathy


@dataclass
class Elite:
    sequence: str
    fitness: float
    charge: float
    hydropathy: float
    generation: int = 0


class MapElitesArchive:
    """10x10 MAP-Elites grid keyed by (charge, hydropathy) behavioral descriptors."""

    def __init__(
        self,
        grid_size: int = GRID_SIZE,
        charge_bounds: Tuple[float, float] = (CHARGE_MIN, CHARGE_MAX),
        hydropathy_bounds: Tuple[float, float] = (HYDROPATHY_MIN, HYDROPATHY_MAX),
    ):
        self.grid_size = grid_size
        self.charge_min, self.charge_max = charge_bounds
        self.hydropathy_min, self.hydropathy_max = hydropathy_bounds

        self.charge_bins = np.linspace(self.charge_min, self.charge_max, grid_size + 1)
        self.hydropathy_bins = np.linspace(self.hydropathy_min, self.hydropathy_max, grid_size + 1)

        self._grid: List[List[Optional[Elite]]] = [
            [None] * grid_size for _ in range(grid_size)
        ]
        self._n_filled: int = 0

    def _bin_index(self, value: float, bins: np.ndarray) -> int:
        idx = int(np.searchsorted(bins, value, side="right")) - 1
        return max(0, min(idx, self.grid_size - 1))

    def behavioral_index(self, seq: str) -> Tuple[int, int]:
        i = self._bin_index(net_charge(seq), self.charge_bins)
        j = self._bin_index(mean_hydropathy(seq), self.hydropathy_bins)
        return i, j

    def add(self, seq: str, fitness_val: float, generation: int = 0) -> bool:
        if math.isinf(fitness_val) or math.isnan(fitness_val):
            return False
        i, j = self.behavioral_index(seq)
        current = self._grid[i][j]
        if current is None or fitness_val < current.fitness:
            was_empty = current is None
            self._grid[i][j] = Elite(
                sequence=seq,
                fitness=fitness_val,
                charge=net_charge(seq),
                hydropathy=mean_hydropathy(seq),
                generation=generation,
            )
            if was_empty:
                self._n_filled += 1
            return True
        return False

    def sample(self) -> Optional[str]:
        occupied = [
            self._grid[i][j]
            for i in range(self.grid_size)
            for j in range(self.grid_size)
            if self._grid[i][j] is not None
        ]
        if not occupied:
            return None
        return random.choice(occupied).sequence

    def sample_with_target(self) -> Tuple[Optional[str], str]:
        parent = self.sample()
        if parent is None:
            return None, ""
        directions = [
            "increase hydrophobicity and introduce an aromatic anchor",
            "decrease net charge and increase hydrophobicity",
            "increase net charge and maintain hydropathy",
            "decrease hydrophobicity and add charged residues",
            "move to a more cationic, hydrophilic region",
            "move to a more anionic, hydrophobic region",
            "maximize sequence diversity while preserving the RGD motif",
            "add bulky aromatic residues to improve structural stability",
        ]
        return parent, random.choice(directions)

    def coverage(self) -> float:
        return self._n_filled / (self.grid_size * self.grid_size)

    def coverage_percent(self) -> float:
        return self.coverage() * 100.0

    @property
    def n_filled(self) -> int:
        return self._n_filled

    def all_elites(self) -> List[Elite]:
        return [
            self._grid[i][j]
            for i in range(self.grid_size)
            for j in range(self.grid_size)
            if self._grid[i][j] is not None
        ]

    def all_sequences(self) -> List[str]:
        return [e.sequence for e in self.all_elites()]

    def best_elite(self) -> Optional[Elite]:
        elites = self.all_elites()
        return min(elites, key=lambda e: e.fitness) if elites else None

    def to_dict(self) -> dict:
        grid_data = {}
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                e = self._grid[i][j]
                if e is not None:
                    grid_data[f"{i},{j}"] = {
                        "sequence": e.sequence, "fitness": e.fitness,
                        "charge": e.charge, "hydropathy": e.hydropathy,
                        "generation": e.generation,
                    }
        return {
            "grid_size": self.grid_size,
            "charge_bounds": [self.charge_min, self.charge_max],
            "hydropathy_bounds": [self.hydropathy_min, self.hydropathy_max],
            "n_filled": self._n_filled,
            "coverage_percent": self.coverage_percent(),
            "grid": grid_data,
        }

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "MapElitesArchive":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        archive = cls(
            grid_size=data["grid_size"],
            charge_bounds=tuple(data["charge_bounds"]),
            hydropathy_bounds=tuple(data["hydropathy_bounds"]),
        )
        for key, entry in data["grid"].items():
            i, j = map(int, key.split(","))
            archive._grid[i][j] = Elite(
                sequence=entry["sequence"], fitness=entry["fitness"],
                charge=entry["charge"], hydropathy=entry["hydropathy"],
                generation=entry.get("generation", 0),
            )
            archive._n_filled += 1
        return archive

    def __repr__(self) -> str:
        return (
            f"MapElitesArchive(grid={self.grid_size}x{self.grid_size}, "
            f"filled={self._n_filled}, coverage={self.coverage_percent():.1f}%)"
        )
