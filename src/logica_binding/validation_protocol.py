"""Shared, immutable scaffold assignment and conservative DEL pair weights."""
from __future__ import annotations

import hashlib
from functools import lru_cache

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

PROTOCOL = "pgk2_scaffold_v2_interface_v2_count_reliability_v1"


@lru_cache(maxsize=100_000)
def structure_identity(smiles: str) -> tuple[str, str]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles!r}")
    canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    # Stereoisomers must stay in the same split.
    group = Chem.MolToSmiles(scaffold, canonical=True, isomericSmiles=False)
    if not group:
        group = "acyclic:" + Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
    return canonical, group


def group_split(group: str) -> str:
    digest = hashlib.sha256(("pgk2-split-v2:" + group).encode()).digest()
    bucket = int.from_bytes(digest[:8], "little") % 1000
    return "train" if bucket < 800 else "dev" if bucket < 900 else "holdout"


def count_reliability(row: dict) -> float:
    """Heuristic evidence weight, NOT a binding probability or calibrated ratio.

    Missing inhibitor reads never add confidence. Positive target support and
    observed comparator control reads determine confidence. Depths are unknown;
    this does not attempt to reconstruct an uncensored assay from sparse counts.
    """
    target = float(row["positive_count_PGK2"])
    control = max(float(row["negative_count_inhibitor"]),
                  float(row["negative_count_NTC_evidence"]))
    if min(target, control) < 0:
        raise ValueError("Counts cannot be negative")
    return target / (target + 20.0) * control / (control + 5.0)


def validate_pair_splits(rows: list[dict]) -> None:
    for row in rows:
        a, ga = structure_identity(row["positive_smiles"])
        b, gb = structure_identity(row["negative_smiles"])
        if a == b or group_split(ga) != row["split"] or group_split(gb) != row["split"]:
            raise ValueError("Pair violates the shared molecule/scaffold split")
