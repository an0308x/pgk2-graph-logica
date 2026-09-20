from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


@dataclass(frozen=True)
class Compound:
    compound_id: str
    smiles: str
    label: int | None
    mw: float | None = None
    alogp: float | None = None


def read_fasta(path: Path) -> str:
    lines = path.read_text(encoding="utf-8").splitlines()
    sequence = "".join(line.strip().replace(" ", "") for line in lines if not line.startswith(">"))
    invalid = set(sequence) - set("ACDEFGHIKLMNPQRSTVWYBXZJUO")
    if not sequence or invalid:
        raise ValueError(f"Invalid FASTA sequence in {path}: {sorted(invalid)}")
    return sequence


def load_matched_evaluation(
    path: Path,
    max_positive_pairs: int | None = None,
    seed: int = 2026,
) -> list[Compound]:
    """Return all (or a capped number of) positives and unique matched negatives.

    Matching is nearest-neighbour in standardized molecular weight and ALOGP.
    This keeps the smoke benchmark from succeeding on trivial bulk-property
    differences alone.
    """

    table = pq.read_table(path, columns=["SMILES", "RandomID", "LABEL", "MW", "ALOGP"])
    labels = table.column("LABEL").to_numpy(zero_copy_only=False)
    smiles = table.column("SMILES").to_pylist()
    ids = table.column("RandomID").to_pylist()
    mw = table.column("MW").to_numpy(zero_copy_only=False).astype(np.float64)
    alogp = table.column("ALOGP").to_numpy(zero_copy_only=False).astype(np.float64)

    positive_idx = np.flatnonzero(labels == 1)
    negative_idx = np.flatnonzero(labels == 0)
    rng = np.random.default_rng(seed)
    rng.shuffle(positive_idx)
    if max_positive_pairs is not None:
        positive_idx = positive_idx[:max_positive_pairs]

    features = np.column_stack([mw, alogp])
    center = np.nanmean(features[negative_idx], axis=0)
    scale = np.nanstd(features[negative_idx], axis=0)
    scale[scale == 0] = 1.0
    normalized = (features - center) / scale

    available = np.ones(len(negative_idx), dtype=bool)
    matched_negative_idx: list[int] = []
    negative_features = normalized[negative_idx]
    for idx in positive_idx:
        distance = np.sum((negative_features - normalized[idx]) ** 2, axis=1)
        distance[~available] = np.inf
        chosen_position = int(np.argmin(distance))
        available[chosen_position] = False
        matched_negative_idx.append(int(negative_idx[chosen_position]))

    selected = np.concatenate([positive_idx, np.asarray(matched_negative_idx, dtype=np.int64)])
    rng.shuffle(selected)
    return [
        Compound(
            compound_id=str(ids[i]),
            smiles=str(smiles[i]),
            label=int(labels[i]),
            mw=float(mw[i]),
            alogp=float(alogp[i]),
        )
        for i in selected
    ]


def load_test_compounds(path: Path) -> list[Compound]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"Ccompound_ID", "SMILES"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Expected columns {sorted(required)} in {path}")
    return [
        Compound(compound_id=row["Ccompound_ID"], smiles=row["SMILES"], label=None)
        for row in rows
    ]

