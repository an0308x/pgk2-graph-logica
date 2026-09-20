#!/usr/bin/env python3
"""Profile the released PGK2 data without converting assay evidence into arbitrary labels.

The selection release contains positive-side PGK2 counts and enrichment scores;
the NTC supplement contains control-side counts. This script records their
distributions and exact-SMILES overlap with the two unlabeled candidate panels.
It deliberately does not create binary labels: that requires a documented
thresholding/calibration decision.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, default=ROOT / "PGK2_selection.parquet")
    parser.add_argument(
        "--ntc-supplement", type=Path, default=ROOT / "PGK2_NTC_supplement.parquet"
    )
    parser.add_argument(
        "--candidate-panels",
        type=Path,
        default=ROOT / "DREAM_challenge_2026" / "Val-Test-set.zip",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "artifacts" / "pgk2_data_profile"
    )
    parser.add_argument(
        "--scaffold-sample-size",
        type=int,
        default=25_000,
        help="Structures sampled per panel for local scaffold profiling; use 0 for a full scan.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read_candidate_panels(path: Path) -> dict[str, list[dict[str, str]]]:
    members = {
        "validation": "Val-Test-set/PGK2_Validation_split.csv",
        "test": "Val-Test-set/PGK2_Test_split.csv",
    }
    panels: dict[str, list[dict[str, str]]] = {}
    with zipfile.ZipFile(path) as archive:
        for name, member in members.items():
            with archive.open(member) as handle:
                panels[name] = list(csv.DictReader(line.decode("utf-8") for line in handle))
    for name, rows in panels.items():
        if not rows or set(rows[0]) != {"CatalogID", "SMILES"}:
            raise ValueError(f"Unexpected {name} panel columns")
        if len({row["CatalogID"] for row in rows}) != len(rows):
            raise ValueError(f"Duplicate CatalogID in {name} panel")
        if len({row["SMILES"] for row in rows}) != len(rows):
            raise ValueError(f"Duplicate SMILES in {name} panel")
    return panels


def murcko_scaffold(smiles: str) -> str:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return ""
    scaffold = MurckoScaffold.GetScaffoldForMol(molecule)
    return Chem.MolToSmiles(scaffold, canonical=True) if scaffold.GetNumAtoms() else ""


def panel_scaffold_summary(
    panels: dict[str, list[dict[str, str]]], scaffold_sample_size: int
) -> dict[str, object]:
    if scaffold_sample_size < 0:
        raise ValueError("--scaffold-sample-size must be non-negative")
    result: dict[str, dict[str, object]] = {}
    scaffold_sets: dict[str, set[str]] = {}
    rng = np.random.default_rng(2026)
    for name, rows in panels.items():
        if scaffold_sample_size == 0 or scaffold_sample_size >= len(rows):
            selected = rows
        else:
            indices = rng.choice(len(rows), size=scaffold_sample_size, replace=False)
            selected = [rows[int(index)] for index in indices]
        scaffolds = [murcko_scaffold(row["SMILES"]) for row in selected]
        scaffold_sets[name] = set(filter(None, scaffolds))
        result[name] = {
            "scanned_rows": len(selected),
            "full_panel_scanned": len(selected) == len(rows),
            "unique_nonempty_scaffolds": len(scaffold_sets[name]),
            "acyclic_structures": int(sum(not scaffold for scaffold in scaffolds)),
        }
    return {
        "per_panel": result,
        "shared_nonempty_scaffolds": len(scaffold_sets["validation"] & scaffold_sets["test"]),
    }


def numeric_summary(state: dict[str, float], sample: np.ndarray) -> dict[str, float]:
    return {
        "n": int(state["n"]),
        "min": float(state["min"]),
        "p01_approx": float(np.quantile(sample, 0.01)),
        "median_approx": float(np.median(sample)),
        "p99_approx": float(np.quantile(sample, 0.99)),
        "max": float(state["max"]),
        "mean": float(state["sum"] / state["n"]),
        "quantile_sample_size": int(len(sample)),
    }


def profile_parquet(path: Path, panel_smiles: dict[str, set[str]]) -> dict[str, object]:
    parquet = pq.ParquetFile(path)
    columns = parquet.schema_arrow.names
    required = {"compound", "SMILES"}
    if not required.issubset(columns):
        raise ValueError(f"{path} is missing {sorted(required - set(columns))}")
    numeric_columns = [name for name in columns if name.startswith(("count_", "zscore_"))]
    numeric_state: dict[str, dict[str, float]] = {
        name: {"n": 0.0, "sum": 0.0, "min": np.inf, "max": -np.inf}
        for name in numeric_columns
    }
    numeric_samples: dict[str, list[np.ndarray]] = {name: [] for name in numeric_columns}
    rng = np.random.default_rng(2026)
    count_values: dict[str, Counter[int]] = {
        name: Counter() for name in numeric_columns if name.startswith("count_")
    }
    overlap = {name: set() for name in panel_smiles}
    rows = 0
    blank_smiles = 0
    for batch in parquet.iter_batches(batch_size=262_144, columns=["SMILES", *numeric_columns]):
        data = batch.to_pydict()
        rows += len(data["SMILES"])
        blank_smiles += sum(not value for value in data["SMILES"])
        smiles_batch = set(value for value in data["SMILES"] if value)
        for name, candidates in panel_smiles.items():
            overlap[name].update(smiles_batch & candidates)
        for name in numeric_columns:
            values = data[name]
            if any(value is None for value in values):
                raise ValueError(f"Missing numeric value in {path}: {name}")
            array = np.asarray(values, dtype=np.float64)
            state = numeric_state[name]
            state["n"] += len(array)
            state["sum"] += float(array.sum())
            state["min"] = min(state["min"], float(array.min()))
            state["max"] = max(state["max"], float(array.max()))
            sample_size = min(25_000, len(array))
            numeric_samples[name].append(
                array[rng.choice(len(array), size=sample_size, replace=False)]
            )
            if name in count_values:
                count_values[name].update(int(value) for value in values)
    return {
        "path": str(path),
        "sha256": sha256(path),
        "rows": rows,
        "columns": columns,
        "blank_smiles": blank_smiles,
        "numeric_summary": {
            name: numeric_summary(state, np.concatenate(numeric_samples[name]))
            for name, state in numeric_state.items()
        },
        "count_distributions": {
            name: {str(value): count for value, count in sorted(values.items())}
            for name, values in count_values.items()
        },
        "exact_smiles_overlap_with_candidate_panels": {
            name: {"count": len(values), "smiles": sorted(values)} for name, values in overlap.items()
        },
    }


def main() -> int:
    args = parse_args()
    started = time.perf_counter()
    for path in (args.selection, args.ntc_supplement, args.candidate_panels):
        if not path.exists():
            raise FileNotFoundError(path)
    panels = read_candidate_panels(args.candidate_panels)
    panel_smiles = {name: {row["SMILES"] for row in rows} for name, rows in panels.items()}
    scaffold_summary = panel_scaffold_summary(panels, args.scaffold_sample_size)
    selection = profile_parquet(args.selection, panel_smiles)
    ntc = profile_parquet(args.ntc_supplement, panel_smiles)
    report = {
        "target": "PGK2",
        "protein": {"uniprot": "P07205", "sequence_file": "data/proteins/PGK2_P07205.fasta"},
        "candidate_panels": {
            name: {
                "rows": len(rows),
                "unique_catalog_ids": len({row["CatalogID"] for row in rows}),
                "unique_smiles": len(panel_smiles[name]),
                "labelled": False,
            }
            for name, rows in panels.items()
        },
        "candidate_panel_exact_smiles_overlap": len(panel_smiles["validation"] & panel_smiles["test"]),
        "candidate_panel_scaffold_summary": scaffold_summary,
        "selection_release": selection,
        "ntc_supplement": ntc,
        "labeling_status": {
            "binary_labels_created": False,
            "reason": (
                "The releases provide count and z-score evidence, not a challenge-provided binary label. "
                "Thresholding, duplicate aggregation, and the control definition must be decided and recorded "
                "before supervised training."
            ),
        },
        "leakage_notes": [
            "Exact-SMILES matches between a candidate panel and released evidence are reported above and must be excluded from blind evaluation.",
            "Only one copy of each duplicate root/DREAM_challenge_2026 PGK2 Parquet should be used.",
        ],
        "elapsed_seconds": time.perf_counter() - started,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
