#!/usr/bin/env python3
"""Train and score a Morgan-fingerprint PGK2 matched-SAR baseline.

This baseline intentionally has no protein representation.  PGK2 is fixed in
this challenge, so its purpose is to test whether the matched DEL pairs alone
contain a transferable small-molecule signal before attributing failure or
success to LogiCA's protein--compound architecture.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
import zipfile
from pathlib import Path

import numpy as np
import polars as pl
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
FP_SIZE = 2048


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, default=ROOT / "artifacts" / "pgk2_matched_sar_pairs" / "pgk2_matched_sar_pairs.parquet")
    parser.add_argument("--candidate-panels", type=Path, default=ROOT / "DREAM_challenge_2026" / "Val-Test-set.zip")
    parser.add_argument("--panel", choices=("validation", "test"), default="validation")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts" / "pgk2_fingerprint_baseline")
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-iter", type=int, default=400)
    parser.add_argument("--max-panel-rows", type=int, default=None, help="Bounded scoring smoke test only.")
    return parser.parse_args()


def fingerprint(smiles: str) -> np.ndarray:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"Invalid SMILES: {smiles!r}")
    bit_vector = AllChem.GetMorganGenerator(radius=2, fpSize=FP_SIZE).GetFingerprint(molecule)
    result = np.zeros((FP_SIZE,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(bit_vector, result)
    return result


def matrix(smiles: list[str]) -> np.ndarray:
    return np.vstack([fingerprint(value) for value in smiles])


def murcko_group(smiles: str) -> str:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return smiles
    scaffold = MurckoScaffold.GetScaffoldForMol(molecule)
    return Chem.MolToSmiles(scaffold, canonical=True) if scaffold.GetNumAtoms() else smiles


def group_split(groups: np.ndarray, fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    unique = np.unique(groups)
    if unique.size < 2:
        raise ValueError("Need at least two positive scaffold groups")
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    heldout_groups = set(unique[: max(1, min(unique.size - 1, round(unique.size * fraction)))])
    heldout = np.flatnonzero(np.isin(groups, list(heldout_groups)))
    return np.flatnonzero(~np.isin(groups, list(heldout_groups))), heldout


def read_panel(path: Path, panel: str) -> list[dict[str, str]]:
    member = {
        "validation": "Val-Test-set/PGK2_Validation_split.csv",
        "test": "Val-Test-set/PGK2_Test_split.csv",
    }[panel]
    with zipfile.ZipFile(path) as archive, archive.open(member) as handle:
        return list(csv.DictReader(line.decode("utf-8") for line in handle))


def main() -> int:
    args = parse_args()
    if not args.pairs.exists() or not args.candidate_panels.exists():
        raise FileNotFoundError("Pair data or candidate-panel archive is missing")
    if not 0 < args.test_fraction < 1:
        raise ValueError("test-fraction must be in (0, 1)")
    started = time.perf_counter()
    pairs = pl.read_parquet(args.pairs)
    needed = {"positive_smiles", "negative_smiles", "pair_weight"}
    if not needed.issubset(pairs.columns):
        raise ValueError(f"Missing pair columns: {sorted(needed - set(pairs.columns))}")
    rows = pairs.to_dicts()
    print(json.dumps({"stage": "loaded_pairs", "pairs": len(rows)}), flush=True)
    groups = np.asarray([murcko_group(str(row["positive_smiles"])) for row in rows])
    train_idx, heldout_idx = group_split(groups, args.test_fraction, args.seed)

    def fit(indices: np.ndarray) -> LogisticRegression:
        smiles = [str(rows[index]["positive_smiles"]) for index in indices] + [str(rows[index]["negative_smiles"]) for index in indices]
        labels = np.concatenate([np.ones(len(indices), dtype=np.int8), np.zeros(len(indices), dtype=np.int8)])
        weights = np.concatenate([
            np.asarray([float(rows[index]["pair_weight"]) for index in indices]),
            np.asarray([float(rows[index]["pair_weight"]) for index in indices]),
        ])
        model = LogisticRegression(C=1.0, max_iter=args.max_iter, solver="liblinear", random_state=args.seed)
        model.fit(matrix(smiles), labels, sample_weight=weights)
        return model

    print(json.dumps({"stage": "fit_diagnostic", "train_pairs": int(len(train_idx))}), flush=True)
    diagnostic_model = fit(train_idx)
    heldout_positive = matrix([str(rows[index]["positive_smiles"]) for index in heldout_idx])
    heldout_negative = matrix([str(rows[index]["negative_smiles"]) for index in heldout_idx])
    heldout_win_rate = float(np.mean(
        diagnostic_model.predict_proba(heldout_positive)[:, 1]
        > diagnostic_model.predict_proba(heldout_negative)[:, 1]
    ))

    # Refit on every pair before blinded panel prediction.
    print(json.dumps({"stage": "fit_full", "pairs": len(rows)}), flush=True)
    model = fit(np.arange(len(rows)))
    panel = read_panel(args.candidate_panels, args.panel)
    if args.max_panel_rows is not None:
        panel = panel[: args.max_panel_rows]
    print(json.dumps({"stage": "score_panel", "rows": len(panel)}), flush=True)
    scores = model.predict_proba(matrix([row["SMILES"] for row in panel]))[:, 1]
    result = pl.DataFrame(
        {
            "CatalogID": [row["CatalogID"] for row in panel],
            "SMILES": [row["SMILES"] for row in panel],
            "fingerprint_score": scores,
        }
    ).sort("fingerprint_score", descending=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ranked = args.output_dir / f"{args.panel}_ranked_scores.csv"
    top50 = args.output_dir / f"{args.panel}_top50.csv"
    result.write_csv(ranked)
    result.head(50).write_csv(top50)
    report = {
        "target": "PGK2",
        "model": "Morgan radius-2 / 2048-bit logistic-regression baseline",
        "pairs": int(pairs.height),
        "heldout_scaffold_pair_win_rate": heldout_win_rate,
        "panel": args.panel,
        "scored_rows": len(panel),
        "outputs": {"ranked": str(ranked), "top50": str(top50)},
        "limitations": [
            "The holdout is DEL-derived and not an ASMS estimate.",
            "The selected model must be chosen by blinded validation retrieval, not this diagnostic.",
        ],
        "elapsed_seconds": time.perf_counter() - started,
    }
    (args.output_dir / f"{args.panel}_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
