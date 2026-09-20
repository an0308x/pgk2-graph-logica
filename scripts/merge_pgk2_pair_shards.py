#!/usr/bin/env python3
"""Merge PGK2 pairwise-training shards and verify the leakage boundary."""

from __future__ import annotations

import argparse
import csv
import json
import zipfile
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-root", type=Path, default=ROOT / "artifacts" / "pgk2_pairs")
    parser.add_argument(
        "--candidate-panels", type=Path, default=ROOT / "DREAM_challenge_2026" / "Val-Test-set.zip"
    )
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "pgk2_pairs" / "pgk2_ranking_pairs.parquet")
    return parser.parse_args()


def candidate_smiles(path: Path) -> set[str]:
    result: set[str] = set()
    with zipfile.ZipFile(path) as archive:
        for member in ("Val-Test-set/PGK2_Validation_split.csv", "Val-Test-set/PGK2_Test_split.csv"):
            with archive.open(member) as handle:
                result.update(row["SMILES"] for row in csv.DictReader(line.decode() for line in handle))
    return result


def main() -> int:
    args = parse_args()
    shards = sorted(args.shard_root.rglob("pgk2_ranking_pairs_offset*_n*.parquet"))
    if not shards:
        raise FileNotFoundError(f"No shard pair files under {args.shard_root}")
    frames = [pl.read_parquet(path) for path in shards]
    pairs = pl.concat(frames, how="vertical")
    required = {"positive_smiles", "negative_smiles", "pair_weight", "pair_type"}
    if not required.issubset(pairs.columns):
        raise ValueError(f"Missing required columns: {sorted(required - set(pairs.columns))}")
    if pairs.filter(pl.col("positive_smiles") == pl.col("negative_smiles")).height:
        raise ValueError("A pair contains the same molecule on both sides")
    if pairs.filter(pl.col("pair_weight") <= 0).height:
        raise ValueError("A pair has a non-positive weight")
    duplicates = pairs.unique(subset=["positive_smiles", "negative_smiles"]).height != pairs.height
    if duplicates:
        raise ValueError("Duplicate pair rows across shards")
    candidates = candidate_smiles(args.candidate_panels)
    candidate_overlap = pairs.filter(
        pl.col("positive_smiles").is_in(candidates) | pl.col("negative_smiles").is_in(candidates)
    ).height
    if candidate_overlap:
        raise ValueError(f"Found {candidate_overlap} pairs overlapping blinded candidate panels")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pairs.write_parquet(args.output, compression="zstd")
    report = {
        "input_shards": [str(path) for path in shards],
        "rows": pairs.height,
        "unique_positive_smiles": pairs.get_column("positive_smiles").n_unique(),
        "unique_negative_smiles": pairs.get_column("negative_smiles").n_unique(),
        "weight_sum": float(pairs.get_column("pair_weight").sum()),
        "by_type": pairs.group_by("pair_type").len().sort("pair_type").to_dicts(),
        "candidate_panel_pair_overlap": candidate_overlap,
        "output": str(args.output),
    }
    (args.output.parent / "merge_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
