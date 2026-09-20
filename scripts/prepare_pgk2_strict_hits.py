#!/usr/bin/env python3
"""Create an auditable, deduplicated high-confidence PGK2-positive set.

This follows the organizer-linked deduplication recipe: exact-SMILES groups
sum count columns, Stouffer-combine z-scores, and retain the maximum number of
historic hits. It then applies the published recommended global hit criteria:
count_PGK2 > 3, low inhibitor/NTC counts relative to PGK2, and historic_hits < 5.
No negatives are constructed here; their cluster-aware selection from OpenDEL
is intentionally a separate, computationally larger step.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
import zipfile
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, default=ROOT / "PGK2_selection.parquet")
    parser.add_argument(
        "--candidate-panels",
        type=Path,
        default=ROOT / "DREAM_challenge_2026" / "Val-Test-set.zip",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts" / "pgk2_training")
    return parser.parse_args()


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def candidate_smiles(path: Path) -> dict[str, set[str]]:
    members = {
        "validation": "Val-Test-set/PGK2_Validation_split.csv",
        "test": "Val-Test-set/PGK2_Test_split.csv",
    }
    result: dict[str, set[str]] = {}
    with zipfile.ZipFile(path) as archive:
        for name, member in members.items():
            with archive.open(member) as handle:
                result[name] = {row["SMILES"] for row in csv.DictReader(line.decode("utf-8") for line in handle)}
    return result


def main() -> int:
    args = parse_args()
    started = time.perf_counter()
    if not args.selection.exists() or not args.candidate_panels.exists():
        raise FileNotFoundError("Selection release or candidate-panel archive is missing")
    score_columns = [
        "zscore_PGK2",
        "zscore_PGK2_with_inhibitor",
        "zscore_NTC",
    ]
    count_columns = ["count_PGK2", "count_PGK2_with_inhibitor", "count_NTC"]
    aggregate = [
        pl.col("compound").first().alias("representative_compound"),
        pl.len().alias("source_rows"),
        *[pl.col(column).sum().alias(column) for column in count_columns],
        *[
            (pl.col(column).sum() / pl.len().cast(pl.Float64).sqrt()).alias(column)
            for column in score_columns
        ],
        pl.col("historic_hits").max().alias("historic_hits"),
    ]
    deduplicated = (
        pl.scan_parquet(args.selection)
        .filter(pl.col("SMILES").is_not_null() & (pl.col("SMILES") != ""))
        .group_by("SMILES")
        .agg(aggregate)
    )
    strict_hits = (
        deduplicated.filter(
            (pl.col("count_PGK2") > 3)
            & ((pl.col("count_PGK2_with_inhibitor") == 0) | (pl.col("count_PGK2_with_inhibitor") < 0.1 * pl.col("count_PGK2")))
            & ((pl.col("count_NTC") == 0) | (pl.col("count_NTC") < 0.1 * pl.col("count_PGK2")))
            & (pl.col("historic_hits") < 5)
        )
        .with_columns(pl.lit(1).alias("label"))
        .collect(engine="streaming")
    )
    panels = candidate_smiles(args.candidate_panels)
    strict_hits = strict_hits.with_columns(
        pl.col("SMILES").is_in(panels["validation"]).alias("in_validation_panel"),
        pl.col("SMILES").is_in(panels["test"]).alias("in_test_panel"),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "strict_positive_hits.parquet"
    strict_hits.write_parquet(output, compression="zstd")
    report = {
        "target": "PGK2",
        "input": {"selection": str(args.selection), "selection_sha256": sha256(args.selection)},
        "deduplication": {
            "key": "exact SMILES",
            "count_columns": "sum",
            "zscore_columns": "Stouffer sum(z) / sqrt(n)",
            "historic_hits": "max",
        },
        "strict_hit_criteria": {
            "count_PGK2": "> 3",
            "count_PGK2_with_inhibitor": "== 0 or < 0.1 * count_PGK2",
            "count_NTC": "== 0 or < 0.1 * count_PGK2",
            "historic_hits": "< 5",
        },
        "strict_positive_rows": strict_hits.height,
        "candidate_panel_overlap": {
            "validation": int(strict_hits["in_validation_panel"].sum()),
            "test": int(strict_hits["in_test_panel"].sum()),
        },
        "output": str(output),
        "elapsed_seconds": time.perf_counter() - started,
    }
    (args.output_dir / "strict_positive_hits_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
