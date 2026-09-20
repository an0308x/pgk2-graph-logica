#!/usr/bin/env python3
"""Prepare an auditable high-information pool for SELFormer/SELFIES mining.

The pool deliberately excludes ASMS validation and test SMILES.  Its anchors
are rechecked stringent DEL positives.  Its comparison molecules have
*observed* inhibitor or NTC counter-signal, so a control zero is never used as
positive evidence of active-site specificity.
"""

from __future__ import annotations

import argparse
import csv
import json
import zipfile
from pathlib import Path

import polars as pl


ROOT = Path(__file__).resolve().parents[1]


def panel_smiles(path: Path) -> set[str]:
    result: set[str] = set()
    members = ("Val-Test-set/PGK2_Validation_split.csv", "Val-Test-set/PGK2_Test_split.csv")
    with zipfile.ZipFile(path) as archive:
        for member in members:
            with archive.open(member) as handle:
                result.update(row["SMILES"] for row in csv.DictReader(line.decode() for line in handle))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, default=ROOT / "PGK2_selection.parquet")
    parser.add_argument("--ntc-supplement", type=Path, default=ROOT / "PGK2_NTC_supplement.parquet")
    parser.add_argument(
        "--candidate-panels", type=Path, default=ROOT / "DREAM_challenge_2026/Val-Test-set.zip"
    )
    parser.add_argument(
        "--strict-positives",
        type=Path,
        default=ROOT / "artifacts/pgk2_training/strict_positive_hits.parquet",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "artifacts/pgk2_selfies_neighbor_pool"
    )
    args = parser.parse_args()
    for path in (args.selection, args.ntc_supplement, args.candidate_panels, args.strict_positives):
        if not path.is_file():
            raise FileNotFoundError(path)
    excluded = panel_smiles(args.candidate_panels)
    selection = (
        pl.scan_parquet(args.selection)
        .filter(pl.col("SMILES").is_not_null() & (pl.col("SMILES") != ""))
        .group_by("SMILES")
        .agg(
            pl.col("compound").first().alias("representative_compound"),
            pl.len().alias("source_rows"),
            *[pl.col(column).sum().alias(column) for column in ("count_PGK2", "count_PGK2_with_inhibitor", "count_NTC")],
            *[
                (pl.col(column).sum() / pl.len().cast(pl.Float64).sqrt()).alias(column)
                for column in ("zscore_PGK2", "zscore_PGK2_with_inhibitor", "zscore_NTC")
            ],
            pl.col("historic_hits").max().alias("historic_hits"),
        )
    )
    supplement = (
        pl.scan_parquet(args.ntc_supplement)
        .filter(pl.col("SMILES").is_not_null() & (pl.col("SMILES") != ""))
        .group_by("SMILES")
        .agg(pl.col("count_NTC").sum().alias("count_NTC_supplement"))
    )
    base = (
        selection.join(supplement, on="SMILES", how="left")
        .with_columns(pl.col("count_NTC_supplement").fill_null(0))
        .with_columns(pl.max_horizontal("count_NTC", "count_NTC_supplement").alias("count_NTC_evidence"))
        .filter(~pl.col("SMILES").is_in(excluded))
        .with_columns(pl.col("representative_compound").str.split("-").list.first().alias("library"))
    )
    rechecked_anchors = (
        pl.read_parquet(args.strict_positives)
        .select("SMILES")
        .lazy()
        .join(base, on="SMILES", how="inner")
        .filter(pl.col("count_NTC_evidence") < 0.1 * pl.col("count_PGK2"))
        .with_columns(pl.lit("anchor").alias("pool_role"))
    )
    comparisons = (
        base.join(rechecked_anchors.select("SMILES"), on="SMILES", how="anti")
        .filter((pl.col("count_PGK2_with_inhibitor") > 0) | (pl.col("count_NTC_evidence") > 0))
        .with_columns(pl.lit("counter_evidence_comparator").alias("pool_role"))
    )
    pool = pl.concat([rechecked_anchors, comparisons], how="vertical").collect(engine="streaming")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "selfies_neighbor_pool.parquet"
    pool.write_parquet(output, compression="zstd")
    report = {
        "purpose": "Frozen SELFormer/SELFIES local-neighbor mining; no ASMS labels.",
        "excluded_asms_smiles": len(excluded),
        "rows": pool.height,
        "anchors": int((pool["pool_role"] == "anchor").sum()),
        "counter_evidence_comparators": int((pool["pool_role"] == "counter_evidence_comparator").sum()),
        "columns": pool.columns,
        "output": str(output),
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
