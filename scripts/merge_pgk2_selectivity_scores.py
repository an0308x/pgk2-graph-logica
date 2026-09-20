#!/usr/bin/env python3
"""Validate and merge chunked PGK2-versus-PGK1 LogiCA scores."""

from __future__ import annotations

import argparse
import csv
import json
import zipfile
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
FIELDS = {
    "CatalogID",
    "SMILES",
    "pgk2_logica_score",
    "pgk1_logica_score",
    "logica_selectivity_delta",
}


def panel_rows(path: Path, panel: str) -> list[dict[str, str]]:
    member = f"Val-Test-set/PGK2_{panel.capitalize()}_split.csv"
    with zipfile.ZipFile(path) as archive, archive.open(member) as handle:
        return list(csv.DictReader(line.decode("utf-8") for line in handle))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", choices=("validation", "test"), default="validation")
    parser.add_argument("--score-dir", type=Path, required=True)
    parser.add_argument(
        "--candidate-panels",
        type=Path,
        default=ROOT / "DREAM_challenge_2026" / "Val-Test-set.zip",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    expected = panel_rows(args.candidate_panels, args.panel)
    expected_by_id = {row["CatalogID"]: row["SMILES"] for row in expected}
    chunks = sorted(args.score_dir.glob(f"{args.panel}_offset*_n*.csv"))
    if not chunks:
        raise FileNotFoundError(f"No {args.panel} score chunks in {args.score_dir}")
    scores = pl.concat([pl.read_csv(path) for path in chunks], how="vertical")
    if set(scores.columns) != FIELDS:
        raise ValueError(f"Unexpected score columns: {scores.columns}")
    if scores.get_column("CatalogID").n_unique() != scores.height:
        raise ValueError("Duplicate CatalogIDs across score chunks")
    observed_ids = set(scores.get_column("CatalogID").to_list())
    missing = set(expected_by_id) - observed_ids
    extra = observed_ids - set(expected_by_id)
    if missing or extra:
        raise ValueError(f"Incomplete panel score coverage: missing={len(missing)}, extra={len(extra)}")
    for catalog_id, smiles in scores.select("CatalogID", "SMILES").iter_rows():
        if expected_by_id[catalog_id] != smiles:
            raise ValueError(f"SMILES mismatch for {catalog_id}")

    ranked = scores.sort("logica_selectivity_delta", descending=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / f"{args.panel}_ranked_scores.csv"
    ranked.write_csv(output)
    report = {
        "panel": args.panel,
        "scored_rows": ranked.height,
        "chunks": [str(path) for path in chunks],
        "output": str(output),
        "delta_range": [
            float(ranked.get_column("logica_selectivity_delta").min()),
            float(ranked.get_column("logica_selectivity_delta").max()),
        ],
    }
    (args.output_dir / f"{args.panel}_merge_report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
