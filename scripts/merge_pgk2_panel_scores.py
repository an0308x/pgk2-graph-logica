#!/usr/bin/env python3
"""Validate and merge chunked PGK2 challenge-panel scores into a top-50 ranking."""

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
    parser.add_argument("--panel", choices=("validation", "test"), default="validation")
    parser.add_argument("--score-dir", type=Path, required=True)
    parser.add_argument("--candidate-panels", type=Path, default=ROOT / "DREAM_challenge_2026" / "Val-Test-set.zip")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def panel_rows(path: Path, panel: str) -> list[dict[str, str]]:
    member = f"Val-Test-set/PGK2_{panel.capitalize()}_split.csv"
    with zipfile.ZipFile(path) as archive, archive.open(member) as handle:
        return list(csv.DictReader(line.decode() for line in handle))


def main() -> int:
    args = parse_args()
    expected = panel_rows(args.candidate_panels, args.panel)
    expected_ids = {row["CatalogID"] for row in expected}
    chunks = sorted(args.score_dir.glob(f"{args.panel}_offset*_n*.csv"))
    if not chunks:
        raise FileNotFoundError(f"No {args.panel} score chunks in {args.score_dir}")
    scores = pl.concat([pl.read_csv(path) for path in chunks], how="vertical")
    if set(scores.columns) != {"CatalogID", "SMILES", "logica_score"}:
        raise ValueError(f"Unexpected score columns: {scores.columns}")
    if scores.get_column("CatalogID").n_unique() != scores.height:
        raise ValueError("Duplicate CatalogIDs across score chunks")
    observed_ids = set(scores.get_column("CatalogID").to_list())
    missing, extra = expected_ids - observed_ids, observed_ids - expected_ids
    if missing or extra:
        raise ValueError(f"Incomplete panel score coverage: missing={len(missing)}, extra={len(extra)}")
    ranked = scores.sort("logica_score", descending=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ranked.write_csv(args.output_dir / f"{args.panel}_ranked_scores.csv")
    ranked.head(50).write_csv(args.output_dir / f"{args.panel}_top50.csv")
    report = {
        "panel": args.panel,
        "scored_rows": ranked.height,
        "chunks": [str(path) for path in chunks],
        "top_score": float(ranked[0, "logica_score"]),
        "bottom_score": float(ranked[-1, "logica_score"]),
    }
    (args.output_dir / f"{args.panel}_merge_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
