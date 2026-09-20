#!/usr/bin/env python3
"""Mine high-confidence, local PGK2 DEL SAR ranking pairs.

Each OpenDEL compound identifier has the form ``library-BB1-BB2-BB3``.  This
builder compares a stringent PGK2 DEL positive only with selection compounds
from the same library that share two building blocks--i.e. one synthetic
building-block substitution.  A Morgan similarity threshold is applied as an
independent structural check.  Unlike the earlier broad-pool builder, this
does not use random low-count negatives and it requires *observed* inhibitor
or NTC evidence in the lower-ranked compound.

The objective remains a DEL-derived ranking diagnostic, not an ASMS label.
All blinded candidate SMILES are excluded before pair construction.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import polars as pl
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, default=ROOT / "PGK2_selection.parquet")
    parser.add_argument("--ntc-supplement", type=Path, default=ROOT / "PGK2_NTC_supplement.parquet")
    parser.add_argument(
        "--strict-positives",
        type=Path,
        default=ROOT / "artifacts" / "pgk2_training" / "strict_positive_hits.parquet",
    )
    parser.add_argument(
        "--candidate-panels",
        type=Path,
        default=ROOT / "DREAM_challenge_2026" / "Val-Test-set.zip",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts" / "pgk2_matched_sar_pairs")
    parser.add_argument("--minimum-similarity", type=float, default=0.60)
    parser.add_argument("--pairs-per-positive", type=int, default=2)
    parser.add_argument("--max-positives", type=int, default=None, help="Bounded smoke run only.")
    parser.add_argument("--positive-offset", type=int, default=0, help="Offset in deterministic positive ordering.")
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def candidate_smiles(path: Path) -> set[str]:
    members = ("Val-Test-set/PGK2_Validation_split.csv", "Val-Test-set/PGK2_Test_split.csv")
    result: set[str] = set()
    with zipfile.ZipFile(path) as archive:
        for member in members:
            with archive.open(member) as handle:
                result.update(row["SMILES"] for row in csv.DictReader(line.decode() for line in handle))
    return result


def selection_frame(selection: Path, ntc_supplement: Path) -> pl.LazyFrame:
    """Deduplicate exact SMILES and retain one representative OpenDEL code."""

    selection_counts = ["count_PGK2", "count_PGK2_with_inhibitor", "count_NTC"]
    zscores = ["zscore_PGK2", "zscore_PGK2_with_inhibitor", "zscore_NTC"]
    selection_lf = (
        pl.scan_parquet(selection)
        .filter(pl.col("SMILES").is_not_null() & (pl.col("SMILES") != ""))
        .group_by("SMILES")
        .agg(
            pl.col("compound").first().alias("compound"),
            pl.len().alias("source_rows"),
            *[pl.col(column).sum().alias(column) for column in selection_counts],
            *[(pl.col(column).sum() / pl.len().cast(pl.Float64).sqrt()).alias(column) for column in zscores],
            pl.col("historic_hits").max().alias("historic_hits"),
        )
    )
    supplement_lf = (
        pl.scan_parquet(ntc_supplement)
        .filter(pl.col("SMILES").is_not_null() & (pl.col("SMILES") != ""))
        .group_by("SMILES")
        .agg(pl.col("count_NTC").sum().alias("count_NTC_supplement"))
    )
    return (
        selection_lf.join(supplement_lf, on="SMILES", how="left")
        .with_columns(pl.col("count_NTC_supplement").fill_null(0))
        .with_columns(pl.max_horizontal("count_NTC", "count_NTC_supplement").alias("count_NTC_evidence"))
    )


def rechecked_positives(path: Path, supplement: Path, excluded: set[str], seed: int) -> pl.DataFrame:
    positives = pl.read_parquet(path).filter(~pl.col("SMILES").is_in(excluded))
    positive_ntc = (
        pl.scan_parquet(supplement)
        .filter(pl.col("SMILES").is_not_null() & (pl.col("SMILES") != ""))
        .group_by("SMILES")
        .agg(pl.col("count_NTC").sum().alias("count_NTC_supplement"))
        .collect(engine="streaming")
    )
    return (
        positives.lazy()
        .join(positive_ntc.lazy(), on="SMILES", how="left")
        .with_columns(pl.col("count_NTC_supplement").fill_null(0))
        .with_columns(pl.max_horizontal("count_NTC", "count_NTC_supplement").alias("count_NTC_evidence"))
        .filter(pl.col("count_NTC_evidence") < 0.1 * pl.col("count_PGK2"))
        .with_columns(pl.col("SMILES").hash(seed=seed).alias("_order"))
        .sort("_order")
        .drop("_order")
        .collect(engine="streaming")
    )


def building_block_keys(compound: str) -> tuple[str, str, str] | None:
    parts = compound.split("-")
    if len(parts) != 4 or any(not part for part in parts):
        return None
    library, first, second, third = parts
    return (
        f"{library}:{first}:{second}",
        f"{library}:{first}:{third}",
        f"{library}:{second}:{third}",
    )


def morgan_fingerprint(smiles: str):
    molecule = Chem.MolFromSmiles(smiles)
    return AllChem.GetMorganGenerator(radius=2, fpSize=2048).GetFingerprint(molecule) if molecule else None


def candidate_has_clear_counter_evidence(row: dict[str, object]) -> bool:
    return int(row["count_PGK2_with_inhibitor"]) > 0 or int(row["count_NTC_evidence"]) > 0


def positive_dominates(positive: dict[str, object], negative: dict[str, object]) -> bool:
    """Require strong, directional DEL evidence without interpreting a zero as positive evidence."""

    return (
        int(positive["count_PGK2"]) > int(negative["count_PGK2"])
        and int(positive["count_PGK2_with_inhibitor"]) <= int(negative["count_PGK2_with_inhibitor"])
        and int(positive["count_NTC_evidence"]) <= int(negative["count_NTC_evidence"])
        and candidate_has_clear_counter_evidence(negative)
    )


def keyed_rows(rows: list[dict[str, object]]) -> dict[str, list[dict[str, object]]]:
    result: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        keys = building_block_keys(str(row["compound"]))
        if keys is not None:
            for key in keys:
                result[key].append(row)
    return result


def main() -> int:
    args = parse_args()
    if not 0 <= args.minimum_similarity <= 1 or args.pairs_per_positive <= 0 or args.positive_offset < 0:
        raise ValueError("invalid similarity, pairs-per-positive, or positive offset")
    for path in (args.selection, args.ntc_supplement, args.strict_positives, args.candidate_panels):
        if not path.exists():
            raise FileNotFoundError(path)
    started = time.perf_counter()
    excluded = candidate_smiles(args.candidate_panels)
    positives_all = rechecked_positives(args.strict_positives, args.ntc_supplement, excluded, args.seed)
    remaining = positives_all.height - args.positive_offset
    length = remaining if args.max_positives is None else args.max_positives
    positives = positives_all.slice(args.positive_offset, max(0, length))
    if not positives.height:
        raise ValueError("No positives in requested range")

    positive_smiles = positives.get_column("SMILES").to_list()
    # Only candidates with an observed counter arm can enter a pair.  This is
    # both biologically more defensible and much smaller than all DEL entries.
    candidate_frame = (
        selection_frame(args.selection, args.ntc_supplement)
        .filter(
            ~pl.col("SMILES").is_in(positive_smiles)
            & ~pl.col("SMILES").is_in(excluded)
            & ((pl.col("count_PGK2_with_inhibitor") > 0) | (pl.col("count_NTC_evidence") > 0))
        )
        .collect(engine="streaming")
    )
    candidates_by_key = keyed_rows(candidate_frame.to_dicts())
    pairs: list[dict[str, object]] = []
    positives_with_neighbors = 0
    for positive in positives.to_dicts():
        keys = building_block_keys(str(positive["representative_compound"]))
        pos_fp = morgan_fingerprint(str(positive["SMILES"]))
        if keys is None or pos_fp is None:
            continue
        # A molecule can share two blocks in more than one representation;
        # preserve only its strongest qualifying comparison for this positive.
        seen: dict[str, tuple[dict[str, object], float, str]] = {}
        for key in keys:
            for negative in candidates_by_key.get(key, []):
                negative_smiles = str(negative["SMILES"])
                if negative_smiles in seen or not positive_dominates(positive, negative):
                    continue
                neg_fp = morgan_fingerprint(negative_smiles)
                if neg_fp is None:
                    continue
                similarity = float(DataStructs.TanimotoSimilarity(pos_fp, neg_fp))
                if similarity < args.minimum_similarity:
                    continue
                seen[negative_smiles] = (negative, similarity, key)
        selected = sorted(
            seen.values(),
            key=lambda item: (-item[1], int(item[0]["count_PGK2"]), str(item[0]["SMILES"])),
        )[: args.pairs_per_positive]
        if selected:
            positives_with_neighbors += 1
        for negative, similarity, key in selected:
            target_margin = np.log1p(int(positive["count_PGK2"])) - np.log1p(int(negative["count_PGK2"]))
            # All pairs have observed counter evidence.  The modest margin term
            # prioritizes robust orderings without allowing count magnitude to dominate.
            weight = float(1.0 + min(target_margin, 3.0) / 3.0)
            pairs.append(
                {
                    "positive_smiles": positive["SMILES"],
                    "negative_smiles": negative["SMILES"],
                    "pair_weight": weight,
                    "pair_type": "matched_building_block_control_confirmed",
                    "shared_building_block_key": key,
                    "tanimoto_similarity": similarity,
                    "positive_count_PGK2": positive["count_PGK2"],
                    "negative_count_PGK2": negative["count_PGK2"],
                    "positive_count_inhibitor": positive["count_PGK2_with_inhibitor"],
                    "negative_count_inhibitor": negative["count_PGK2_with_inhibitor"],
                    "positive_count_NTC_evidence": positive["count_NTC_evidence"],
                    "negative_count_NTC_evidence": negative["count_NTC_evidence"],
                }
            )
    if not pairs:
        raise ValueError("No matched, control-confirmed SAR pairs were generated")
    frame = pl.DataFrame(pairs)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "" if args.max_positives is None and args.positive_offset == 0 else f"_offset{args.positive_offset}_n{positives.height}"
    output = args.output_dir / f"pgk2_matched_sar_pairs{suffix}.parquet"
    frame.write_parquet(output, compression="zstd")
    report = {
        "target": "PGK2",
        "policy": {
            "neighborhood": "same OpenDEL library and exactly two shared building blocks",
            "minimum_morgan_tanimoto": args.minimum_similarity,
            "negative_evidence": "observed inhibitor or NTC count, with no better control counts than the positive",
            "candidate_panels_excluded": len(excluded),
        },
        "positives": {"eligible": positives_all.height, "used": positives.height, "with_matched_neighbor": positives_with_neighbors},
        "counter_evidence_candidates": candidate_frame.height,
        "pairs": {
            "rows": frame.height,
            "weight_sum": float(frame.get_column("pair_weight").sum()),
            "similarity": {key: float(frame.get_column("tanimoto_similarity").quantile(q)) for key, q in (("q10", 0.1), ("median", 0.5), ("q90", 0.9))},
        },
        "output": str(output),
        "elapsed_seconds": time.perf_counter() - started,
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
