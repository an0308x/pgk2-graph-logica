#!/usr/bin/env python3
"""Build a leakage-free, control-aware PGK2 pairwise-ranking dataset.

The released PGK2 selection table only contains compounds observed in the
target arm.  Consequently, an absent inhibitor/NTC count is treated as
*missing evidence*, never as positive specificity evidence.  Pairs are made
only when a published strict positive has stronger target evidence than a
candidate with observed counter-evidence, or (at lower weight) a low-target
candidate.  Candidate-panel SMILES are excluded before any training pairs are
made.

The output is intentionally a set of pairwise preferences, not binary binding
labels for every DEL compound.
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
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts" / "pgk2_pairs")
    parser.add_argument("--counter-pool-size", type=int, default=20_000)
    parser.add_argument("--low-target-pool-size", type=int, default=20_000)
    parser.add_argument("--local-negatives-per-positive", type=int, default=1)
    parser.add_argument("--global-negatives-per-positive", type=int, default=1)
    parser.add_argument("--max-positives", type=int, default=None, help="Bounded smoke run only.")
    parser.add_argument("--positive-offset", type=int, default=0, help="Offset in the deterministic positive ordering; supports cluster shards.")
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def candidate_smiles(path: Path) -> set[str]:
    members = (
        "Val-Test-set/PGK2_Validation_split.csv",
        "Val-Test-set/PGK2_Test_split.csv",
    )
    smiles: set[str] = set()
    with zipfile.ZipFile(path) as archive:
        for member in members:
            with archive.open(member) as handle:
                smiles.update(row["SMILES"] for row in csv.DictReader(line.decode() for line in handle))
    return smiles


def deduplicated_selection(selection: Path, ntc_supplement: Path) -> pl.LazyFrame:
    """Aggregate exact SMILES consistently with the strict-positive procedure."""

    selection_counts = ["count_PGK2", "count_PGK2_with_inhibitor", "count_NTC"]
    zscores = ["zscore_PGK2", "zscore_PGK2_with_inhibitor", "zscore_NTC"]
    selection_lf = (
        pl.scan_parquet(selection)
        .filter(pl.col("SMILES").is_not_null() & (pl.col("SMILES") != ""))
        .group_by("SMILES")
        .agg(
            pl.col("compound").first().alias("representative_compound"),
            pl.len().alias("source_rows"),
            *[pl.col(column).sum().alias(column) for column in selection_counts],
            *[(pl.col(column).sum() / pl.len().cast(pl.Float64).sqrt()).alias(column) for column in zscores],
            pl.col("historic_hits").max().alias("historic_hits"),
        )
    )
    # The supplement includes control-arm observations absent from the original
    # target-conditioned table.  max() avoids double-counting overlapping reads
    # while preserving the fact that an NTC observation exists.
    ntc_lf = (
        pl.scan_parquet(ntc_supplement)
        .filter(pl.col("SMILES").is_not_null() & (pl.col("SMILES") != ""))
        .group_by("SMILES")
        .agg(pl.col("count_NTC").sum().alias("count_NTC_supplement"))
    )
    return (
        selection_lf.join(ntc_lf, on="SMILES", how="left")
        .with_columns(pl.col("count_NTC_supplement").fill_null(0))
        .with_columns(
            pl.max_horizontal("count_NTC", "count_NTC_supplement").alias("count_NTC_evidence")
        )
    )


def sampled_pool(frame: pl.LazyFrame, size: int, seed: int) -> pl.DataFrame:
    """Return a deterministic hash sample without relying on file order."""

    if size <= 0:
        raise ValueError("pool sizes must be positive")
    return (
        frame.with_columns(pl.col("SMILES").hash(seed=seed).alias("_sample_hash"))
        .sort("_sample_hash")
        .head(size)
        .collect(engine="streaming")
        .drop("_sample_hash")
    )


def fingerprint(smiles: str):
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return None
    return AllChem.GetMorganGenerator(radius=2, fpSize=2048).GetFingerprint(molecule)


def classify_pair(positive: dict[str, object], negative: dict[str, object]) -> tuple[str, float] | None:
    """Assign confidence without ever inferring specificity from a control zero."""

    if int(positive["count_PGK2"]) <= int(negative["count_PGK2"]):
        return None
    positive_i = int(positive["count_PGK2_with_inhibitor"])
    negative_i = int(negative["count_PGK2_with_inhibitor"])
    positive_n = int(positive["count_NTC_evidence"])
    negative_n = int(negative["count_NTC_evidence"])
    controls_no_worse = positive_i <= negative_i and positive_n <= negative_n
    negative_has_observed_control = negative_i > 0 or negative_n > 0
    if controls_no_worse and negative_has_observed_control:
        return "control_confirmed_dominance", 1.0
    # Target-only ordering is useful but weaker: both control values can be
    # absent due to release/sampling mechanics rather than true specificity.
    if controls_no_worse:
        return "target_dominance_without_control_confirmation", 0.35
    return None


def make_pairs(
    positives: pl.DataFrame,
    counter_pool: pl.DataFrame,
    low_pool: pl.DataFrame,
    local_per_positive: int,
    global_per_positive: int,
    seed: int,
) -> list[dict[str, object]]:
    if local_per_positive < 0 or global_per_positive < 0:
        raise ValueError("negatives per positive must be non-negative")
    if local_per_positive + global_per_positive == 0:
        raise ValueError("at least one negative per positive is required")

    rng = np.random.default_rng(seed)
    counter_rows = counter_pool.to_dicts()
    low_rows = low_pool.to_dicts()
    counter_fps = [fingerprint(str(row["SMILES"])) for row in counter_rows]
    usable = [(row, fp) for row, fp in zip(counter_rows, counter_fps, strict=True) if fp is not None]
    if not usable or not low_rows:
        raise ValueError("Negative pools were empty after filtering invalid SMILES")
    counter_rows, counter_fps = map(list, zip(*usable, strict=True))

    pairs: list[dict[str, object]] = []
    for positive in positives.to_dicts():
        pos_fp = fingerprint(str(positive["SMILES"]))
        if pos_fp is None:
            continue
        similarities = np.asarray(DataStructs.BulkTanimotoSimilarity(pos_fp, counter_fps))
        local_indices = np.argsort(-similarities)[:local_per_positive]
        global_indices = rng.integers(0, len(low_rows), size=global_per_positive)
        candidates = [(counter_rows[int(index)], float(similarities[int(index)]), "local_counter") for index in local_indices]
        candidates.extend((low_rows[int(index)], None, "global_low_target") for index in global_indices)
        for negative, similarity, source in candidates:
            classification = classify_pair(positive, negative)
            if classification is None:
                continue
            pair_type, weight = classification
            pairs.append(
                {
                    "positive_smiles": positive["SMILES"],
                    "negative_smiles": negative["SMILES"],
                    "pair_weight": weight,
                    "pair_type": pair_type,
                    "negative_source": source,
                    "tanimoto_similarity": similarity,
                    "positive_count_PGK2": positive["count_PGK2"],
                    "negative_count_PGK2": negative["count_PGK2"],
                    "positive_count_inhibitor": positive["count_PGK2_with_inhibitor"],
                    "negative_count_inhibitor": negative["count_PGK2_with_inhibitor"],
                    "positive_count_NTC_evidence": positive["count_NTC_evidence"],
                    "negative_count_NTC_evidence": negative["count_NTC_evidence"],
                    "negative_historic_hits": negative["historic_hits"],
                }
            )
    return pairs


def main() -> int:
    args = parse_args()
    started = time.perf_counter()
    for path in (args.selection, args.ntc_supplement, args.strict_positives, args.candidate_panels):
        if not path.exists():
            raise FileNotFoundError(path)
    excluded = candidate_smiles(args.candidate_panels)
    positives_before_supplement = pl.read_parquet(args.strict_positives).filter(
        ~pl.col("SMILES").is_in(excluded)
    )
    # Recheck strict positives against the subsequently released NTC evidence.
    # A control observation can rule a compound out; a missing observation can
    # never promote one.
    positive_ntc = (
        pl.scan_parquet(args.ntc_supplement)
        .filter(pl.col("SMILES").is_not_null() & (pl.col("SMILES") != ""))
        .group_by("SMILES")
        .agg(pl.col("count_NTC").sum().alias("count_NTC_supplement"))
        .collect(engine="streaming")
    )
    positives_after_supplement_recheck = (
        positives_before_supplement.lazy()
        .join(positive_ntc.lazy(), on="SMILES", how="left")
        .with_columns(pl.col("count_NTC_supplement").fill_null(0))
        .with_columns(pl.max_horizontal("count_NTC", "count_NTC_supplement").alias("count_NTC_evidence"))
        .filter(pl.col("count_NTC_evidence") < 0.1 * pl.col("count_PGK2"))
        .collect(engine="streaming")
    )
    if args.positive_offset < 0:
        raise ValueError("--positive-offset must be non-negative")
    ordered_positives = (
        positives_after_supplement_recheck.with_columns(pl.col("SMILES").hash(seed=args.seed).alias("_positive_hash"))
        .sort("_positive_hash")
        .drop("_positive_hash")
    )
    positive_length = ordered_positives.height - args.positive_offset if args.max_positives is None else args.max_positives
    positives = ordered_positives.slice(args.positive_offset, max(0, positive_length))
    if not positives.height:
        raise ValueError("No positives in the requested positive shard")
    positive_smiles = positives.get_column("SMILES").to_list()
    universe = deduplicated_selection(args.selection, args.ntc_supplement).filter(
        ~pl.col("SMILES").is_in(positive_smiles) & ~pl.col("SMILES").is_in(excluded)
    )
    counter_pool = sampled_pool(
        universe.filter(
            (pl.col("count_PGK2_with_inhibitor") > 0)
            | (pl.col("count_NTC_evidence") > 0)
            | (pl.col("historic_hits") >= 5)
        ),
        args.counter_pool_size,
        args.seed,
    )
    low_pool = sampled_pool(
        universe.filter(
            (pl.col("count_PGK2") <= 3)
            & (pl.col("count_PGK2_with_inhibitor") == 0)
            & (pl.col("count_NTC_evidence") == 0)
            & (pl.col("historic_hits") < 5)
        ),
        args.low_target_pool_size,
        args.seed + 1,
    )
    pairs = make_pairs(
        positives,
        counter_pool,
        low_pool,
        args.local_negatives_per_positive,
        args.global_negatives_per_positive,
        args.seed,
    )
    if not pairs:
        raise ValueError("No unambiguous ranking pairs were generated")
    pair_frame = pl.DataFrame(pairs)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_name = "pgk2_ranking_pairs.parquet" if args.max_positives is None and args.positive_offset == 0 else f"pgk2_ranking_pairs_offset{args.positive_offset}_n{positives.height}.parquet"
    output = args.output_dir / output_name
    pair_frame.write_parquet(output, compression="zstd")
    report = {
        "target": "PGK2",
        "strict_positives": {
            "before_supplement_recheck": positives_before_supplement.height,
            "after_supplement_recheck": positives_after_supplement_recheck.height,
            "used": positives.height,
            "positive_offset": args.positive_offset,
            "removed_by_NTC_supplement": positives_before_supplement.height - positives_after_supplement_recheck.height,
        },
        "candidate_panel_smiles_excluded": len(excluded),
        "pools": {"observed_counter_evidence": counter_pool.height, "low_target_evidence": low_pool.height},
        "pairs": {
            "rows": pair_frame.height,
            "by_type": pair_frame.group_by("pair_type").len().sort("pair_type").to_dicts(),
            "weight_sum": float(pair_frame.get_column("pair_weight").sum()),
        },
        "policy": {
            "missing_control_interpretation": "absence is missing evidence, not positive specificity evidence",
            "strong_pair": "target dominance with an observed counter-control signal in the lower-ranked compound",
            "weaker_pair": "target-only dominance, weight 0.35",
            "ntc_supplement": "max(original, supplement) used as observed NTC evidence",
        },
        "output": str(output),
        "elapsed_seconds": time.perf_counter() - started,
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
