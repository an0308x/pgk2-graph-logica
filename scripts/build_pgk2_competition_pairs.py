#!/usr/bin/env python3
"""Build DEL pairs that isolate PGK2 active-site competition.

The preferred molecule is target-enriched, inhibitor-depleted, low-background,
and low-promiscuity.  Its comparator is also target-enriched and low-background
but retains signal in the active-site-inhibitor condition.  Requiring both
molecules to bind PGK2 makes the pair primarily about competition rather than
generic target enrichment.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import polars as pl
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem


ROOT = Path(__file__).resolve().parents[1]
FP_GENERATOR = AllChem.GetMorganGenerator(radius=2, fpSize=2048)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, default=ROOT / "PGK2_selection.parquet")
    parser.add_argument(
        "--ntc-supplement", type=Path, default=ROOT / "PGK2_NTC_supplement.parquet"
    )
    parser.add_argument(
        "--candidate-panels",
        type=Path,
        default=ROOT / "DREAM_challenge_2026/Val-Test-set.zip",
    )
    parser.add_argument("--minimum-target-count", type=int, default=4)
    parser.add_argument("--sensitive-ratio", type=float, default=0.10)
    parser.add_argument("--retained-ratio", type=float, default=0.50)
    parser.add_argument("--maximum-ntc-ratio", type=float, default=0.10)
    parser.add_argument("--maximum-historic-hits", type=int, default=4)
    parser.add_argument("--minimum-similarity", type=float, default=0.60)
    parser.add_argument(
        "--fallback-minimum-similarity",
        type=float,
        default=0.50,
        help="Same-library fingerprint fallback for retained binders lacking a two-block match.",
    )
    parser.add_argument("--maximum-target-fold", type=float, default=2.0)
    parser.add_argument("--pairs-per-negative", type=int, default=3)
    parser.add_argument("--pseudocount", type=float, default=0.5)
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "artifacts/pgk2_competition_pairs"
    )
    return parser.parse_args()


def candidate_smiles(path: Path) -> set[str]:
    members = (
        "Val-Test-set/PGK2_Validation_split.csv",
        "Val-Test-set/PGK2_Test_split.csv",
    )
    result: set[str] = set()
    with zipfile.ZipFile(path) as archive:
        for member in members:
            with archive.open(member) as handle:
                result.update(
                    row["SMILES"] for row in csv.DictReader(line.decode() for line in handle)
                )
    return result


def aggregate_selection(selection: Path, ntc_supplement: Path) -> pl.DataFrame:
    counts = ["count_PGK2", "count_PGK2_with_inhibitor", "count_NTC"]
    zscores = ["zscore_PGK2", "zscore_PGK2_with_inhibitor", "zscore_NTC"]
    selection_frame = (
        pl.scan_parquet(selection)
        .filter(pl.col("SMILES").is_not_null() & (pl.col("SMILES") != ""))
        .group_by("SMILES")
        .agg(
            pl.col("compound").first().alias("compound"),
            pl.len().alias("source_rows"),
            *[pl.col(column).sum().alias(column) for column in counts],
            *[
                (pl.col(column).sum() / pl.len().cast(pl.Float64).sqrt()).alias(column)
                for column in zscores
            ],
            pl.col("historic_hits").max().alias("historic_hits"),
        )
    )
    supplement = (
        pl.scan_parquet(ntc_supplement)
        .filter(pl.col("SMILES").is_not_null() & (pl.col("SMILES") != ""))
        .group_by("SMILES")
        .agg(pl.col("count_NTC").sum().alias("count_NTC_supplement"))
    )
    return (
        selection_frame.join(supplement, on="SMILES", how="left")
        .with_columns(pl.col("count_NTC_supplement").fill_null(0))
        .with_columns(
            pl.max_horizontal("count_NTC", "count_NTC_supplement").alias(
                "count_NTC_evidence"
            )
        )
        .collect(engine="streaming")
    )


def building_block_keys(compound: str) -> tuple[str, str, str] | None:
    parts = compound.split("-")
    if len(parts) != 4 or any(not value for value in parts):
        return None
    library, first, second, third = parts
    return (
        f"{library}:{first}:{second}",
        f"{library}:{first}:{third}",
        f"{library}:{second}:{third}",
    )


def library_name(compound: str) -> str | None:
    parts = compound.split("-")
    return parts[0] if len(parts) == 4 and all(parts) else None


def competition_log_ratio(target: int, inhibitor: int, pseudocount: float = 0.5) -> float:
    """Shrink a sparse target/inhibitor count ratio toward zero evidence."""

    if target < 0 or inhibitor < 0 or pseudocount <= 0:
        raise ValueError("Counts must be non-negative and pseudocount must be positive")
    return math.log((target + pseudocount) / (inhibitor + pseudocount))


def fingerprint(smiles: str):
    molecule = Chem.MolFromSmiles(smiles)
    return FP_GENERATOR.GetFingerprint(molecule) if molecule is not None else None


def quantiles(values: list[float]) -> dict[str, float]:
    return {
        name: float(np.quantile(values, probability))
        for name, probability in (("q10", 0.1), ("median", 0.5), ("q90", 0.9))
    }


def main() -> int:
    args = parse_args()
    if not 0 < args.sensitive_ratio < args.retained_ratio:
        raise ValueError("Require 0 < sensitive-ratio < retained-ratio")
    if not 0 <= args.maximum_ntc_ratio < 1:
        raise ValueError("maximum-ntc-ratio must be in [0, 1)")
    if (
        not 0 <= args.fallback_minimum_similarity <= args.minimum_similarity <= 1
        or args.maximum_target_fold < 1
    ):
        raise ValueError("Invalid similarity or target-fold bound")
    if args.pairs_per_negative <= 0 or args.minimum_target_count <= 0:
        raise ValueError("Pair and target-count bounds must be positive")
    for path in (args.selection, args.ntc_supplement, args.candidate_panels):
        if not path.exists():
            raise FileNotFoundError(path)

    started = time.perf_counter()
    excluded = candidate_smiles(args.candidate_panels)
    frame = aggregate_selection(args.selection, args.ntc_supplement).filter(
        ~pl.col("SMILES").is_in(excluded)
    )
    common = (
        (pl.col("count_PGK2") >= args.minimum_target_count)
        & (pl.col("count_NTC_evidence") < args.maximum_ntc_ratio * pl.col("count_PGK2"))
        & (pl.col("historic_hits") <= args.maximum_historic_hits)
    )
    positives = frame.filter(
        common
        & (
            pl.col("count_PGK2_with_inhibitor")
            < args.sensitive_ratio * pl.col("count_PGK2")
        )
    )
    negatives = frame.filter(
        common
        & (
            pl.col("count_PGK2_with_inhibitor")
            >= args.retained_ratio * pl.col("count_PGK2")
        )
    )
    if positives.is_empty() or negatives.is_empty():
        raise ValueError("Competition-sensitive positives or retained-signal negatives are empty")

    positive_rows = positives.to_dicts()
    negative_rows = negatives.to_dicts()
    positives_by_key: dict[str, list[dict[str, object]]] = defaultdict(list)
    positives_by_library: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in positive_rows:
        keys = building_block_keys(str(row["compound"]))
        if keys is not None:
            for key in keys:
                positives_by_key[key].append(row)
        library = library_name(str(row["compound"]))
        if library is not None:
            positives_by_library[library].append(row)

    fp_cache: dict[str, object | None] = {}

    def cached_fp(smiles: str):
        if smiles not in fp_cache:
            fp_cache[smiles] = fingerprint(smiles)
        return fp_cache[smiles]

    pairs: list[dict[str, object]] = []
    covered_negatives: set[str] = set()
    for negative in sorted(negative_rows, key=lambda row: str(row["SMILES"])):
        keys = building_block_keys(str(negative["compound"]))
        negative_fp = cached_fp(str(negative["SMILES"]))
        if keys is None or negative_fp is None:
            continue
        candidates: dict[str, tuple[dict[str, object], float, str, float, float]] = {}
        negative_target = int(negative["count_PGK2"])
        negative_competition = competition_log_ratio(
            negative_target,
            int(negative["count_PGK2_with_inhibitor"]),
            args.pseudocount,
        )
        for key in keys:
            for positive in positives_by_key.get(key, []):
                positive_smiles = str(positive["SMILES"])
                if positive_smiles in candidates:
                    continue
                positive_target = int(positive["count_PGK2"])
                target_fold = max(
                    positive_target / negative_target, negative_target / positive_target
                )
                if target_fold > args.maximum_target_fold:
                    continue
                positive_fp = cached_fp(positive_smiles)
                if positive_fp is None:
                    continue
                similarity = float(
                    DataStructs.TanimotoSimilarity(positive_fp, negative_fp)
                )
                if similarity < args.minimum_similarity:
                    continue
                positive_competition = competition_log_ratio(
                    positive_target,
                    int(positive["count_PGK2_with_inhibitor"]),
                    args.pseudocount,
                )
                margin = positive_competition - negative_competition
                if margin <= 0:
                    continue
                candidates[positive_smiles] = (
                    positive,
                    similarity,
                    key,
                    target_fold,
                    margin,
                )
        selected = sorted(
            candidates.values(),
            key=lambda item: (-item[1], item[3], -item[4], str(item[0]["SMILES"])),
        )[: args.pairs_per_negative]
        for positive, similarity, key, target_fold, margin in selected:
            pair_weight = 1.0 + min(margin / math.log(10), 2.0)
            pairs.append(
                {
                    "positive_smiles": positive["SMILES"],
                    "negative_smiles": negative["SMILES"],
                    "pair_weight": pair_weight,
                    "pair_type": "target_matched_active_site_competition",
                    "shared_building_block_key": key,
                    "tanimoto_similarity": similarity,
                    "target_count_fold_difference": target_fold,
                    "competition_log_ratio_margin": margin,
                    "positive_count_PGK2": positive["count_PGK2"],
                    "negative_count_PGK2": negative["count_PGK2"],
                    "positive_count_inhibitor": positive["count_PGK2_with_inhibitor"],
                    "negative_count_inhibitor": negative["count_PGK2_with_inhibitor"],
                    "positive_count_NTC_evidence": positive["count_NTC_evidence"],
                    "negative_count_NTC_evidence": negative["count_NTC_evidence"],
                    "positive_historic_hits": positive["historic_hits"],
                    "negative_historic_hits": negative["historic_hits"],
                }
            )
            covered_negatives.add(str(negative["SMILES"]))

    # Recover additional target-matched comparisons without crossing DEL
    # libraries.  This lower-weight tier is used only when no two-building-
    # block match exists for a retained binder.
    for negative in sorted(negative_rows, key=lambda row: str(row["SMILES"])):
        negative_smiles = str(negative["SMILES"])
        if negative_smiles in covered_negatives:
            continue
        library = library_name(str(negative["compound"]))
        negative_fp = cached_fp(negative_smiles)
        if library is None or negative_fp is None:
            continue
        negative_target = int(negative["count_PGK2"])
        negative_competition = competition_log_ratio(
            negative_target,
            int(negative["count_PGK2_with_inhibitor"]),
            args.pseudocount,
        )
        fallback: list[tuple[dict[str, object], float, float, float]] = []
        for positive in positives_by_library.get(library, []):
            positive_target = int(positive["count_PGK2"])
            target_fold = max(
                positive_target / negative_target, negative_target / positive_target
            )
            if target_fold > args.maximum_target_fold:
                continue
            positive_fp = cached_fp(str(positive["SMILES"]))
            if positive_fp is None:
                continue
            similarity = float(DataStructs.TanimotoSimilarity(positive_fp, negative_fp))
            if similarity < args.fallback_minimum_similarity:
                continue
            positive_competition = competition_log_ratio(
                positive_target,
                int(positive["count_PGK2_with_inhibitor"]),
                args.pseudocount,
            )
            margin = positive_competition - negative_competition
            if margin > 0:
                fallback.append((positive, similarity, target_fold, margin))
        if not fallback:
            continue
        positive, similarity, target_fold, margin = min(
            fallback,
            key=lambda item: (-item[1], item[2], -item[3], str(item[0]["SMILES"])),
        )
        pairs.append(
            {
                "positive_smiles": positive["SMILES"],
                "negative_smiles": negative["SMILES"],
                "pair_weight": 0.5 * (1.0 + min(margin / math.log(10), 2.0)),
                "pair_type": "same_library_fingerprint_active_site_competition",
                "shared_building_block_key": "",
                "tanimoto_similarity": similarity,
                "target_count_fold_difference": target_fold,
                "competition_log_ratio_margin": margin,
                "positive_count_PGK2": positive["count_PGK2"],
                "negative_count_PGK2": negative["count_PGK2"],
                "positive_count_inhibitor": positive["count_PGK2_with_inhibitor"],
                "negative_count_inhibitor": negative["count_PGK2_with_inhibitor"],
                "positive_count_NTC_evidence": positive["count_NTC_evidence"],
                "negative_count_NTC_evidence": negative["count_NTC_evidence"],
                "positive_historic_hits": positive["historic_hits"],
                "negative_historic_hits": negative["historic_hits"],
            }
        )
        covered_negatives.add(negative_smiles)
    if not pairs:
        raise ValueError("No target-matched active-site competition pairs were generated")

    pair_frame = pl.DataFrame(pairs).unique(
        subset=["positive_smiles", "negative_smiles"], maintain_order=True
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pair_path = args.output_dir / "pgk2_competition_pairs.parquet"
    positive_path = args.output_dir / "competition_sensitive_anchors.parquet"
    negative_path = args.output_dir / "competition_retained_binders.parquet"
    pair_frame.write_parquet(pair_path, compression="zstd")
    positives.write_parquet(positive_path, compression="zstd")
    negatives.write_parquet(negative_path, compression="zstd")

    report = {
        "objective": "active-site competition among target-enriched low-background PGK2 binders",
        "thresholds": {
            "minimum_target_count": args.minimum_target_count,
            "competition_sensitive_inhibitor_over_target": f"<{args.sensitive_ratio}",
            "competition_retained_inhibitor_over_target": f">={args.retained_ratio}",
            "maximum_ntc_over_target": f"<{args.maximum_ntc_ratio}",
            "maximum_historic_hits": args.maximum_historic_hits,
            "minimum_morgan_tanimoto": args.minimum_similarity,
            "same_library_fallback_minimum_tanimoto": args.fallback_minimum_similarity,
            "maximum_target_count_fold": args.maximum_target_fold,
            "pairs_per_retained_binder": args.pairs_per_negative,
            "log_ratio_pseudocount": args.pseudocount,
        },
        "blind_panel_smiles_excluded": len(excluded),
        "competition_sensitive_anchors": positives.height,
        "competition_retained_binders": negatives.height,
        "covered_retained_binders": len(covered_negatives),
        "pairs": {
            "rows": pair_frame.height,
            "unique_positive_smiles": pair_frame["positive_smiles"].n_unique(),
            "unique_negative_smiles": pair_frame["negative_smiles"].n_unique(),
            "tanimoto_similarity": quantiles(
                pair_frame["tanimoto_similarity"].to_list()
            ),
            "target_count_fold_difference": quantiles(
                pair_frame["target_count_fold_difference"].to_list()
            ),
            "competition_log_ratio_margin": quantiles(
                pair_frame["competition_log_ratio_margin"].to_list()
            ),
            "pair_types": {
                row["pair_type"]: int(row["len"])
                for row in pair_frame.group_by("pair_type").len().to_dicts()
            },
        },
        "outputs": {
            "pairs": str(pair_path),
            "positive_anchors": str(positive_path),
            "retained_binders": str(negative_path),
        },
        "limitations": [
            "Count ratios assume the released DEL arms are comparable; no unreported library-size correction is invented.",
            "A zero inhibitor count is sparse observed evidence and is shrunk with a pseudocount, not treated as certainty.",
            "Pairs are DEL-derived active-site preferences, not kinase-assay labels.",
        ],
        "elapsed_seconds": time.perf_counter() - started,
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
