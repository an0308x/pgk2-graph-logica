#!/usr/bin/env python3
"""Build matched DEL ranking pairs against never-observed OpenDEL negatives.

The challenge data documentation states that the selection file lists every
compound seen in the sequencer at least once, and that "all other molecules
from the OpenDEL enumerated library were not detected as binders and can be
considered as inactive". The enumerated library holds 898,311,036 compounds
against 7,487,570 observed structures, so roughly 890.8M organizer-sanctioned
negatives exist that earlier pair builders never touched.

Every previous pair set drew its negatives from inside the selection file, so
the learned contrast was "observed strongly versus observed weakly". The ASMS
panel is instead ~400K arbitrary purchasable compounds of which about 0.1%
inhibit PGK2, which looks far more like "mostly non-binders". This builder
therefore contrasts each orthosteric anchor with close structural analogs that
were never observed at all.

Negatives are matched, not random: a negative shares its library and two of
three building blocks with its anchor, so the pair differs at one synthetic
position. Building blocks are read from the compound ID, whose format is
documented as library-BB1-BB2-BB3.

Anchor selection follows the BCM-recommended orthosteric criteria from the
challenge data wiki:

    count_PGK2 >= 3
    count_PGK2_with_inhibitor < 0.1 * count_PGK2
    count_NTC == 0
    historic_hits < 5

Never-observed status is evidence of absence from the sequencer, not a
measured inactive. Sequencing depth is finite, so some negatives are false
negatives. Pair weights are DEL-derived ranking preferences, never ASMS or
kinase-assay labels.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import polars as pl
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

RDLogger.DisableLog("rdApp.*")
FP_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

# Shared building-block pair -> the synthetic position that varies.
KEY_SPECS = (("k12", "BB3"), ("k13", "BB2"), ("k23", "BB1"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, default=ROOT / "PGK2_selection.parquet")
    parser.add_argument(
        "--ntc-supplement", type=Path, default=ROOT / "PGK2_NTC_supplement.parquet"
    )
    parser.add_argument(
        "--enumeration-dir",
        type=Path,
        default=ROOT / "OpeDELLibrary",
        help="Directory holding qDOS*.enumeration.parquet files.",
    )
    parser.add_argument(
        "--candidate-panels", type=Path, default=ROOT / "DREAM_challenge_2026/Val-Test-set.zip"
    )
    parser.add_argument(
        "--libraries",
        nargs="*",
        default=None,
        help="Restrict to these library names; default is every enumeration file found.",
    )
    parser.add_argument("--min-target-count", type=int, default=3)
    parser.add_argument("--max-inhibitor-ratio", type=float, default=0.1)
    parser.add_argument("--max-historic-hits", type=int, default=5)
    parser.add_argument(
        "--max-anchors",
        type=int,
        default=0,
        help="Bound the anchor set for a smoke run; 0 uses every eligible anchor.",
    )
    parser.add_argument("--negatives-per-anchor", type=int, default=3)
    parser.add_argument(
        "--min-tanimoto",
        type=float,
        default=0.0,
        help="Record-only by default; sharing two building blocks already bounds similarity.",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "artifacts" / "pgk2_unobserved_negative_pairs"
    )
    return parser.parse_args()


def candidate_smiles(path: Path) -> set[str]:
    """Every blinded validation/test SMILES, which must never enter training."""
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
    """Deduplicate by structure: sum counts, Stouffer-combine z, max historic hits."""
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
            pl.max_horizontal("count_NTC", "count_NTC_supplement").alias("count_NTC_evidence")
        )
        .collect(engine="streaming")
    )


def with_building_blocks(frame: pl.LazyFrame) -> pl.LazyFrame:
    """Split library-BB1-BB2-BB3 and derive the three shared-pair keys."""
    parts = (
        pl.col("compound")
        .str.splitn("-", 4)
        .struct.rename_fields(["lib", "b1", "b2", "b3"])
        .alias("parts")
    )
    return (
        frame.with_columns(parts)
        .unnest("parts")
        .filter(
            pl.col("lib").is_not_null()
            & pl.col("b1").is_not_null()
            & pl.col("b2").is_not_null()
            & pl.col("b3").is_not_null()
        )
        .with_columns(
            (pl.col("lib") + ":" + pl.col("b1") + ":" + pl.col("b2")).alias("k12"),
            (pl.col("lib") + ":" + pl.col("b1") + ":" + pl.col("b3")).alias("k13"),
            (pl.col("lib") + ":" + pl.col("b2") + ":" + pl.col("b3")).alias("k23"),
        )
    )


def select_anchors(
    aggregated: pl.DataFrame, libraries: list[str] | None, args: argparse.Namespace
) -> pl.DataFrame:
    anchors = aggregated.filter(
        (pl.col("count_PGK2") >= args.min_target_count)
        & (
            pl.col("count_PGK2_with_inhibitor")
            < args.max_inhibitor_ratio * pl.col("count_PGK2")
        )
        & (pl.col("count_NTC_evidence") == 0)
        & (pl.col("historic_hits") < args.max_historic_hits)
    )
    anchors = with_building_blocks(anchors.lazy()).collect()
    # Restrict before capping, so --max-anchors bounds the libraries actually scanned.
    if libraries:
        anchors = anchors.filter(pl.col("lib").is_in(libraries))
    if args.max_anchors:
        anchors = anchors.sort("count_PGK2", descending=True).head(args.max_anchors)
    return anchors


def sample_library_negatives(
    enumeration: Path,
    anchors: pl.DataFrame,
    observed: pl.DataFrame,
    excluded: set[str],
    args: argparse.Namespace,
) -> pl.DataFrame:
    """Never-observed 2-of-3 building-block analogs of this library's anchors."""
    library = enumeration.name.split(".")[0]
    local = anchors.filter(pl.col("lib") == library)
    if local.is_empty():
        return pl.DataFrame()

    enumerated = with_building_blocks(pl.scan_parquet(enumeration))
    collected: list[pl.DataFrame] = []
    for key, varied in KEY_SPECS:
        anchor_keys = local.select(
            pl.col(key).alias("join_key"),
            pl.col("SMILES").alias("anchor_smiles"),
            pl.col("compound").alias("anchor_compound"),
        ).unique(subset=["join_key", "anchor_smiles"])
        if anchor_keys.is_empty():
            continue
        matched = (
            enumerated.select("compound", "SMILES", pl.col(key).alias("join_key"))
            .join(anchor_keys.lazy(), on="join_key", how="inner")
            .filter(pl.col("SMILES") != pl.col("anchor_smiles"))
            .with_columns(pl.lit(varied).alias("varied_position"))
            .collect(engine="streaming")
        )
        if not matched.is_empty():
            collected.append(matched)
    if not collected:
        return pl.DataFrame()

    candidates = pl.concat(collected, how="vertical")
    # Never observed in the sequencer, and never part of the blinded panels.
    candidates = (
        candidates.join(observed, on="SMILES", how="anti")
        .filter(~pl.col("SMILES").is_in(excluded))
        .unique(subset=["anchor_smiles", "SMILES"])
    )
    if candidates.is_empty():
        return pl.DataFrame()

    # Deterministic bounded sample per anchor.
    return (
        candidates.with_columns(
            (pl.col("anchor_smiles") + "|" + pl.col("SMILES")).hash(seed=args.seed).alias("_h")
        )
        .sort(["anchor_smiles", "_h"])
        .group_by("anchor_smiles", maintain_order=True)
        .head(args.negatives_per_anchor)
        .drop("_h")
    )


def tanimoto(first: str, second: str) -> float | None:
    left, right = Chem.MolFromSmiles(first), Chem.MolFromSmiles(second)
    if left is None or right is None:
        return None
    from rdkit import DataStructs

    return float(
        DataStructs.TanimotoSimilarity(
            FP_GENERATOR.GetFingerprint(left), FP_GENERATOR.GetFingerprint(right)
        )
    )


def quantiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    return {
        name: float(np.quantile(values, probability))
        for name, probability in (("q10", 0.1), ("median", 0.5), ("q90", 0.9))
    }


def main() -> int:
    args = parse_args()
    started = time.perf_counter()
    for path in (args.selection, args.ntc_supplement, args.candidate_panels):
        if not path.exists():
            raise FileNotFoundError(path)
    enumerations = sorted(args.enumeration_dir.glob("qDOS*.enumeration.parquet"))
    if args.libraries:
        wanted = set(args.libraries)
        enumerations = [p for p in enumerations if p.name.split(".")[0] in wanted]
    if not enumerations:
        raise FileNotFoundError(f"No enumeration parquet files under {args.enumeration_dir}")

    excluded = candidate_smiles(args.candidate_panels)
    aggregated = aggregate_selection(args.selection, args.ntc_supplement)
    observed = aggregated.select("SMILES")
    library_names = [path.name.split(".")[0] for path in enumerations]
    anchors = select_anchors(aggregated, library_names, args)
    print(
        f"observed structures={len(aggregated):,} eligible anchors={len(anchors):,} "
        f"panel SMILES excluded={len(excluded):,}",
        flush=True,
    )

    per_library: dict[str, int] = {}
    frames: list[pl.DataFrame] = []
    for enumeration in enumerations:
        library = enumeration.name.split(".")[0]
        elapsed = time.perf_counter()
        sampled = sample_library_negatives(enumeration, anchors, observed, excluded, args)
        per_library[library] = 0 if sampled.is_empty() else len(sampled)
        print(
            f"  {library:10s} negatives={per_library[library]:>7,} "
            f"({time.perf_counter() - elapsed:.1f}s)",
            flush=True,
        )
        if not sampled.is_empty():
            frames.append(sampled)

    if not frames:
        raise RuntimeError("No never-observed matched negatives were found")
    pairs = pl.concat(frames, how="vertical")

    anchor_evidence = anchors.select(
        pl.col("SMILES").alias("anchor_smiles"),
        pl.col("count_PGK2").alias("positive_count_PGK2"),
        pl.col("count_PGK2_with_inhibitor").alias("positive_count_inhibitor"),
        pl.col("count_NTC_evidence").alias("positive_count_NTC_evidence"),
        pl.col("historic_hits").alias("positive_historic_hits"),
        pl.col("zscore_PGK2").alias("positive_zscore_PGK2"),
        pl.col("lib").alias("library"),
    )
    pairs = pairs.join(anchor_evidence, on="anchor_smiles", how="left")

    similarities = [
        tanimoto(row["anchor_smiles"], row["SMILES"])
        for row in pairs.select("anchor_smiles", "SMILES").iter_rows(named=True)
    ]
    pairs = pairs.with_columns(
        pl.Series("tanimoto_similarity", similarities, dtype=pl.Float64)
    ).filter(pl.col("tanimoto_similarity").is_not_null())
    if args.min_tanimoto > 0:
        pairs = pairs.filter(pl.col("tanimoto_similarity") >= args.min_tanimoto)

    pairs = pairs.drop("join_key").rename(
        {"anchor_smiles": "positive_smiles", "SMILES": "negative_smiles", "compound": "negative_compound"}
    ).with_columns(
        # Confident anchors carry more weight; a never-observed negative is
        # absence of evidence, so weights stay bounded and below 1.0.
        (
            pl.col("positive_count_PGK2").cast(pl.Float64).log1p()
            / pl.col("positive_count_PGK2").cast(pl.Float64).log1p().max()
        ).alias("pair_weight"),
        pl.lit("unobserved_matched_analog").alias("pair_type"),
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "pgk2_unobserved_negative_pairs.parquet"
    pairs.write_parquet(output)

    report = {
        "target": "PGK2",
        "uniprot": "P07205",
        "observed_structures": len(aggregated),
        "eligible_anchors": len(anchors),
        "anchors_with_negatives": int(pairs["positive_smiles"].n_unique()),
        "pairs": len(pairs),
        "negatives_per_library": per_library,
        "varied_position_counts": {
            row["varied_position"]: row["count"]
            for row in pairs["varied_position"].value_counts().to_dicts()
        },
        "tanimoto_quantiles": quantiles(pairs["tanimoto_similarity"].to_list()),
        "anchor_criteria": {
            "count_PGK2_min": args.min_target_count,
            "inhibitor_ratio_max": args.max_inhibitor_ratio,
            "count_NTC_evidence": 0,
            "historic_hits_max": args.max_historic_hits,
            "source": "BCM-recommended orthosteric criteria, challenge data wiki 641134",
        },
        "blind_panel_smiles_excluded": len(excluded),
        "output": str(output),
        "elapsed_seconds": time.perf_counter() - started,
        "limitations": [
            "A never-observed compound is absence of sequencing evidence, not a measured inactive.",
            "Pair weights are DEL-derived ranking preferences, not ASMS or kinase-assay labels.",
            "Deduplication keeps one compound ID per structure, so an anchor's building-block decomposition is one of possibly several.",
        ],
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
