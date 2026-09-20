#!/usr/bin/env python3
"""Build the leakage-safe canonical PGK2 DEL index used by full-data models.

The expensive RDKit pass is written in restartable Parquet parts.  Canonical
SMILES are then aggregated once, candidate-panel molecules are excluded by
canonical identity, and scaffold-grouped internal splits are assigned without
using challenge feedback.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import polars as pl
import pyarrow.parquet as pq
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold


ROOT = Path(__file__).resolve().parents[1]
MAIN_COLUMNS = (
    "compound",
    "SMILES",
    "count_PGK2",
    "count_PGK2_with_inhibitor",
    "count_NTC",
    "zscore_PGK2",
    "zscore_PGK2_with_inhibitor",
    "zscore_NTC",
    "historic_hits",
)
SUPPLEMENT_COLUMNS = ("compound", "SMILES", "count_NTC", "zscore_NTC")
PANEL_MEMBERS = (
    "Val-Test-set/PGK2_Validation_split.csv",
    "Val-Test-set/PGK2_Test_split.csv",
)


@dataclass(frozen=True)
class CanonicalRecord:
    canonical_smiles: str
    molecule_hash: str
    murcko_scaffold: str
    scaffold_split: str
    sample_bucket: int
    status: str
    error: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, default=ROOT / "PGK2_selection.parquet")
    parser.add_argument(
        "--ntc-supplement", type=Path, default=ROOT / "PGK2_NTC_supplement.parquet"
    )
    parser.add_argument(
        "--candidate-panels",
        type=Path,
        default=ROOT / "DREAM_challenge_2026" / "Val-Test-set.zip",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "artifacts" / "pgk2_full_canonical_index",
    )
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=100_000)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def stable_bucket(value: str, modulus: int) -> int:
    if modulus <= 0:
        raise ValueError("modulus must be positive")
    digest = hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") % modulus


def scaffold_split(scaffold: str, canonical_smiles: str) -> str:
    """Assign every scaffold to exactly one deterministic 80/10/10 split."""
    key = f"scaffold:{scaffold}" if scaffold else f"acyclic:{canonical_smiles}"
    bucket = stable_bucket(key, 1000)
    return "train" if bucket < 800 else "dev" if bucket < 900 else "holdout"


def canonicalize_structure(raw_smiles: Any) -> CanonicalRecord:
    smiles = "" if raw_smiles is None else str(raw_smiles).strip()
    if not smiles:
        return CanonicalRecord("", "", "", "", -1, "blank_smiles", "")
    try:
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            return CanonicalRecord("", "", "", "", -1, "parse_failure", "RDKit parse failed")
        canonical = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
        scaffold_molecule = MurckoScaffold.GetScaffoldForMol(molecule)
        scaffold = (
            Chem.MolToSmiles(scaffold_molecule, canonical=True, isomericSmiles=True)
            if scaffold_molecule.GetNumAtoms()
            else ""
        )
        molecule_hash = hashlib.blake2b(
            canonical.encode("utf-8"), digest_size=16
        ).hexdigest()
        return CanonicalRecord(
            canonical_smiles=canonical,
            molecule_hash=molecule_hash,
            murcko_scaffold=scaffold,
            scaffold_split=scaffold_split(scaffold, canonical),
            sample_bucket=stable_bucket(f"sample:{canonical}", 1000),
            status="ok",
            error="",
        )
    except Exception as exc:
        return CanonicalRecord(
            "", "", "", "", -1, "canonicalization_failure", f"{type(exc).__name__}: {exc}"[:1000]
        )


def panel_smiles(path: Path) -> list[str]:
    values: list[str] = []
    with zipfile.ZipFile(path) as archive:
        for member in PANEL_MEMBERS:
            with archive.open(member) as handle:
                values.extend(
                    str(row["SMILES"]).strip()
                    for row in csv.DictReader(line.decode("utf-8") for line in handle)
                    if str(row["SMILES"]).strip()
                )
    return values


def initialize_canonical_worker() -> None:
    """Silence expected RDKit parse diagnostics independently in each process."""
    RDLogger.DisableLog("rdApp.*")


def canonicalize_values(values: Iterable[Any], workers: int) -> list[CanonicalRecord]:
    # RDKit parsing and Murcko extraction do not scale reliably through Python
    # threads on McCleary.  Processes make --workers correspond to real CPU use.
    with ProcessPoolExecutor(
        max_workers=workers, initializer=initialize_canonical_worker
    ) as executor:
        return list(executor.map(canonicalize_structure, values, chunksize=256))


def validate_columns(path: Path, required: tuple[str, ...]) -> None:
    names = set(pq.ParquetFile(path).schema_arrow.names)
    missing = set(required) - names
    if missing:
        raise ValueError(f"{path} is missing columns {sorted(missing)}")


def canonicalize_parquet_parts(
    source: Path,
    columns: tuple[str, ...],
    prefix: str,
    raw_dir: Path,
    candidate_canonical: set[str],
    workers: int,
    batch_size: int,
    resume: bool,
) -> list[Path]:
    parquet = pq.ParquetFile(source)
    outputs: list[Path] = []
    for part_index, batch in enumerate(
        parquet.iter_batches(batch_size=batch_size, columns=list(columns))
    ):
        output = raw_dir / f"{prefix}-{part_index:05d}.parquet"
        outputs.append(output)
        if output.exists():
            if not resume:
                raise FileExistsError(f"Refusing to overwrite {output}; pass --resume")
            continue
        payload = batch.to_pydict()
        records = canonicalize_values(payload["SMILES"], workers)
        frame = pl.DataFrame(payload).with_columns(
            pl.Series("canonical_smiles", [record.canonical_smiles for record in records]),
            pl.Series("molecule_hash", [record.molecule_hash for record in records]),
            pl.Series("murcko_scaffold", [record.murcko_scaffold for record in records]),
            pl.Series("scaffold_split", [record.scaffold_split for record in records]),
            pl.Series("sample_bucket", [record.sample_bucket for record in records], dtype=pl.Int32),
            pl.Series("canonical_status", [record.status for record in records]),
            pl.Series("canonical_error", [record.error for record in records]),
            pl.Series(
                "candidate_panel_overlap",
                [
                    bool(record.canonical_smiles)
                    and record.canonical_smiles in candidate_canonical
                    for record in records
                ],
            ),
        )
        temporary = output.with_suffix(".parquet.tmp")
        frame.write_parquet(temporary, compression="zstd", statistics=True)
        temporary.replace(output)
        print(
            json.dumps(
                {
                    "source": prefix,
                    "part": part_index,
                    "rows": frame.height,
                    "ok": int((frame["canonical_status"] == "ok").sum()),
                    "candidate_overlap": int(frame["candidate_panel_overlap"].sum()),
                }
            ),
            flush=True,
        )
    return outputs


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    if args.workers <= 0 or args.batch_size <= 0:
        raise ValueError("workers and batch-size must be positive")
    for path in (args.selection, args.ntc_supplement, args.candidate_panels):
        if not path.exists():
            raise FileNotFoundError(path)
    validate_columns(args.selection, MAIN_COLUMNS)
    validate_columns(args.ntc_supplement, SUPPLEMENT_COLUMNS)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = args.output_dir / "canonical_parts"
    raw_dir.mkdir(parents=True, exist_ok=True)
    index_path = args.output_dir / "canonical_training_index.parquet"
    report_path = args.output_dir / "report.json"
    if index_path.exists() or report_path.exists():
        raise FileExistsError(
            f"Completed outputs already exist in {args.output_dir}; use a new directory"
        )

    RDLogger.DisableLog("rdApp.*")
    started = time.perf_counter()
    candidate_records = canonicalize_values(panel_smiles(args.candidate_panels), args.workers)
    candidate_canonical = {
        record.canonical_smiles for record in candidate_records if record.status == "ok"
    }
    main_parts = canonicalize_parquet_parts(
        args.selection,
        MAIN_COLUMNS,
        "selection",
        raw_dir,
        candidate_canonical,
        args.workers,
        args.batch_size,
        args.resume,
    )
    supplement_parts = canonicalize_parquet_parts(
        args.ntc_supplement,
        SUPPLEMENT_COLUMNS,
        "ntc-supplement",
        raw_dir,
        candidate_canonical,
        args.workers,
        args.batch_size,
        args.resume,
    )

    main_scan = pl.scan_parquet([str(path) for path in main_parts])
    main_valid = main_scan.filter(
        (pl.col("canonical_status") == "ok") & ~pl.col("candidate_panel_overlap")
    )
    main_aggregate = main_valid.group_by("canonical_smiles").agg(
        pl.col("molecule_hash").first(),
        pl.col("murcko_scaffold").first(),
        pl.col("scaffold_split").first(),
        pl.col("sample_bucket").first(),
        pl.col("compound").first().alias("representative_compound"),
        pl.len().alias("source_rows"),
        pl.col("count_PGK2").sum(),
        pl.col("count_PGK2_with_inhibitor").sum(),
        pl.col("count_NTC").sum().alias("count_NTC_selection"),
        (pl.col("zscore_PGK2").sum() / pl.len().cast(pl.Float64).sqrt()).alias(
            "zscore_PGK2"
        ),
        (
            pl.col("zscore_PGK2_with_inhibitor").sum()
            / pl.len().cast(pl.Float64).sqrt()
        ).alias("zscore_PGK2_with_inhibitor"),
        (pl.col("zscore_NTC").sum() / pl.len().cast(pl.Float64).sqrt()).alias(
            "zscore_NTC_selection"
        ),
        pl.col("historic_hits").max(),
    )
    supplement_scan = pl.scan_parquet([str(path) for path in supplement_parts])
    supplement_aggregate = (
        supplement_scan.filter(
            (pl.col("canonical_status") == "ok") & ~pl.col("candidate_panel_overlap")
        )
        .group_by("canonical_smiles")
        .agg(
            pl.len().alias("ntc_supplement_rows"),
            pl.col("count_NTC").sum().alias("count_NTC_supplement"),
            (pl.col("zscore_NTC").sum() / pl.len().cast(pl.Float64).sqrt()).alias(
                "zscore_NTC_supplement"
            ),
        )
    )
    index = (
        main_aggregate.join(supplement_aggregate, on="canonical_smiles", how="left")
        .with_columns(
            pl.col("ntc_supplement_rows").fill_null(0),
            pl.col("count_NTC_supplement").fill_null(0),
            pl.col("zscore_NTC_supplement").fill_null(0.0),
        )
        .with_columns(
            (
                (pl.col("count_PGK2") >= 2)
                | (pl.col("count_PGK2_with_inhibitor") > 0)
                | (pl.col("count_NTC_selection") > 0)
                | (pl.col("count_NTC_supplement") > 0)
                | (pl.col("historic_hits") > 0)
            ).alias("informative")
        )
        .sort("molecule_hash")
        .collect(engine="streaming")
        .with_row_index("molecule_id")
    )
    temporary_index = index_path.with_suffix(".parquet.tmp")
    index.write_parquet(
        temporary_index, compression="zstd", statistics=True, row_group_size=100_000
    )
    temporary_index.replace(index_path)

    main_status = (
        main_scan.group_by("canonical_status").agg(pl.len().alias("rows")).collect()
    )
    supplement_status = (
        supplement_scan.group_by("canonical_status").agg(pl.len().alias("rows")).collect()
    )
    report = {
        "definition": "Leakage-safe canonical full-data PGK2 DEL index",
        "inputs": {
            "selection": str(args.selection),
            "selection_sha256": sha256(args.selection),
            "ntc_supplement": str(args.ntc_supplement),
            "ntc_supplement_sha256": sha256(args.ntc_supplement),
            "candidate_panels": str(args.candidate_panels),
            "candidate_panels_sha256": sha256(args.candidate_panels),
        },
        "candidate_panel_canonical_smiles": len(candidate_canonical),
        "selection_status": dict(zip(main_status["canonical_status"], main_status["rows"])),
        "supplement_status": dict(
            zip(supplement_status["canonical_status"], supplement_status["rows"])
        ),
        "candidate_overlap_rows": int(
            main_scan.select(pl.col("candidate_panel_overlap").sum()).collect().item()
        ),
        "unique_training_molecules": index.height,
        "informative_molecules": int(index["informative"].sum()),
        "split_counts": {
            str(row["scaffold_split"]): int(row["len"])
            for row in index.group_by("scaffold_split").len().to_dicts()
        },
        "ablation_subset_definition": "informative OR sample_bucket < 14",
        "ablation_subset_molecules": int(
            (index["informative"] | (index["sample_bucket"] < 14)).sum()
        ),
        "output": str(index_path),
        "elapsed_seconds": time.perf_counter() - started,
        "notes": [
            "Challenge validation and test molecules are excluded by canonical SMILES.",
            "NTC supplement measurements remain separate from selection NTC measurements.",
            "Scaffold splits and the low-evidence ablation sample are deterministic.",
        ],
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
