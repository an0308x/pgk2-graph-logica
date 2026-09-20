#!/usr/bin/env python3
"""Precompute restartable packed RDKit graph shards for the PGK2 DEL release.

The pilot samples rows across the complete release with a deterministic stride.
Each shard contains one metadata Parquet file and one packed NumPy archive;
millions of per-molecule files are deliberately avoided.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
import zipfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from logica_binding.graph_model import smiles_to_rdkit_3d_graph_with_quality


SELECTION_COLUMNS = (
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
PANEL_MEMBERS = (
    "Val-Test-set/PGK2_Validation_split.csv",
    "Val-Test-set/PGK2_Test_split.csv",
)


@dataclass(frozen=True)
class GraphResult:
    status: str
    canonical_smiles: str
    molecule_hash: str
    geometry_quality: str
    node_features: np.ndarray | None
    coordinates: np.ndarray | None
    edge_index: np.ndarray | None
    edge_features: np.ndarray | None
    fingerprint: np.ndarray | None
    error: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--selection",
        type=Path,
        default=ROOT / "PGK2_selection.parquet",
    )
    parser.add_argument(
        "--candidate-panels",
        type=Path,
        default=ROOT / "DREAM_challenge_2026" / "Val-Test-set.zip",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "artifacts" / "pgk2_rdkit_graph_pilot",
    )
    parser.add_argument("--max-molecules", type=int, default=100_000)
    parser.add_argument("--shard-size", type=int, default=25_000)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--spatial-cutoff", type=float, default=4.5)
    parser.add_argument("--optimization-max-iters", type=int, default=50)
    parser.add_argument("--fingerprint-bits", type=int, default=2048)
    parser.add_argument("--fingerprint-radius", type=int, default=2)
    return parser.parse_args()


def canonical_smiles(smiles: str) -> str | None:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return None
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)


def read_candidate_smiles(path: Path) -> set[str]:
    values: set[str] = set()
    with zipfile.ZipFile(path) as archive:
        for member in PANEL_MEMBERS:
            with archive.open(member) as handle:
                for row in csv.DictReader(line.decode("utf-8") for line in handle):
                    value = str(row["SMILES"]).strip()
                    if value:
                        values.add(value)
    return values


def canonicalize_many(values: list[str], workers: int) -> set[str]:
    with ThreadPoolExecutor(max_workers=workers) as executor:
        canonical = executor.map(canonical_smiles, values)
        return {value for value in canonical if value is not None}


def select_pilot_rows(path: Path, limit: int) -> tuple[list[dict[str, Any]], int, int]:
    schema = pl.scan_parquet(path).collect_schema()
    missing = set(SELECTION_COLUMNS) - set(schema.names())
    if missing:
        raise ValueError(f"Selection table is missing columns: {sorted(missing)}")
    total_rows = int(pl.scan_parquet(path).select(pl.len()).collect().item())
    stride = max(1, total_rows // limit)
    sampled = (
        pl.scan_parquet(path)
        .with_row_index("source_row")
        .filter(
            pl.col("SMILES").is_not_null()
            & (pl.col("SMILES").str.strip_chars().str.len_chars() > 0)
            & ((pl.col("source_row") % stride) == 0)
        )
        .select(("source_row",) + SELECTION_COLUMNS)
        .head(limit)
        .collect(engine="streaming")
    )
    return sampled.to_dicts(), total_rows, stride


def build_graph_task(
    task: tuple[str, int, float, int, int, int]
) -> GraphResult:
    smiles, seed, cutoff, max_iters, fp_radius, fp_bits = task
    try:
        canonical = canonical_smiles(smiles)
        if canonical is None:
            return GraphResult(
                "parse_failure", "", "", "", None, None, None, None, None,
                "RDKit could not parse the SMILES",
            )
        graph, quality = smiles_to_rdkit_3d_graph_with_quality(
            canonical,
            seed=seed,
            spatial_cutoff=cutoff,
            optimization_max_iters=max_iters,
            warn_on_fallback=False,
        )
        molecule = Chem.MolFromSmiles(canonical)
        if molecule is None:
            raise ValueError("Canonical SMILES unexpectedly failed to parse")
        generator = rdFingerprintGenerator.GetMorganGenerator(
            radius=fp_radius, fpSize=fp_bits
        )
        fingerprint = np.packbits(
            np.asarray(generator.GetFingerprintAsNumPy(molecule), dtype=np.uint8),
            bitorder="little",
        )
        return GraphResult(
            status="ok",
            canonical_smiles=canonical,
            molecule_hash=hashlib.blake2b(
                canonical.encode("utf-8"), digest_size=16
            ).hexdigest(),
            geometry_quality=quality,
            node_features=graph.node_features.numpy().astype(np.float32, copy=False),
            coordinates=graph.coordinates.numpy().astype(np.float32, copy=False),
            edge_index=graph.edge_index.numpy().astype(np.int32, copy=False),
            edge_features=graph.edge_features.numpy().astype(np.float32, copy=False),
            fingerprint=fingerprint,
            error="",
        )
    except Exception as exc:  # preserve coverage and record molecule-level failures
        return GraphResult(
            "graph_failure", "", "", "", None, None, None, None, None,
            f"{type(exc).__name__}: {exc}"[:1000],
        )


def empty_array(shape: tuple[int, ...], dtype: np.dtype[Any]) -> np.ndarray:
    return np.empty(shape, dtype=dtype)


def write_shard(
    output_dir: Path,
    shard_index: int,
    rows: list[dict[str, Any]],
    results: list[GraphResult],
    excluded_canonical: set[str],
    seen_canonical: set[str],
    fingerprint_bytes: int,
) -> dict[str, Any]:
    stem = f"shard-{shard_index:05d}"
    metadata_path = output_dir / f"{stem}-metadata.parquet"
    graph_path = output_dir / f"{stem}-graphs.npz"
    report_path = output_dir / f"{stem}-report.json"
    if metadata_path.exists() or graph_path.exists() or report_path.exists():
        raise FileExistsError(f"Refusing to overwrite an existing shard: {stem}")

    metadata: list[dict[str, Any]] = []
    nodes: list[np.ndarray] = []
    coordinates: list[np.ndarray] = []
    edge_indices: list[np.ndarray] = []
    edge_features: list[np.ndarray] = []
    fingerprints: list[np.ndarray] = []
    node_offsets = [0]
    edge_offsets = [0]
    status_counts: Counter[str] = Counter()
    geometry_counts: Counter[str] = Counter()

    for row, result in zip(rows, results, strict=True):
        status = result.status
        if status == "ok" and result.canonical_smiles in excluded_canonical:
            status = "candidate_panel_overlap"
        elif status == "ok" and result.canonical_smiles in seen_canonical:
            status = "canonical_duplicate"

        valid = status == "ok"
        if valid:
            assert result.node_features is not None
            assert result.coordinates is not None
            assert result.edge_index is not None
            assert result.edge_features is not None
            assert result.fingerprint is not None
            seen_canonical.add(result.canonical_smiles)
            nodes.append(result.node_features)
            coordinates.append(result.coordinates)
            edge_indices.append(result.edge_index)
            edge_features.append(result.edge_features)
            fingerprints.append(result.fingerprint)
            node_offsets.append(node_offsets[-1] + result.node_features.shape[0])
            edge_offsets.append(edge_offsets[-1] + result.edge_features.shape[0])
            geometry_counts[result.geometry_quality] += 1
        else:
            fingerprints.append(np.zeros(fingerprint_bytes, dtype=np.uint8))
            node_offsets.append(node_offsets[-1])
            edge_offsets.append(edge_offsets[-1])
        status_counts[status] += 1
        metadata.append(
            {
                **row,
                "canonical_smiles": result.canonical_smiles,
                "molecule_hash": result.molecule_hash,
                "status": status,
                "geometry_quality": result.geometry_quality if valid else "",
                "atom_count": 0 if not valid else int(result.node_features.shape[0]),
                "directed_edge_count": 0 if not valid else int(result.edge_features.shape[0]),
                "error": result.error,
            }
        )

    node_matrix = (
        np.concatenate(nodes, axis=0)
        if nodes
        else empty_array((0, 20), np.dtype(np.float32))
    )
    coordinate_matrix = (
        np.concatenate(coordinates, axis=0)
        if coordinates
        else empty_array((0, 3), np.dtype(np.float32))
    )
    edge_index_matrix = (
        np.concatenate(edge_indices, axis=1)
        if edge_indices
        else empty_array((2, 0), np.dtype(np.int32))
    )
    edge_feature_matrix = (
        np.concatenate(edge_features, axis=0)
        if edge_features
        else empty_array((0, 7), np.dtype(np.float32))
    )

    metadata_tmp = metadata_path.with_suffix(".parquet.tmp")
    pl.DataFrame(metadata).write_parquet(metadata_tmp, compression="zstd")
    metadata_tmp.replace(metadata_path)
    graph_tmp = graph_path.with_suffix(".npz.tmp")
    with graph_tmp.open("wb") as handle:
        np.savez_compressed(
            handle,
            node_features=node_matrix,
            coordinates=coordinate_matrix,
            edge_index=edge_index_matrix,
            edge_features=edge_feature_matrix,
            node_offsets=np.asarray(node_offsets, dtype=np.int64),
            edge_offsets=np.asarray(edge_offsets, dtype=np.int64),
            fingerprints=np.stack(fingerprints).astype(np.uint8, copy=False),
        )
    graph_tmp.replace(graph_path)
    report = {
        "shard": shard_index,
        "rows": len(rows),
        "successful_graphs": status_counts["ok"],
        "status_counts": dict(status_counts),
        "geometry_counts": dict(geometry_counts),
        "nodes": int(node_matrix.shape[0]),
        "directed_edges": int(edge_feature_matrix.shape[0]),
        "metadata_bytes": metadata_path.stat().st_size,
        "graph_bytes": graph_path.stat().st_size,
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> int:
    args = parse_args()
    if args.max_molecules <= 0 or args.shard_size <= 0 or args.workers <= 0:
        raise ValueError("max-molecules, shard-size, and workers must be positive")
    if args.fingerprint_bits <= 0 or args.fingerprint_bits % 8:
        raise ValueError("fingerprint-bits must be a positive multiple of eight")
    for path in (args.selection, args.candidate_panels):
        if not path.exists():
            raise FileNotFoundError(path)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if any(args.output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {args.output_dir}")

    RDLogger.DisableLog("rdApp.*")
    torch.set_num_threads(1)
    started = time.perf_counter()
    excluded_exact = read_candidate_smiles(args.candidate_panels)
    exclusion_started = time.perf_counter()
    excluded_canonical = canonicalize_many(sorted(excluded_exact), args.workers)
    exclusion_seconds = time.perf_counter() - exclusion_started
    rows, source_rows, stride = select_pilot_rows(args.selection, args.max_molecules)

    reports: list[dict[str, Any]] = []
    seen_canonical: set[str] = set()
    graph_started = time.perf_counter()
    for shard_index, start in enumerate(range(0, len(rows), args.shard_size)):
        shard_rows = rows[start : start + args.shard_size]
        tasks = [
            (
                str(row["SMILES"]),
                args.seed,
                args.spatial_cutoff,
                args.optimization_max_iters,
                args.fingerprint_radius,
                args.fingerprint_bits,
            )
            for row in shard_rows
        ]
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            results = list(executor.map(build_graph_task, tasks))
        report = write_shard(
            args.output_dir,
            shard_index,
            shard_rows,
            results,
            excluded_canonical,
            seen_canonical,
            args.fingerprint_bits // 8,
        )
        reports.append(report)
        print(json.dumps(report), flush=True)

    graph_seconds = time.perf_counter() - graph_started
    total_seconds = time.perf_counter() - started
    attempted = sum(int(report["rows"]) for report in reports)
    successful = sum(int(report["successful_graphs"]) for report in reports)
    total_bytes = sum(
        int(report["metadata_bytes"]) + int(report["graph_bytes"])
        for report in reports
    )
    status_counts: Counter[str] = Counter()
    geometry_counts: Counter[str] = Counter()
    for report in reports:
        status_counts.update(report["status_counts"])
        geometry_counts.update(report["geometry_counts"])
    full_unique_estimate = 7_487_569
    manifest = {
        "selection": str(args.selection),
        "candidate_panels": str(args.candidate_panels),
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "source_rows": source_rows,
        "pilot_sampling": {"method": "source_row_modulo_stride", "stride": stride},
        "candidate_exclusions": {
            "exact_smiles": len(excluded_exact),
            "canonical_smiles": len(excluded_canonical),
            "canonicalization_seconds": exclusion_seconds,
        },
        "attempted_rows": attempted,
        "successful_graphs": successful,
        "status_counts": dict(status_counts),
        "geometry_counts": dict(geometry_counts),
        "shards": reports,
        "elapsed_seconds": total_seconds,
        "graph_elapsed_seconds": graph_seconds,
        "successful_graphs_per_second": successful / graph_seconds if graph_seconds else 0.0,
        "stored_bytes": total_bytes,
        "stored_bytes_per_successful_graph": total_bytes / successful if successful else None,
        "full_unique_molecule_estimate": full_unique_estimate,
        "estimated_full_graph_hours_at_pilot_rate": (
            full_unique_estimate / (successful / graph_seconds) / 3600
            if successful and graph_seconds
            else None
        ),
        "estimated_full_stored_bytes": (
            total_bytes / successful * full_unique_estimate if successful else None
        ),
        "notes": [
            "Coordinates are ligand-internal and are not PGK2 complex poses.",
            "Every attempted row remains represented in metadata, including failures and exclusions.",
            "The full run requires a canonicalized global index before distributed shard construction.",
        ],
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
