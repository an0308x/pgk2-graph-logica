#!/usr/bin/env python3
"""Build one packed ligand-graph shard from the canonical PGK2 index."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
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

from logica_binding.graph_model import (  # noqa: E402
    smiles_to_ligand_graph,
    smiles_to_rdkit_3d_graph_with_quality,
)


@dataclass(frozen=True)
class GraphResult:
    status: str
    quality: str
    node_features: np.ndarray | None
    coordinates: np.ndarray | None
    edge_index: np.ndarray | None
    edge_features: np.ndarray | None
    fingerprint: np.ndarray | None
    error: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--index",
        type=Path,
        default=ROOT
        / "artifacts"
        / "pgk2_full_canonical_index"
        / "canonical_training_index.parquet",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--graph-mode", choices=("2d", "etkdg"), default="2d")
    parser.add_argument("--subset", choices=("all", "ablation"), default="all")
    parser.add_argument("--shard-id", type=int, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--spatial-cutoff", type=float, default=4.5)
    parser.add_argument("--optimization-max-iters", type=int, default=0)
    parser.add_argument("--fingerprint-bits", type=int, default=2048)
    parser.add_argument("--fingerprint-radius", type=int, default=2)
    return parser.parse_args()


def shard_bounds(total: int, shard_id: int, num_shards: int) -> tuple[int, int]:
    if total < 0 or num_shards <= 0 or not 0 <= shard_id < num_shards:
        raise ValueError("invalid shard request")
    start = total * shard_id // num_shards
    stop = total * (shard_id + 1) // num_shards
    return start, stop


def build_graph(
    task: tuple[str, str, int, float, int, int, int]
) -> GraphResult:
    smiles, mode, seed, cutoff, max_iters, fp_radius, fp_bits = task
    try:
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            raise ValueError("RDKit parse failed")
        if mode == "2d":
            graph = smiles_to_ligand_graph(smiles)
            quality = "topology_2d"
        else:
            graph, quality = smiles_to_rdkit_3d_graph_with_quality(
                smiles,
                seed=seed,
                spatial_cutoff=cutoff,
                optimization_max_iters=max_iters,
                warn_on_fallback=False,
            )
        generator = rdFingerprintGenerator.GetMorganGenerator(
            radius=fp_radius, fpSize=fp_bits
        )
        fingerprint = np.packbits(
            np.asarray(generator.GetFingerprintAsNumPy(molecule), dtype=np.uint8),
            bitorder="little",
        )
        return GraphResult(
            status="ok",
            quality=quality,
            node_features=graph.node_features.numpy().astype(np.float32, copy=False),
            coordinates=(
                None
                if graph.coordinates is None
                else graph.coordinates.numpy().astype(np.float32, copy=False)
            ),
            edge_index=graph.edge_index.numpy().astype(np.int32, copy=False),
            edge_features=graph.edge_features.numpy().astype(np.float32, copy=False),
            fingerprint=fingerprint,
            error="",
        )
    except Exception as exc:
        return GraphResult(
            "graph_failure", "", None, None, None, None, None,
            f"{type(exc).__name__}: {exc}"[:1000],
        )


def initialize_graph_worker() -> None:
    """Keep each RDKit worker single-threaded and its logs bounded."""
    RDLogger.DisableLog("rdApp.*")
    torch.set_num_threads(1)


def concatenate_or_empty(
    arrays: list[np.ndarray], shape: tuple[int, ...], axis: int, dtype: np.dtype[Any]
) -> np.ndarray:
    return np.concatenate(arrays, axis=axis) if arrays else np.empty(shape, dtype=dtype)


def main() -> int:
    args = parse_args()
    if args.workers <= 0 or args.fingerprint_bits <= 0 or args.fingerprint_bits % 8:
        raise ValueError("workers must be positive and fingerprint bits divisible by eight")
    if not args.index.exists():
        raise FileNotFoundError(args.index)
    if args.optimization_max_iters < 0:
        raise ValueError("optimization-max-iters must be non-negative")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"shard-{args.shard_id:05d}-of-{args.num_shards:05d}"
    metadata_path = args.output_dir / f"{stem}-metadata.parquet"
    graph_path = args.output_dir / f"{stem}-graphs.npz"
    report_path = args.output_dir / f"{stem}-report.json"
    existing = [path.exists() for path in (metadata_path, graph_path, report_path)]
    if all(existing):
        print(report_path.read_text(encoding="utf-8"), end="")
        return 0
    if any(existing):
        raise FileExistsError(f"Partial output exists for {stem}; inspect it before retrying")

    RDLogger.DisableLog("rdApp.*")
    torch.set_num_threads(1)
    started = time.perf_counter()
    scan = pl.scan_parquet(args.index)
    if args.subset == "ablation":
        scan = scan.filter(pl.col("informative") | (pl.col("sample_bucket") < 14))
    total = int(scan.select(pl.len()).collect().item())
    start, stop = shard_bounds(total, args.shard_id, args.num_shards)
    frame = scan.slice(start, stop - start).collect(engine="streaming")
    tasks = [
        (
            smiles,
            args.graph_mode,
            args.seed,
            args.spatial_cutoff,
            args.optimization_max_iters,
            args.fingerprint_radius,
            args.fingerprint_bits,
        )
        for smiles in frame["canonical_smiles"].to_list()
    ]
    with ProcessPoolExecutor(
        max_workers=args.workers, initializer=initialize_graph_worker
    ) as executor:
        results = list(executor.map(build_graph, tasks, chunksize=128))

    nodes: list[np.ndarray] = []
    coordinates: list[np.ndarray] = []
    edges: list[np.ndarray] = []
    edge_features: list[np.ndarray] = []
    fingerprints: list[np.ndarray] = []
    node_offsets = [0]
    edge_offsets = [0]
    statuses: Counter[str] = Counter()
    qualities: Counter[str] = Counter()
    atom_counts: list[int] = []
    edge_counts: list[int] = []
    errors: list[str] = []
    for result in results:
        valid = result.status == "ok"
        statuses[result.status] += 1
        if valid:
            assert result.node_features is not None
            assert result.edge_index is not None
            assert result.edge_features is not None
            assert result.fingerprint is not None
            nodes.append(result.node_features)
            if result.coordinates is not None:
                coordinates.append(result.coordinates)
            edges.append(result.edge_index)
            edge_features.append(result.edge_features)
            fingerprints.append(result.fingerprint)
            node_offsets.append(node_offsets[-1] + len(result.node_features))
            edge_offsets.append(edge_offsets[-1] + len(result.edge_features))
            qualities[result.quality] += 1
            atom_counts.append(len(result.node_features))
            edge_counts.append(len(result.edge_features))
        else:
            fingerprints.append(np.zeros(args.fingerprint_bits // 8, dtype=np.uint8))
            node_offsets.append(node_offsets[-1])
            edge_offsets.append(edge_offsets[-1])
            atom_counts.append(0)
            edge_counts.append(0)
        errors.append(result.error)

    node_matrix = concatenate_or_empty(nodes, (0, 20), 0, np.dtype(np.float32))
    edge_matrix = concatenate_or_empty(edges, (2, 0), 1, np.dtype(np.int32))
    edge_feature_matrix = concatenate_or_empty(
        edge_features, (0, 7), 0, np.dtype(np.float32)
    )
    metadata = frame.with_columns(
        pl.Series("graph_status", [result.status for result in results]),
        pl.Series("geometry_quality", [result.quality for result in results]),
        pl.Series("atom_count", atom_counts, dtype=pl.Int32),
        pl.Series("directed_edge_count", edge_counts, dtype=pl.Int32),
        pl.Series("graph_error", errors),
    )
    metadata_tmp = metadata_path.with_suffix(".parquet.tmp")
    metadata.write_parquet(metadata_tmp, compression="zstd", statistics=True)
    metadata_tmp.replace(metadata_path)
    graph_tmp = graph_path.with_suffix(".npz.tmp")
    arrays: dict[str, np.ndarray] = {
        "node_features": node_matrix,
        "edge_index": edge_matrix,
        "edge_features": edge_feature_matrix,
        "node_offsets": np.asarray(node_offsets, dtype=np.int64),
        "edge_offsets": np.asarray(edge_offsets, dtype=np.int64),
        "fingerprints": np.stack(fingerprints).astype(np.uint8, copy=False),
    }
    if args.graph_mode == "etkdg":
        arrays["coordinates"] = concatenate_or_empty(
            coordinates, (0, 3), 0, np.dtype(np.float32)
        )
    with graph_tmp.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    graph_tmp.replace(graph_path)
    elapsed = time.perf_counter() - started
    report = {
        "completed": True,
        "graph_mode": args.graph_mode,
        "subset": args.subset,
        "shard_id": args.shard_id,
        "num_shards": args.num_shards,
        "global_subset_rows": total,
        "range": [start, stop],
        "rows": frame.height,
        "status_counts": dict(statuses),
        "quality_counts": dict(qualities),
        "nodes": int(node_matrix.shape[0]),
        "directed_edges": int(edge_feature_matrix.shape[0]),
        "metadata_bytes": metadata_path.stat().st_size,
        "graph_bytes": graph_path.stat().st_size,
        "elapsed_seconds": elapsed,
        "graphs_per_second": statuses["ok"] / elapsed if elapsed else 0.0,
        "coordinates_stored": args.graph_mode == "etkdg",
        "notes": [
            "2d mode contains atom and covalent-bond topology only.",
            "ETKDG coordinates are ligand-internal and are not PGK2 docking poses.",
        ],
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
