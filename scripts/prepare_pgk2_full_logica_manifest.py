#!/usr/bin/env python3
"""Versioned metadata overlay; reuse all existing graphs without rebuilding them."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import sys

import polars as pl
from rdkit import Chem

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from build_pgk2_canonical_index import panel_smiles
from logica_binding.validation_protocol import group_split, structure_identity
from logica_binding.streaming_graph_logica import FULL_PROTOCOL, observed_evidence


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def scaffold_key(value):
    raw, acyclic = value
    mol = Chem.MolFromSmiles(raw)
    if mol is None:
        raise ValueError("Invalid cached scaffold")
    key = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
    return ("acyclic:" if acyclic else "") + key


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--graph-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--candidate-panels", type=Path, default=ROOT / "DREAM_challenge_2026/Val-Test-set.zip")
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    excluded = {structure_identity(s)[0] for s in panel_smiles(args.candidate_panels)}
    reports = sorted(args.graph_dir.glob("*-report.json"))
    if not reports:
        raise ValueError("No graph shards")
    cache, entries = {}, []
    expected_start = 0
    counts = {s: {"molecules": 0, "observed_control_molecules": 0,
                  "competition_observed": 0} for s in ("train", "dev", "holdout")}
    columns = ["molecule_id", "canonical_smiles", "murcko_scaffold", "source_rows",
               "count_PGK2", "count_PGK2_with_inhibitor", "count_NTC_selection",
               "count_NTC_supplement", "ntc_supplement_rows", "historic_hits", "graph_status"]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for report_path in reports:
            report = json.loads(report_path.read_text())
            if not report["completed"] or report["graph_mode"] != "2d" or report["subset"] != "all":
                raise ValueError(f"Not a complete full-library 2D shard: {report_path}")
            start, stop = report["range"]
            if start != expected_start:
                raise ValueError("Noncontiguous graph coverage")
            stem = report_path.name.removesuffix("-report.json")
            metadata = args.graph_dir / f"{stem}-metadata.parquet"
            graphs = args.graph_dir / f"{stem}-graphs.npz"
            frame = pl.read_parquet(metadata, columns=columns)
            if frame["molecule_id"].to_list() != list(range(start, stop)):
                raise ValueError("Graph IDs are not contiguous and aligned")
            if (frame["graph_status"] != "ok").any():
                raise ValueError("Failed graph: full coverage cannot be claimed")
            if set(frame["canonical_smiles"].to_list()) & excluded:
                raise ValueError("Challenge candidate leaked into graph library")
            raw_keys = [(s or smi, not bool(s)) for s, smi in
                        zip(frame["murcko_scaffold"], frame["canonical_smiles"])]
            missing = sorted(set(raw_keys) - cache.keys())
            for raw, key in zip(missing, pool.map(scaffold_key, missing, chunksize=256)):
                cache[raw] = key
            keys = [cache[k] for k in raw_keys]
            splits = [group_split(k) for k in keys]
            proxy, confidence = observed_evidence(frame)
            output = frame.drop("murcko_scaffold").with_columns(
                pl.Series("scaffold_key", keys), pl.Series("split", splits),
                *[pl.Series(f"proxy_{i}", proxy[:, i]) for i in range(3)],
                *[pl.Series(f"confidence_{i}", confidence[:, i]) for i in range(3)])
            path = args.output_dir / f"{stem}-metadata.parquet"
            output.write_parquet(path, compression="zstd")
            for split in counts:
                selected = output.filter(pl.col("split") == split)
                counts[split]["molecules"] += selected.height
                counts[split]["observed_control_molecules"] += selected.filter(
                    pl.any_horizontal([pl.col(f"confidence_{i}") > 0 for i in range(3)])).height
                counts[split]["competition_observed"] += selected.filter(pl.col("confidence_0") > 0).height
            entries.append({"metadata": path.name, "metadata_sha256": sha256(path),
                            "graphs": str(graphs.resolve()), "graphs_sha256": sha256(graphs),
                            "source_metadata_sha256": sha256(metadata), "range": [start, stop]})
            expected_start = stop
            print(json.dumps({"shard": stem, "cumulative_molecules": stop}), flush=True)
    if expected_start != report["global_subset_rows"]:
        raise ValueError("Missing final graph shard")
    manifest = {"protocol": FULL_PROTOCOL, "completed": True, "graph_mode": "2d",
                "molecules": expected_start, "splits": counts, "shards": entries,
                "candidate_panels_sha256": sha256(args.candidate_panels),
                "excluded_candidate_canonical_count": len(excluded),
                "split_rule": "validation_protocol.group_split; nonstereochemical Murcko; shared v2 across stages",
                "zscore_columns_used": False, "duplicate_counts": "mean per source row; not independent replicates",
                "zero_control_semantics": "unobserved/uncertain; no pair supervision",
                "objective": "ligand-only reconstruction on all train molecules plus confidence-scaled LogiCA ranking from observed controls"}
    manifest["manifest_sha256"] = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: v for k, v in manifest.items() if k != "shards"}, indent=2))


if __name__ == "__main__":
    main()
