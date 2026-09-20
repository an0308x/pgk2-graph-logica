#!/usr/bin/env python3
"""Repair pair splits across stages and downweight uncertain DEL preferences."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import zipfile
from pathlib import Path

import polars as pl
from rdkit import Chem

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from logica_binding.validation_protocol import (
    PROTOCOL, count_reliability, group_split, structure_identity, validate_pair_splits,
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--candidate-panels", type=Path, default=ROOT / "DREAM_challenge_2026/Val-Test-set.zip")
    args = p.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    excluded = set()
    with zipfile.ZipFile(args.candidate_panels) as z:
        for name in ["Val-Test-set/PGK2_Validation_split.csv", "Val-Test-set/PGK2_Test_split.csv"]:
            with z.open(name) as f:
                for r in csv.DictReader(line.decode() for line in f):
                    mol = Chem.MolFromSmiles(r["SMILES"])
                    if mol is None:
                        raise ValueError("Unparseable candidate prevents exclusion audit")
                    excluded.add(Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True))
            print(json.dumps({"excluded_panel": name, "canonical_count": len(excluded)}), flush=True)
    sources = {
        "pretrain": ROOT / "artifacts/pgk2_matched_sar_pairs/pgk2_matched_sar_pairs.parquet",
        "refine": ROOT / "artifacts/pgk2_competition_pairs/pgk2_competition_pairs.parquet",
    }
    tables, manifest, reports = {}, {}, {}
    for stage, path in sources.items():
        original = pl.read_parquet(path).to_dicts()
        kept = []
        rejected = {"candidate_overlap": 0, "cross_split": 0, "no_control_support": 0, "same_molecule": 0}
        for r in original:
            a, ga = structure_identity(r["positive_smiles"])
            b, gb = structure_identity(r["negative_smiles"])
            if a in excluded or b in excluded:
                rejected["candidate_overlap"] += 1
                continue
            if a == b:
                rejected["same_molecule"] += 1
                continue
            sa, sb = group_split(ga), group_split(gb)
            for smi, group, split in [(a, ga, sa), (b, gb, sb)]:
                manifest[smi] = {"canonical_smiles": smi, "scaffold": group, "split": split}
            if sa != sb:
                rejected["cross_split"] += 1
                continue
            confidence = count_reliability(r)
            if confidence <= 0:
                rejected["no_control_support"] += 1
                continue
            kept.append({**r, "positive_smiles": a, "negative_smiles": b,
                         "original_pair_weight": r["pair_weight"],
                         "pair_weight": float(r["pair_weight"]) * confidence,
                         "count_reliability": confidence, "split": sa})
        validate_pair_splits(kept)
        tables[stage] = kept
        reports[stage] = {"source_rows": len(original), "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                          "retained_rows": len(kept), "rejected": rejected,
                          "split_counts": {s: sum(r["split"] == s for r in kept) for s in ["train", "dev", "holdout"]}}
    manifest_rows = sorted(manifest.values(), key=lambda r: r["canonical_smiles"])
    manifest_id = hashlib.sha256(json.dumps(manifest_rows, sort_keys=True).encode()).hexdigest()
    all_rows = [r for rows in tables.values() for r in rows]
    sets = {s: {r[k] for r in all_rows if r["split"] == s for k in ["positive_smiles", "negative_smiles"]}
            for s in ["train", "dev", "holdout"]}
    overlaps = {f"{a}_{b}": len(sets[a] & sets[b]) for a,b in [("train","dev"),("train","holdout"),("dev","holdout")]}
    if any(overlaps.values()):
        raise ValueError(f"Cross-stage leakage: {overlaps}")
    args.output_dir.mkdir(parents=True)
    for stage, rows in tables.items():
        pl.DataFrame(rows).with_columns(pl.lit(manifest_id).alias("split_manifest_sha256"),
                                      pl.lit(PROTOCOL).alias("protocol")).write_parquet(args.output_dir / f"{stage}.parquet")
    pl.DataFrame(manifest_rows).write_parquet(args.output_dir / "molecule_splits.parquet")
    report = {"protocol": PROTOCOL, "split_manifest_sha256": manifest_id, "stages": reports,
              "cross_stage_molecule_overlap": overlaps, "excluded_candidate_canonical_count": len(excluded),
              "weight_definition": "original_weight * target/(target+20) * observed_comparator_control/(control+5)",
              "weight_is_calibrated_probability": False,
              "scope": "Repair diagnostic using existing pair pool; not full-library training or a challenge submission"}
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
