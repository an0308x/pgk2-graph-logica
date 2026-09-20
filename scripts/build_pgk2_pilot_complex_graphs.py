#!/usr/bin/env python3
"""Build Graph LogiCA inputs for the 20 completed PGK2 structural-pilot poses."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from logica_binding.graph_model import build_boltz_complex_graphs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidates",
        type=Path,
        default=ROOT / "artifacts" / "pgk2_structural_pilot" / "pilot_candidates.csv",
    )
    parser.add_argument(
        "--pocket-json",
        type=Path,
        default=ROOT / "artifacts" / "pgk2_inhibitor_pocket" / "inhibitor_pocket.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "artifacts" / "graph_logica" / "pilot_complex_graphs",
    )
    parser.add_argument("--contact-cutoff", type=float, default=6.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    with args.candidates.open(newline="", encoding="utf-8") as handle:
        candidates = list(csv.DictReader(handle))
    conditions = (
        (
            "cmp47",
            ROOT / "artifacts" / "pgk2_structural_pilot" / "boltz_runs",
            "pgk2-pilot",
        ),
        (
            "atp_3pg",
            ROOT / "artifacts" / "pgk2_structural_pilot" / "boltz_runs_2paa",
            "pgk2-2paa",
        ),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    for condition, runs_dir, prefix in conditions:
        for row in candidates:
            rank = int(row["pilot_rank"])
            catalog_id = row["CatalogID"]
            run = runs_dir / f"{prefix}-{rank:02d}-{catalog_id.lower()}"
            cif_paths = list(run.glob("outputs/files/prediction/*_predicted.cif"))
            if len(cif_paths) != 1:
                raise RuntimeError(f"Expected one predicted CIF under {run}, found {len(cif_paths)}")
            pocket, ligand, contacts = build_boltz_complex_graphs(
                cif_paths[0],
                row["SMILES"],
                args.pocket_json,
                contact_cutoff=args.contact_cutoff,
            )
            output = args.output_dir / f"{rank:02d}_{catalog_id}_{condition}.pt"
            torch.save(
                {
                    "catalog_id": catalog_id,
                    "smiles": row["SMILES"],
                    "condition": condition,
                    "source_cif": str(cif_paths[0]),
                    "pocket_graph": pocket,
                    "ligand_graph": ligand,
                    "contact_graph": contacts,
                },
                output,
            )
            records.append(
                {
                    "catalog_id": catalog_id,
                    "condition": condition,
                    "graph": str(output),
                    "ligand_atoms": int(ligand.node_features.shape[0]),
                    "direct_contact_edges": int(contacts.edge_index.shape[1]),
                    "contacted_residues": int(torch.unique(contacts.edge_index[0]).numel()),
                }
            )
    report = {
        "complex_graphs": len(records),
        "candidates": len(candidates),
        "conditions": [value[0] for value in conditions],
        "contact_cutoff_angstrom": args.contact_cutoff,
        "records": records,
        "scope": (
            "Diagnostic pilot only. These candidates have no kinase-assay labels, "
            "so these graphs cannot by themselves train or validate Graph LogiCA."
        ),
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in report.items() if key != "records"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
