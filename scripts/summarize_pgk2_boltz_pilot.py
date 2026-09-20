#!/usr/bin/env python3
"""Summarize Boltz pilot metrics and predicted catalytic-cleft contacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shlex
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def predicted_contacts(cif_path: Path, active_site: dict[str, object], cutoff: float) -> dict[str, object]:
    """Residues with a ligand heavy atom within cutoff in the predicted CIF."""
    lines = cif_path.read_text().splitlines()
    header = lines.index("_atom_site.group_PDB")
    columns: list[str] = []
    index = header
    while lines[index].startswith("_atom_site."):
        columns.append(lines[index].removeprefix("_atom_site."))
        index += 1
    column_index = {name: i for i, name in enumerate(columns)}
    needed = {"type_symbol", "label_asym_id", "label_seq_id", "Cartn_x", "Cartn_y", "Cartn_z"}
    if not needed.issubset(column_index):
        raise ValueError(f"Unexpected atom-site columns in {cif_path}")
    protein: dict[int, list[tuple[float, float, float]]] = defaultdict(list)
    ligand: list[tuple[float, float, float]] = []
    for line in lines[index:]:
        if not line or line == "#" or line.startswith("loop_") or line.startswith("_"):
            break
        fields = shlex.split(line)
        if len(fields) != len(columns):
            continue
        if fields[column_index["type_symbol"]] == "H":
            continue
        coords = tuple(float(fields[column_index[key]]) for key in ("Cartn_x", "Cartn_y", "Cartn_z"))
        chain = fields[column_index["label_asym_id"]]
        if chain == "A":
            protein[int(fields[column_index["label_seq_id"]]) - 1].append(coords)
        elif chain == "B":
            ligand.append(coords)
    if not ligand:
        raise ValueError(f"No ligand atoms found in predicted chain B: {cif_path}")
    contact_indices: list[int] = []
    minimum_distance = float("inf")
    for residue_index, protein_atoms in protein.items():
        distance = min(math.dist(a, b) for a in protein_atoms for b in ligand)
        minimum_distance = min(minimum_distance, distance)
        if distance <= cutoff:
            contact_indices.append(residue_index)
    subsites = active_site["subsites"]
    atp = {int(row["human_index_0_based"]) for row in subsites["ATP_ADP"]}
    pg3 = {int(row["human_index_0_based"]) for row in subsites["3PG"]}
    active = atp | pg3
    active_contacts = sorted(active & set(contact_indices))
    return {
        "ligand_heavy_atoms": len(ligand),
        "active_site_contacts_5A": active_contacts,
        "atp_adp_contacts_5A": sorted(atp & set(contact_indices)),
        "pg3_contacts_5A": sorted(pg3 & set(contact_indices)),
        "active_site_contact_count_5A": len(active_contacts),
        "atp_adp_contact_count_5A": len(atp & set(contact_indices)),
        "pg3_contact_count_5A": len(pg3 & set(contact_indices)),
        "closest_protein_ligand_distance_angstrom": round(minimum_distance, 3),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pilot", type=Path, default=ROOT / "artifacts/pgk2_structural_pilot/pilot_candidates.csv"
    )
    parser.add_argument(
        "--active-site", type=Path,
        default=ROOT / "artifacts/pgk2_structural_pilot/active_site_union.json",
    )
    parser.add_argument(
        "--runs-dir", type=Path,
        default=ROOT / "artifacts/pgk2_structural_pilot/boltz_runs",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "artifacts/pgk2_structural_pilot/boltz_pilot_results.csv",
    )
    parser.add_argument(
        "--run-prefix",
        default="pgk2-pilot",
        help="Boltz run-name prefix, before -NN-<CatalogID> (default: %(default)s).",
    )
    args = parser.parse_args()
    active_site = json.loads(args.active_site.read_text())
    with args.pilot.open(newline="") as handle:
        candidates = list(csv.DictReader(handle))
    output_rows: list[dict[str, object]] = []
    for row in candidates:
        run_name = f"{args.run_prefix}-{int(row['pilot_rank']):02d}-{row['CatalogID'].lower()}"
        run_dir = args.runs_dir / run_name
        metrics_paths = list(run_dir.glob("outputs/files/prediction/metrics.json"))
        result: dict[str, object] = {
            **row,
            "run_name": run_name,
            "status": "pending",
            "binding_confidence": "",
            "optimization_score": "",
            "ligand_iptm": "",
            "active_site_contact_count_5A": "",
            "atp_adp_contact_count_5A": "",
            "pg3_contact_count_5A": "",
            "active_site_contacts_5A": "",
            "atp_adp_contacts_5A": "",
            "pg3_contacts_5A": "",
            "closest_protein_ligand_distance_angstrom": "",
            "predicted_cif": "",
        }
        if metrics_paths:
            metrics = json.loads(metrics_paths[0].read_text())
            binding = metrics["binding_metrics"]
            sample = metrics["best_sample"]["metrics"]
            cif_paths = list(metrics_paths[0].parent.glob("*_predicted.cif"))
            if len(cif_paths) != 1:
                raise RuntimeError(f"Expected exactly one predicted CIF in {metrics_paths[0].parent}")
            contacts = predicted_contacts(cif_paths[0], active_site, cutoff=5.0)
            result.update(
                {
                    "status": "succeeded",
                    "binding_confidence": binding["binding_confidence"],
                    "optimization_score": binding["optimization_score"],
                    "ligand_iptm": sample["ligand_iptm"],
                    "active_site_contact_count_5A": contacts["active_site_contact_count_5A"],
                    "atp_adp_contact_count_5A": contacts["atp_adp_contact_count_5A"],
                    "pg3_contact_count_5A": contacts["pg3_contact_count_5A"],
                    "active_site_contacts_5A": ";".join(str(i + 1) for i in contacts["active_site_contacts_5A"]),
                    "atp_adp_contacts_5A": ";".join(str(i + 1) for i in contacts["atp_adp_contacts_5A"]),
                    "pg3_contacts_5A": ";".join(str(i + 1) for i in contacts["pg3_contacts_5A"]),
                    "closest_protein_ligand_distance_angstrom": contacts["closest_protein_ligand_distance_angstrom"],
                    "predicted_cif": str(cif_paths[0]),
                }
            )
        output_rows.append(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
