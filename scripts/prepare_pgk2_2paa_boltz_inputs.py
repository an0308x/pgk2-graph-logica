#!/usr/bin/env python3
"""Prepare an alternate 2PAA-template PGK2 Boltz control pilot."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from rdkit import Chem


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pilot",
        type=Path,
        default=ROOT / "artifacts/pgk2_structural_pilot/pilot_candidates.csv",
    )
    parser.add_argument(
        "--active-site",
        type=Path,
        default=ROOT / "artifacts/pgk2_structural_pilot/active_site_union.json",
    )
    parser.add_argument(
        "--template",
        type=Path,
        default=ROOT / "artifacts/pgk2_structural_pilot/2PAA_mouse_PGK2_ATP_3PG.pdb",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "artifacts/pgk2_structural_pilot/boltz_inputs_2paa",
    )
    args = parser.parse_args()

    sequence = "".join(
        line.strip()
        for line in (ROOT / "data/proteins/PGK2_P07205.fasta").read_text().splitlines()
        if not line.startswith(">")
    )
    active_site = json.loads(args.active_site.read_text())
    pocket = active_site["human_active_site_union_indices_0_based"]
    template = args.template.resolve()
    if not template.is_file():
        raise FileNotFoundError(f"2PAA template not found: {template}")

    with args.pilot.open(newline="") as handle:
        candidates = list(csv.DictReader(handle))
    if len(candidates) != 10:
        raise ValueError(f"Expected 10 pilot candidates, found {len(candidates)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, str]] = []
    for row in candidates:
        molecule = Chem.MolFromSmiles(row["SMILES"])
        if molecule is None or molecule.GetNumAtoms() >= 50:
            raise ValueError(f"Unsupported Boltz ligand: {row['CatalogID']}")
        name = f"pgk2-2paa-{int(row['pilot_rank']):02d}-{row['CatalogID'].lower()}"
        lines = [
            f"# {name}; alternate 2PAA-template conformational control.",
            "entities:",
            "  - type: protein",
            "    chain_ids: [A]",
            f'    value: "{sequence}"',
            "  - type: ligand_smiles",
            "    chain_ids: [B]",
            f"    value: {json.dumps(row['SMILES'])}",
            "binding:",
            "  type: ligand_protein_binding",
            "  binder_chain_id: B",
            "constraints:",
            "  - type: pocket",
            "    binder_chain_id: B",
            "    contact_residues:",
            f"      A: {pocket}",
            "    max_distance_angstrom: 6.0",
            "    force: false",
            "templates:",
            "  - template_structure:",
            "      type: base64",
            "      media_type: chemical/x-pdb",
            f'      data: "@data://{template}"',
            "    template_chains:",
            "      - input_chain_id: A",
            "        template_chain_id: A",
            "num_samples: 1",
            "",
        ]
        path = args.output_dir / f"{name}.yaml"
        path.write_text("\n".join(lines))
        records.append({"name": name, "input": str(path), **row})

    manifest = args.output_dir / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "purpose": "Conformational-control reranking pilot; not a challenge submission.",
                "target": "human PGK2 (P07205)",
                "template": "mouse PGK2 2PAA, ATP + 3-PG-bound conformation",
                "template_path": str(template),
                "active_site_definition": active_site["pocket_definition"],
                "active_site_residue_indices_0_based": pocket,
                "n_candidates": len(records),
                "candidates": records,
            },
            indent=2,
        )
        + "\n"
    )
    print(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
