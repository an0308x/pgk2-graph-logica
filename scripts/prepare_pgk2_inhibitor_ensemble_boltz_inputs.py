#!/usr/bin/env python3
"""Prepare PGK2 compound-21/47 inhibitor-pocket ensemble Boltz inputs.

This writes inputs only.  It deliberately omits PGK1 from the primary DREAM
ranking workflow and does not submit paid predictions.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from rdkit import Chem


ROOT = Path(__file__).resolve().parents[1]
QUINAZOLINE = Chem.MolFromSmarts("c1ccc2ncncc2c1")


def sequence(path: Path) -> str:
    return "".join(
        line.strip() for line in path.read_text().splitlines() if not line.startswith(">")
    )


def payload(
    run_name: str,
    protein_sequence: str,
    smiles: str,
    pocket_indices: list[int],
    template: Path,
) -> str:
    return f'''# {run_name}; PGK2 active-site-inhibition diagnostic, not an affinity prediction.
entities:
  - type: protein
    chain_ids: [A]
    value: "{protein_sequence}"
  - type: ligand_smiles
    chain_ids: [B]
    value: {json.dumps(smiles)}
binding:
  type: ligand_protein_binding
  binder_chain_id: B
constraints:
  - type: pocket
    binder_chain_id: B
    contact_residues:
      A: {pocket_indices}
    max_distance_angstrom: 6.0
    force: false
templates:
  - template_structure:
      type: base64
      media_type: chemical/x-pdb
      data: "@data://{template.resolve()}"
    template_chains:
      - input_chain_id: A
        template_chain_id: A
num_samples: 1
'''


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shortlist", type=Path, required=True)
    parser.add_argument("--max-compounds", type=int, default=10)
    parser.add_argument("--include-2paa", action="store_true")
    parser.add_argument(
        "--pgk2-fasta", type=Path, default=ROOT / "data/proteins/PGK2_P07205.fasta"
    )
    parser.add_argument(
        "--pocket",
        type=Path,
        default=ROOT / "artifacts/pgk2_inhibitor_pocket/inhibitor_pocket.json",
    )
    parser.add_argument(
        "--compound21-template",
        type=Path,
        default=ROOT / "data/structures/cache7/PGK2_cmp21.pdb",
    )
    parser.add_argument(
        "--compound47-template",
        type=Path,
        default=ROOT / "data/structures/cache7/PGK2_cmp47.pdb",
    )
    parser.add_argument(
        "--atp-template",
        type=Path,
        default=ROOT / "artifacts/pgk2_structural_pilot/2PAA_mouse_PGK2_ATP_3PG.pdb",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "artifacts/pgk2_inhibitor_ensemble_boltz_inputs",
    )
    args = parser.parse_args()
    if args.max_compounds <= 0:
        raise ValueError("--max-compounds must be positive")
    templates = [
        ("cmp21", args.compound21_template),
        ("cmp47", args.compound47_template),
    ]
    if args.include_2paa:
        templates.append(("2paa", args.atp_template))
    required = [args.shortlist, args.pgk2_fasta, args.pocket, *[path for _, path in templates]]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing inputs: {missing}")

    with args.shortlist.open(newline="") as handle:
        rows = list(csv.DictReader(handle))[: args.max_compounds]
    if not rows or not {"CatalogID", "SMILES"}.issubset(rows[0]):
        raise ValueError("Shortlist must contain CatalogID and SMILES")
    protein_sequence = sequence(args.pgk2_fasta)
    if len(protein_sequence) != 417:
        raise ValueError("Expected canonical 417-residue human PGK2")
    pocket = json.loads(args.pocket.read_text())
    pocket_indices = [
        int(index)
        for index in pocket["direct_ligand_edges"]["residue_indices_0_based"]
    ]
    if len(pocket_indices) != 26:
        raise ValueError(f"Expected 26-residue inhibitor envelope, found {len(pocket_indices)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, object]] = []
    for rank, row in enumerate(rows, start=1):
        molecule = Chem.MolFromSmiles(row["SMILES"])
        if molecule is None:
            raise ValueError(f"Invalid SMILES for {row['CatalogID']}")
        if QUINAZOLINE is not None and molecule.HasSubstructMatch(QUINAZOLINE):
            raise ValueError(f"Quinazoline reached inhibitor ensemble: {row['CatalogID']}")
        if molecule.GetNumHeavyAtoms() >= 50:
            raise ValueError(f"Boltz ligand has >=50 heavy atoms: {row['CatalogID']}")
        stem = f"{rank:04d}-{row['CatalogID'].lower()}"
        for template_name, template_path in templates:
            run_name = f"pgk2-inhibitor-{template_name}-{stem}"
            output = args.output_dir / f"{run_name}.yaml"
            output.write_text(
                payload(
                    run_name,
                    protein_sequence,
                    row["SMILES"],
                    pocket_indices,
                    template_path,
                )
            )
            manifest_rows.append(
                {
                    "shortlist_rank": rank,
                    "CatalogID": row["CatalogID"],
                    "SMILES": row["SMILES"],
                    "template": template_name,
                    "template_path": str(template_path.resolve()),
                    "run_name": run_name,
                    "input": str(output.resolve()),
                }
            )
    manifest_path = args.output_dir / "manifest.csv"
    with manifest_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)
    report = {
        "compounds": len(rows),
        "templates": [name for name, _ in templates],
        "inputs": len(manifest_rows),
        "direct_pocket_residues": len(pocket_indices),
        "pgk1_primary_ranking": False,
        "manifest": str(manifest_path.resolve()),
        "execution": "not submitted; paid prediction requires separate cost confirmation",
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
