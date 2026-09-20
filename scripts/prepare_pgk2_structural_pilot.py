#!/usr/bin/env python3
"""Create a diverse, non-quinazoline PGK2 structural-reranking pilot."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from rdkit import Chem


ROOT = Path(__file__).resolve().parents[1]
# Aromatic quinazoline core; CACHE #7 excludes this already-developed series.
QUINAZOLINE = Chem.MolFromSmarts("c1ccc2ncncc2c1")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "artifacts/pgk2_series_portfolios/validation_series_cap1_ranked.csv",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "artifacts/pgk2_structural_pilot"
    )
    parser.add_argument(
        "--active-site-json", type=Path,
        default=ROOT / "artifacts/pgk2_structural_pilot/active_site_union.json",
    )
    parser.add_argument("--n", type=int, default=10)
    args = parser.parse_args()
    sequence = "".join(
        line.strip()
        for line in (ROOT / "data/proteins/PGK2_P07205.fasta").read_text().splitlines()
        if not line.startswith(">")
    )
    active_site = json.loads(args.active_site_json.read_text())
    pocket_indices = active_site["human_active_site_union_indices_0_based"]
    if not pocket_indices:
        raise ValueError("Active-site definition has no residues")

    with args.input.open(newline="") as handle:
        ranked = list(csv.DictReader(handle))

    selected: list[dict[str, str]] = []
    excluded_quinazolines = 0
    for rank, row in enumerate(ranked, start=1):
        molecule = Chem.MolFromSmiles(row["SMILES"])
        if molecule is None:
            raise ValueError(f"Invalid SMILES for {row['CatalogID']}")
        if QUINAZOLINE is not None and molecule.HasSubstructMatch(QUINAZOLINE):
            excluded_quinazolines += 1
            continue
        selected.append(
            {
                "pilot_rank": str(len(selected) + 1),
                "logica_rank_before_filter": str(rank),
                "CatalogID": row["CatalogID"],
                "SMILES": row["SMILES"],
                "logica_score": row["logica_score"],
                "series_id": row["series_id"],
            }
        )
        if len(selected) == args.n:
            break

    if len(selected) != args.n:
        raise RuntimeError(f"Only selected {len(selected)} candidates; expected {args.n}")
    if len({row["series_id"] for row in selected}) != len(selected):
        raise RuntimeError("Pilot must contain one candidate per inferred series")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "pilot_candidates.csv"
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(selected[0]))
        writer.writeheader()
        writer.writerows(selected)
    reference = (args.output_dir / "PGK2_cmp47_1.pdb").resolve()
    if not reference.exists():
        raise FileNotFoundError(f"Reference PDB not found: {reference}")
    inputs_dir = args.output_dir / "boltz_inputs"
    inputs_dir.mkdir(exist_ok=True)
    for row in selected:
        molecule = Chem.MolFromSmiles(row["SMILES"])
        assert molecule is not None
        if molecule.GetNumAtoms() >= 50:
            raise ValueError(
                f"{row['CatalogID']} has {molecule.GetNumAtoms()} atoms; Boltz supports <50"
            )
        name = f"pgk2-pilot-{int(row['pilot_rank']):02d}-{row['CatalogID'].lower()}"
        payload = f'''# {name}; generated from matched-SAR LogiCA. Not a challenge submission.
entities:
  - type: protein
    chain_ids: [A]
    value: "{sequence}"
  - type: ligand_smiles
    chain_ids: [B]
    value: {json.dumps(row['SMILES'])}
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
      data: "@data://{reference}"
    template_chains:
      - input_chain_id: A
        template_chain_id: A
num_samples: 1
'''
        (inputs_dir / f"{name}.yaml").write_text(payload)
    (args.output_dir / "pilot_manifest.json").write_text(
        json.dumps(
            {
                "target": "PGK2 (human, P07205)",
                "purpose": "Small, structurally diverse LogiCA reranking pilot; not a challenge submission.",
                "source_ranking": str(args.input),
                "selection": "First non-quinazoline molecules from cap-1 inferred-series portfolio",
                "n_candidates": len(selected),
                "distinct_inferred_series": len({row["series_id"] for row in selected}),
                "excluded_quinazolines_before_completion": excluded_quinazolines,
                "reference_structure": {
                    "selected": "CACHE hPGK2 co-crystal with selective compound 47",
                    "local_path": str(args.output_dir / "PGK2_cmp47_1.pdb"),
                    "url": "https://cache-challenge.org/sites/default/files/challenge-7/files/PGK2_cmp47_1.pdb",
                    "template_chain": "A",
                    "ligand": "LIG residue 501",
                    "resolution_angstrom": 1.44,
                    "pocket_definition": active_site["pocket_definition"],
                    "pocket_residue_indices_0_based": pocket_indices,
                },
                "active_site_definition_path": str(args.active_site_json),
                "boltz_inputs_dir": str(inputs_dir),
            },
            indent=2,
        )
        + "\n"
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
