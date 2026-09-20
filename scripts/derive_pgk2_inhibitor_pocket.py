#!/usr/bin/env python3
"""Derive the human PGK2 inhibitor envelope from compounds 21 and 47."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def heavy_atoms(path: Path) -> list[dict[str, object]]:
    atoms: list[dict[str, object]] = []
    for line in path.read_text().splitlines():
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        element = (line[76:78].strip() or line[12:16].strip()[0]).upper()
        if element == "H":
            continue
        atoms.append(
            {
                "record": line[:6].strip(),
                "name": line[12:16].strip(),
                "altloc": line[16].strip(),
                "residue_name": line[17:20].strip(),
                "chain": line[21].strip(),
                "residue_number": int(line[22:26]),
                "coordinates": tuple(
                    float(line[start:end]) for start, end in ((30, 38), (38, 46), (46, 54))
                ),
            }
        )
    if not atoms:
        raise ValueError(f"No coordinate atoms found in {path}")
    return atoms


def ligand_contact_copies(path: Path, cutoff: float) -> list[dict[str, object]]:
    atoms = heavy_atoms(path)
    ligands: dict[tuple[str, int, str], list[dict[str, object]]] = defaultdict(list)
    for atom in atoms:
        if atom["record"] == "HETATM" and atom["residue_name"] == "LIG":
            ligands[(str(atom["chain"]), int(atom["residue_number"]), str(atom["altloc"]))].append(atom)
    copies: list[dict[str, object]] = []
    for (chain, ligand_residue, altloc), ligand in sorted(ligands.items()):
        residues: dict[int, list[dict[str, object]]] = defaultdict(list)
        for atom in atoms:
            if (
                atom["record"] == "ATOM"
                and atom["chain"] == chain
                and atom["altloc"] in {"", altloc}
            ):
                residues[int(atom["residue_number"])].append(atom)
        contacts: list[dict[str, object]] = []
        for residue_number, residue_atoms in sorted(residues.items()):
            distance = min(
                math.dist(protein["coordinates"], ligand_atom["coordinates"])
                for protein in residue_atoms
                for ligand_atom in ligand
            )
            if distance <= cutoff:
                contacts.append(
                    {
                        "residue_number_1_based": residue_number,
                        "residue_index_0_based": residue_number - 1,
                        "minimum_heavy_atom_distance_angstrom": round(distance, 3),
                    }
                )
        copies.append(
            {
                "chain": chain,
                "ligand_residue": ligand_residue,
                "ligand_altloc": altloc or ".",
                "contact_count": len(contacts),
                "contacts": contacts,
            }
        )
    if not copies:
        raise ValueError(f"No LIG copies found in {path}")
    return copies


def union_indices(structures: list[dict[str, object]]) -> list[int]:
    return sorted(
        {
            int(contact["residue_index_0_based"])
            for structure in structures
            for copy in structure["copies"]
            for contact in copy["contacts"]
        }
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--structure-dir", type=Path, default=ROOT / "data/structures/cache7"
    )
    parser.add_argument("--direct-cutoff", type=float, default=6.0)
    parser.add_argument("--context-cutoff", type=float, default=8.0)
    parser.add_argument(
        "--active-site",
        type=Path,
        default=ROOT / "artifacts/pgk2_structural_pilot/active_site_union.json",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "artifacts/pgk2_inhibitor_pocket/inhibitor_pocket.json",
    )
    args = parser.parse_args()
    if not 0 < args.direct_cutoff < args.context_cutoff:
        raise ValueError("Require 0 < direct-cutoff < context-cutoff")
    paths = [args.structure_dir / "PGK2_cmp21.pdb", args.structure_dir / "PGK2_cmp47.pdb"]
    for path in (*paths, args.active_site):
        if not path.exists():
            raise FileNotFoundError(path)

    direct_structures = [
        {"structure": path.name, "copies": ligand_contact_copies(path, args.direct_cutoff)}
        for path in paths
    ]
    context_structures = [
        {"structure": path.name, "copies": ligand_contact_copies(path, args.context_cutoff)}
        for path in paths
    ]
    direct = union_indices(direct_structures)
    context = union_indices(context_structures)
    if not set(direct).issubset(context):
        raise RuntimeError("Direct inhibitor pocket is not contained in the context shell")
    active_site = json.loads(args.active_site.read_text())
    pg3 = sorted(
        int(row["human_index_0_based"]) for row in active_site["subsites"]["3PG"]
    )
    result = {
        "definition": "Known orthosteric-inhibitor envelope from human PGK2 compounds 21 and 47",
        "direct_ligand_edges": {
            "cutoff_angstrom": args.direct_cutoff,
            "residue_indices_0_based": direct,
            "residue_numbers_1_based": [index + 1 for index in direct],
            "residue_count": len(direct),
            "structures": direct_structures,
        },
        "protein_context_shell": {
            "cutoff_angstrom": args.context_cutoff,
            "residue_indices_0_based": context,
            "residue_numbers_1_based": [index + 1 for index in context],
            "residue_count": len(context),
            "structures": context_structures,
        },
        "auxiliary_3pg_subsite": {
            "role": "Catalytic context; not required as a direct ligand-contact region",
            "residue_indices_0_based": pg3,
            "residue_numbers_1_based": [index + 1 for index in pg3],
            "residue_count": len(pg3),
        },
        "template_ensemble": [str(path.resolve()) for path in paths],
        "limitations": [
            "Contact sets are coordinate-derived neighborhoods, not energetic interactions.",
            "The 8 A shell is intended for protein context, not unrestricted direct ligand edges.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "direct_residues": len(direct),
                "context_residues": len(context),
                "auxiliary_3pg_residues": len(pg3),
                "output": str(args.output),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
