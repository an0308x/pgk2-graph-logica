#!/usr/bin/env python3
"""Map ATP and 3-PG contact residues from mPGK2 2PAA onto human PGK2."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


def global_alignment_map(left: str, right: str) -> dict[int, int]:
    """Return an index map from left to right for a simple global alignment."""
    gap, match, mismatch = -2, 2, -1
    scores = [[0] * (len(right) + 1) for _ in range(len(left) + 1)]
    moves = [[""] * (len(right) + 1) for _ in range(len(left) + 1)]
    for i in range(1, len(left) + 1):
        scores[i][0], moves[i][0] = i * gap, "U"
    for j in range(1, len(right) + 1):
        scores[0][j], moves[0][j] = j * gap, "L"
    for i in range(1, len(left) + 1):
        for j in range(1, len(right) + 1):
            options = [
                (scores[i - 1][j - 1] + (match if left[i - 1] == right[j - 1] else mismatch), "D"),
                (scores[i - 1][j] + gap, "U"),
                (scores[i][j - 1] + gap, "L"),
            ]
            scores[i][j], moves[i][j] = max(options, key=lambda item: item[0])
    mapping: dict[int, int] = {}
    i, j = len(left), len(right)
    while i or j:
        move = moves[i][j]
        if move == "D":
            mapping[i - 1] = j - 1
            i, j = i - 1, j - 1
        elif move == "U":
            i -= 1
        elif move == "L":
            j -= 1
        else:
            raise RuntimeError("Incomplete alignment traceback")
    return mapping


def non_hydrogen_coords(line: str) -> tuple[float, float, float] | None:
    if line[76:78].strip() == "H":
        return None
    return tuple(float(line[start:end]) for start, end in ((30, 38), (38, 46), (46, 54)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pdb", type=Path,
        default=ROOT / "artifacts/pgk2_structural_pilot/2PAA_mouse_PGK2_ATP_3PG.pdb",
    )
    parser.add_argument(
        "--human-fasta", type=Path, default=ROOT / "data/proteins/PGK2_P07205.fasta"
    )
    parser.add_argument(
        "--human-atp-pocket-pdb", type=Path,
        default=ROOT / "artifacts/pgk2_structural_pilot/PGK2_cmp47_1.pdb",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "artifacts/pgk2_structural_pilot/active_site_union.json",
    )
    parser.add_argument("--cutoff", type=float, default=6.0)
    args = parser.parse_args()

    lines = args.pdb.read_text().splitlines()
    mouse_seqres: list[str] = []
    residue_atoms: dict[tuple[str, int, str], list[tuple[float, float, float]]] = defaultdict(list)
    atom_residues: list[tuple[int, str]] = []
    ligands = {
        "ATP": {"protein_chain": "A", "coords": []},
        "3PG": {"protein_chain": "B", "coords": []},
    }
    seen_residues: set[tuple[int, str]] = set()
    for line in lines:
        record = line[:6].strip()
        if record == "SEQRES" and line[11] == "A":
            mouse_seqres.extend(line[19:].split())
            continue
        if record not in {"ATOM", "HETATM"}:
            continue
        coord = non_hydrogen_coords(line)
        if coord is None:
            continue
        residue, chain, number = line[17:20].strip(), line[21], int(line[22:26])
        if record == "ATOM" and chain in {"A", "B"}:
            key = (chain, number, residue)
            residue_atoms[key].append(coord)
            if chain == "A":
                sequence_key = (number, residue)
                if sequence_key not in seen_residues:
                    atom_residues.append(sequence_key)
                    seen_residues.add(sequence_key)
        elif record == "HETATM" and residue in ligands:
            ligands[residue]["coords"].append(coord)
    mouse_sequence = "".join(AA3_TO_1[name] for name in mouse_seqres)
    atom_sequence = "".join(AA3_TO_1[name] for _, name in atom_residues)
    atom_to_seqres = global_alignment_map(atom_sequence, mouse_sequence)
    author_to_mouse_index = {
        atom_residues[atom_index]: seqres_index
        for atom_index, seqres_index in atom_to_seqres.items()
    }
    human_sequence = "".join(
        line.strip() for line in args.human_fasta.read_text().splitlines() if not line.startswith(">")
    )
    mouse_to_human = global_alignment_map(mouse_sequence, human_sequence)

    def mouse_contacts(ligand: str) -> list[dict[str, int | str | float]]:
        result = []
        ligand_coords = ligands[ligand]["coords"]
        protein_chain = ligands[ligand]["protein_chain"]
        for key, coordinates in residue_atoms.items():
            if key[0] != protein_chain:
                continue
            distance = min(
                math.dist(protein_atom, ligand_atom)
                for protein_atom in coordinates for ligand_atom in ligand_coords
            )
            if distance <= args.cutoff:
                mouse_index = author_to_mouse_index[(key[1], key[2])]
                human_index = mouse_to_human[mouse_index]
                result.append(
                    {
                        "mouse_author_residue": key[1],
                        "mouse_residue": key[2],
                        "minimum_heavy_atom_distance_angstrom": round(distance, 3),
                        "human_index_0_based": human_index,
                        "human_residue_number_1_based": human_index + 1,
                        "human_residue": human_sequence[human_index],
                    }
                )
        return sorted(result, key=lambda row: int(row["human_index_0_based"]))

    human_protein: dict[tuple[int, str], list[tuple[float, float, float]]] = defaultdict(list)
    human_ligand: list[tuple[float, float, float]] = []
    for line in args.human_atp_pocket_pdb.read_text().splitlines():
        record = line[:6].strip()
        if record not in {"ATOM", "HETATM"}:
            continue
        coord = non_hydrogen_coords(line)
        if coord is None:
            continue
        residue, chain, number = line[17:20].strip(), line[21], int(line[22:26])
        if record == "ATOM" and chain == "A" and number >= 1:
            human_protein[(number, residue)].append(coord)
        elif record == "HETATM" and chain == "A" and residue == "LIG":
            human_ligand.append(coord)
    human_atp_contacts = []
    for (number, residue), coordinates in human_protein.items():
        distance = min(
            math.dist(protein_atom, ligand_atom)
            for protein_atom in coordinates for ligand_atom in human_ligand
        )
        if distance <= args.cutoff:
            index = number - 1
            if index >= len(human_sequence):
                raise RuntimeError(f"Unexpected human reference residue number: {number}")
            human_atp_contacts.append(
                {
                    "human_author_residue": number,
                    "human_reference_residue": residue,
                    "minimum_heavy_atom_distance_angstrom": round(distance, 3),
                    "human_index_0_based": index,
                    "human_residue_number_1_based": number,
                    "human_residue": human_sequence[index],
                }
            )
    by_ligand = {
        "ATP_ADP": sorted(human_atp_contacts, key=lambda row: int(row["human_index_0_based"])),
        "3PG": mouse_contacts("3PG"),
    }
    union = sorted(
        {int(row["human_index_0_based"]) for rows in by_ligand.values() for row in rows}
    )
    if not union or not by_ligand["ATP_ADP"] or not by_ligand["3PG"]:
        raise RuntimeError("2PAA parsing failed to recover both catalytic ligands and their pocket")
    result = {
        "references": {
            "ATP_ADP": "CACHE hPGK2–compound-47 co-crystal; compound 47 occupies the adenine/ATP pocket.",
            "3PG": "RCSB 2PAA, mPGK2 bound to 3-PG; residues mapped by global sequence alignment to human PGK2 P07205.",
        },
        "pocket_definition": "Union of human ATP/ADP-pocket contacts and mouse 3-PG-pocket contacts, each within cutoff of ligand heavy atoms.",
        "cutoff_angstrom": args.cutoff,
        "mouse_pgk2_sequence_length": len(mouse_sequence),
        "human_pgk2_sequence_length": len(human_sequence),
        "subsites": by_ligand,
        "human_active_site_union_indices_0_based": union,
        "human_active_site_union_count": len(union),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
