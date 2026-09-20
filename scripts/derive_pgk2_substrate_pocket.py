#!/usr/bin/env python3
"""Substrate-only residue masks; leave every historical pocket artifact unchanged."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from derive_pgk2_active_site import AA3_TO_1, global_alignment_map


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parse_reference(path):
    seqres = defaultdict(list)
    protein = defaultdict(lambda: defaultdict(list))
    ligands = defaultdict(list)
    models = 0
    for line in Path(path).read_text().splitlines():
        if line.startswith("MODEL "):
            models += 1
            if models > 1:
                raise ValueError("Multiple models need explicit selection")
        if line.startswith("SEQRES"):
            seqres[line[11]].extend(line[19:].split())
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        if line[16] not in (" ", "A") or float(line[54:60]) <= 0:
            continue
        element = (line[76:78].strip() or line[12:16].strip()[0]).upper()
        if element in ("H", "D"):
            continue
        if line[26].strip():
            raise ValueError("Insertion code requires explicit mapping")
        chain, number, name = line[21], int(line[22:26]), line[17:20].strip()
        xyz = [float(line[start:start+8]) for start in (30, 38, 46)]
        if line.startswith("ATOM  "):
            protein[chain][(number, name)].append(xyz)
        elif name in ("ATP", "3PG"):
            ligands[(chain, number, name)].append(xyz)
    sequences = {c: "".join(AA3_TO_1[a] for a in names) for c, names in seqres.items()}
    # 2PAA author numbering follows its 416-aa SEQRES; do not silently assume
    # this for another structure. Missing coordinates are allowed and recorded.
    for chain, residues in protein.items():
        for (number, name) in residues:
            if not 1 <= number <= len(sequences[chain]) or sequences[chain][number-1] != AA3_TO_1[name]:
                raise ValueError(f"Author/SEQRES mismatch: {chain}:{number}:{name}")
    return sequences, protein, ligands


def derive(path, human_sequence, direct_cutoff=6., context_cutoff=8.):
    if not 0 < direct_cutoff < context_cutoff:
        raise ValueError("Require 0 < direct cutoff < context cutoff")
    sequences, protein, ligands = parse_reference(path)
    expected = {("A", 500, "ATP"), ("B", 501, "3PG")}
    if set(ligands) != expected:
        raise ValueError(f"Unexpected substrate identities: {set(ligands)}")
    mapping = {c: global_alignment_map(seq, human_sequence) for c, seq in sequences.items()}
    for c, seq in sequences.items():
        if len(mapping[c]) != len(seq) or len(set(mapping[c].values())) != len(seq):
            raise ValueError("Incomplete/noninjective mouse-to-human mapping")
    contacts = {}; context = set(); direct = set(); alignments = {}
    for c, seq in sequences.items():
        alignments[c] = {
            "seqres_length": len(seq), "mapped_residues": len(mapping[c]),
            "identical_residues": sum(seq[i] == human_sequence[j] for i,j in mapping[c].items()),
            "missing_coordinate_author_residues": sorted(set(range(1,len(seq)+1))-{n for n,_ in protein[c]}),
        }
    for chain, number, name in sorted(ligands):
        ligand = np.asarray(ligands[(chain,number,name)],dtype=float)
        rows = []
        for (resnum,resname), points in sorted(protein[chain].items()):
            distance = float(np.linalg.norm(np.asarray(points)[:,None,:]-ligand[None,:,:],axis=-1).min())
            if distance > context_cutoff:
                continue
            human_index = mapping[chain][resnum-1]
            context.add(human_index)
            is_direct = distance <= direct_cutoff
            if is_direct:
                direct.add(human_index)
            rows.append({"reference_chain":chain,"reference_author_residue":resnum,
                "reference_residue":resname,"human_index_0_based":human_index,
                "human_residue_number_1_based":human_index+1,"human_residue":human_sequence[human_index],
                "sequence_match":AA3_TO_1[resname] == human_sequence[human_index],
                "minimum_heavy_atom_distance_angstrom":distance,"direct":is_direct})
        if not any(r["direct"] for r in rows):
            raise ValueError("Empty substrate neighborhood")
        contacts[name] = {"ligand_chain":chain,"ligand_author_residue":number,
            "ligand_heavy_atoms":len(ligand),"contacts":rows}
    return contacts, sorted(direct), sorted(context), alignments


def region(indices, cutoff=None):
    out = {"residue_indices_0_based":sorted(set(indices)),
           "residue_numbers_1_based":[i+1 for i in sorted(set(indices))]}
    if cutoff is not None:
        out["cutoff_angstrom"] = cutoff
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference",type=Path,default=ROOT/"artifacts/pgk2_structural_pilot/2PAA_mouse_PGK2_ATP_3PG.pdb")
    p.add_argument("--human-fasta",type=Path,default=ROOT/"data/proteins/PGK2_P07205.fasta")
    p.add_argument("--old-pocket",type=Path,default=ROOT/"artifacts/pgk2_inhibitor_pocket/inhibitor_pocket.json")
    p.add_argument("--output-dir",type=Path,required=True)
    a=p.parse_args()
    if a.output_dir.exists():
        raise FileExistsError(a.output_dir)
    human="".join(l.strip() for l in a.human_fasta.read_text().splitlines() if not l.startswith(">"))
    if len(human)!=417:
        raise ValueError("Unexpected human PGK2 length")
    contacts,direct,context,alignment=derive(a.reference,human)
    old=json.loads(a.old_pocket.read_text())
    old_direct=old["direct_ligand_edges"]["residue_indices_0_based"]
    common=sorted(set(context)|set(old["protein_context_shell"]["residue_indices_0_based"])|set(old_direct))
    pg3=[r["human_index_0_based"] for r in contacts["3PG"]["contacts"] if r["direct"]]
    definition={"schema":"pgk2_substrate_mask_v1","definition":"ATP/3-PG heavy-atom neighborhoods only; no inhibitor coordinates used to select direct residues",
        "reference":{"pdb_id":"2PAA","species":"Mus musculus","path":str(a.reference.relative_to(ROOT)),"sha256":digest(a.reference)},
        "human_fasta_sha256":digest(a.human_fasta),"human_sequence_length":len(human),
        "direct_ligand_edges":region(direct,6.),"protein_context_shell":region(context,8.),
        "auxiliary_3pg_subsite":region(pg3),"subsites":contacts,"alignment":alignment,
        "limitations":["Substrate contact neighborhoods are mapped by sequence from mouse to human, not measured in a human substrate complex.",
            "ATP is on chain A; 3PG is on chain B. Contact distances are computed within each chain's own frame; chains are never spatially merged.",
            "The radius defines a prior, not energetic binding contacts or evidence of inhibition.",
            "No candidate ligand pose or protein-ligand distances are created."]}
    controlled=dict(definition,protein_context_shell=region(common))
    controlled["context_provenance"]="Common union of substrate 8A context and historical inhibitor 8A context, shared between controlled graph arms"
    narrow=dict(controlled,definition="Historical inhibitor direct mask with exactly the same context and geometry as the substrate comparison arm",
        direct_ligand_edges=region(old_direct,6.),historical_direct_mask_sha256=digest(a.old_pocket))
    # Same human geometry in both arms isolates the edge mask from species and
    # conformation changes. Ligands in these structures do not define new mask.
    sys.path.insert(0,str(ROOT/"src"))
    from logica_binding.graph_model import _parse_pdb_residues, THREE_TO_ONE
    coverage={}
    for label in (21,47):
        template=ROOT/f"data/structures/cache7/PGK2_cmp{label}.pdb"
        residues=_parse_pdb_residues(template,"A")
        missing=[i+1 for i in common if i+1 not in residues]
        mismatch=[i+1 for i in common if i+1 in residues and THREE_TO_ONE[residues[i+1]["name"]]!=human[i]]
        if missing or mismatch:
            raise ValueError(f"Human template mapping failed: {label}, missing={missing}, mismatch={mismatch}")
        coverage[str(label)]={"sha256":digest(template),"chain":"A","covered_context_residues":len(common),"missing":missing,"sequence_mismatches":mismatch}
    summary={"substrate_direct":len(direct),"substrate_context":len(context),"common_context":len(common),
        "ATP_direct":sum(r["direct"] for r in contacts["ATP"]["contacts"]),"3PG_direct":len(pg3),
        "historical_direct":len(old_direct),"shared_direct":len(set(direct)&set(old_direct)),
        "added_human_residues_1_based":[i+1 for i in sorted(set(direct)-set(old_direct))],
        "removed_human_residues_1_based":[i+1 for i in sorted(set(old_direct)-set(direct))],
        "human_geometry_coverage":coverage,"training_launched":False,
        "distinction":"Direct mask is inhibitor-independent. Matched comparison geometry/context are not inhibitor-structure-independent."}
    a.output_dir.mkdir(parents=True)
    for name,payload in [("substrate_pocket.json",definition),("substrate_controlled_context.json",controlled),
                         ("inhibitor_controlled_context.json",narrow),("report.json",summary)]:
        (a.output_dir/name).write_text(json.dumps(payload,indent=2)+"\n")
    print(json.dumps(summary,indent=2))


if __name__=="__main__":
    main()
