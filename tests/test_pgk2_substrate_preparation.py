import json
from pathlib import Path
import sys
import numpy as np
import polars as pl
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
from derive_pgk2_substrate_pocket import derive, parse_reference
from audit_pgk2_evidence_coverage import coverage, evidence_columns
from logica_binding.graph_model import build_pocket_graph

REFERENCE=ROOT/"artifacts/pgk2_structural_pilot/2PAA_mouse_PGK2_ATP_3PG.pdb"
HUMAN="".join(l.strip() for l in (ROOT/"data/proteins/PGK2_P07205.fasta").read_text().splitlines() if not l.startswith(">"))


def test_substrate_contacts_and_mapping():
    contacts,direct,context,alignment=derive(REFERENCE,HUMAN)
    assert len(direct)==43 and len(context)==78
    assert set(direct)<=set(context)
    assert contacts["ATP"]["ligand_chain"]=="A"
    assert contacts["3PG"]["ligand_chain"]=="B"
    assert contacts["ATP"]["ligand_heavy_atoms"]==31
    assert contacts["3PG"]["ligand_heavy_atoms"]==11
    assert all(r["reference_chain"]==c["ligand_chain"] for c in contacts.values() for r in c["contacts"])
    assert all(r["human_residue"]==HUMAN[r["human_index_0_based"]] for c in contacts.values() for r in c["contacts"])
    assert alignment["A"]["mapped_residues"]==416
    assert 372 in alignment["B"]["missing_coordinate_author_residues"]
    # Coordinates omitted in B must not erase the ATP contacts from A.
    assert 372 in [r["reference_author_residue"] for r in contacts["ATP"]["contacts"]]


def test_wrong_ligand_fails_closed(tmp_path):
    lines=REFERENCE.read_text().splitlines()
    changed=[l[:17]+"ADP"+l[20:] if l.startswith("HETATM") and l[17:20]=="ATP" else l for l in lines]
    target=tmp_path/"wrong.pdb"; target.write_text("\n".join(changed))
    with pytest.raises(ValueError,match="Unexpected substrate"):
        derive(target,HUMAN)


def test_same_context_human_geometry_both_masks():
    base=ROOT/"artifacts/pgk2_substrate_protocol_v1/pocket"
    broad=json.loads((base/"substrate_controlled_context.json").read_text())
    narrow=json.loads((base/"inhibitor_controlled_context.json").read_text())
    assert broad["protein_context_shell"]==narrow["protein_context_shell"]
    for label in (21,47):
        pdb=ROOT/f"data/structures/cache7/PGK2_cmp{label}.pdb"
        a=build_pocket_graph(pdb,base/"substrate_controlled_context.json")
        b=build_pocket_graph(pdb,base/"inhibitor_controlled_context.json")
        assert a.direct_mask.sum()==43 and b.direct_mask.sum()==26
        assert len(a.residue_indices)==90
        np.testing.assert_array_equal(a.coordinates,b.coordinates)
        np.testing.assert_array_equal(a.residue_indices,b.residue_indices)
        np.testing.assert_array_equal(a.edge_index,b.edge_index)
        # Direct-membership features can differ; geometry must not.
        np.testing.assert_array_equal(a.edge_features[:,:5],b.edge_features[:,:5])


def frame():
    return pl.DataFrame({"source_rows":[1,1,2,1],"count_PGK2":[1,10,40,10],
        "count_PGK2_with_inhibitor":[0,1,0,20],"count_NTC_selection":[0,0,2,0],
        "count_NTC_supplement":[0,0,0,0],"ntc_supplement_rows":[0,0,0,0],
        "historic_hits":[0,0,0,0],
        **{f"proxy_{i}":[0.,0.,0.,0.] for i in range(3)},
        **{f"confidence_{i}":[0.,1.,1.,1.] for i in range(3)}})


def test_uncertain_zeros_separate_from_observed_competition():
    f=frame(); c=coverage(f)
    assert c["molecules"]==4 and c["duplicate_source_molecules"]==1
    assert c["target_le1"]==1
    assert c["T5_joint_nonzero_inhibitor"]==1
    assert c["T5_joint_zero_inhibitor_uncertain"]==1
    assert c["T5_competition_retained_observed"]==1
    assert evidence_columns(f)[0].tolist()==[1,10,20,10]


def test_bad_counts_fail_closed():
    with pytest.raises(ValueError,match="Source-row denominator"):
        evidence_columns(frame().with_columns(pl.lit(0).alias("source_rows")))
    with pytest.raises(ValueError,match="Invalid"):
        evidence_columns(frame().with_columns(pl.lit(float("nan")).alias("count_PGK2")))
