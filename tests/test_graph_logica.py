from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest
import torch
from rdkit import Chem
from rdkit.Chem import AllChem

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from logica_binding.graph_model import (
    GraphLogiCAConditioner,
    build_boltz_complex_graphs,
    build_candidate_interaction_graph,
    build_contact_graph,
    build_pocket_graph,
    smiles_to_ligand_graph,
    smiles_to_rdkit_3d_graph,
    smiles_to_rdkit_3d_graph_with_quality,
)


POCKET_JSON = ROOT / "artifacts" / "pgk2_inhibitor_pocket" / "inhibitor_pocket.json"
TEMPLATES = (
    ROOT / "artifacts" / "pgk2_structural_pilot" / "PGK2_cmp21_1.pdb",
    ROOT / "artifacts" / "pgk2_structural_pilot" / "PGK2_cmp47_1.pdb",
)


@pytest.mark.parametrize("template", TEMPLATES)
def test_pocket_graph_is_fixed_inhibitor_context(template: Path) -> None:
    graph = build_pocket_graph(template, POCKET_JSON)
    assert graph.node_features.shape == (49, 24)
    assert int(graph.direct_mask.sum()) == 26
    assert graph.edge_index.shape[0] == 2
    assert graph.edge_index.shape[1] > 49
    assert graph.edge_features.shape == (graph.edge_index.shape[1], 6)
    assert graph.residue_indices.min() >= 0
    assert graph.residue_indices.max() < 417


def test_ligand_graph_preserves_atoms_and_directed_bonds() -> None:
    smiles = "CC(=O)Nc1ccncc1"
    graph = smiles_to_ligand_graph(smiles)
    molecule = Chem.MolFromSmiles(smiles)
    assert graph.node_features.shape == (molecule.GetNumAtoms(), 20)
    assert graph.edge_index.shape[1] == 2 * molecule.GetNumBonds()
    assert graph.coordinates is None


def test_contacts_reject_smiles_without_pose() -> None:
    pocket = build_pocket_graph(TEMPLATES[0], POCKET_JSON)
    ligand = smiles_to_ligand_graph("CCO")
    with pytest.raises(ValueError, match="coordinates in the protein template frame"):
        build_contact_graph(pocket, ligand)


def test_rdkit_3d_graph_is_deterministic_and_has_radius_edges() -> None:
    smiles = "CC(=O)Nc1ccncc1"
    first = smiles_to_rdkit_3d_graph(smiles)
    second = smiles_to_rdkit_3d_graph(smiles)
    molecule = Chem.MolFromSmiles(smiles)
    assert first.coordinates is not None
    assert torch.allclose(first.coordinates, second.coordinates)
    assert first.edge_index.shape[1] > 2 * molecule.GetNumBonds()
    assert torch.isfinite(first.coordinates).all()


def test_rdkit_graph_falls_back_to_deterministic_2d(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(AllChem, "EmbedMolecule", lambda *_args, **_kwargs: -1)
    with pytest.warns(RuntimeWarning, match="deterministic 2-D coordinates"):
        graph, quality = smiles_to_rdkit_3d_graph_with_quality("CCO")
    assert quality == "2d_fallback"
    assert graph.coordinates is not None
    assert torch.isfinite(graph.coordinates).all()
    assert bool((graph.coordinates[:, 2] == 0).all())


def test_candidate_edges_are_complete_only_over_direct_pocket() -> None:
    pocket = build_pocket_graph(TEMPLATES[0], POCKET_JSON)
    ligand = smiles_to_rdkit_3d_graph("CCO")
    interactions = build_candidate_interaction_graph(pocket, ligand)
    assert interactions.edge_index.shape[1] == 26 * 3
    assert bool(pocket.direct_mask[interactions.edge_index[0]].all())
    assert torch.count_nonzero(interactions.edge_features[:, :4]) == 0
    assert bool((interactions.edge_features[:, 4] == 1).all())


def test_contacts_only_reach_direct_pocket_nodes() -> None:
    pocket = build_pocket_graph(TEMPLATES[0], POCKET_JSON)
    coordinates = np.repeat(pocket.coordinates[0:1].numpy(), 3, axis=0)
    ligand = smiles_to_ligand_graph("CCO", coordinates=coordinates)
    contacts = build_contact_graph(pocket, ligand, cutoff=100.0)
    assert contacts.edge_index.shape[1] == 26 * 3
    assert bool(pocket.direct_mask[contacts.edge_index[0]].all())


def test_conditioner_updates_only_mapped_protein_tokens() -> None:
    pocket = build_pocket_graph(TEMPLATES[0], POCKET_JSON)
    ligand = smiles_to_ligand_graph(
        "CCO", coordinates=np.repeat(pocket.coordinates[0:1].numpy(), 3, axis=0)
    )
    contact = build_contact_graph(pocket, ligand, cutoff=100.0)
    conditioner = GraphLogiCAConditioner(protein_width=32, drug_width=24, graph_width=16)
    protein = torch.zeros(1, 419, 32)
    drug = torch.zeros(1, 8, 24)
    protein_out, drug_out = conditioner(
        protein, drug, pocket, [ligand], [contact], require_contacts=True
    )
    mapped = torch.zeros(419, dtype=torch.bool)
    mapped[pocket.residue_indices + 1] = True
    assert torch.count_nonzero(protein_out[0, ~mapped]) == 0
    assert torch.count_nonzero(protein_out[0, mapped]) > 0
    assert torch.count_nonzero(drug_out) > 0


def test_pocket_graph_receives_full_sequence_esm_context() -> None:
    pocket = build_pocket_graph(TEMPLATES[0], POCKET_JSON)
    ligand = smiles_to_rdkit_3d_graph("CCO")
    interactions = build_candidate_interaction_graph(pocket, ligand)
    conditioner = GraphLogiCAConditioner(protein_width=32, drug_width=24, graph_width=16)
    conditioner.eval()
    protein_a = torch.zeros(1, 419, 32)
    protein_b = protein_a.clone()
    protein_b[0, pocket.residue_indices + 1] = 1.0
    drug = torch.zeros(1, 8, 24)
    output_a, _ = conditioner(
        protein_a, drug, pocket, [ligand], [interactions], require_contacts=True
    )
    output_b, _ = conditioner(
        protein_b, drug, pocket, [ligand], [interactions], require_contacts=True
    )
    assert not torch.allclose(
        output_a[0, pocket.residue_indices + 1],
        output_b[0, pocket.residue_indices + 1],
    )


def test_existing_boltz_pose_builds_aligned_complex_graphs() -> None:
    cif = next(
        (
            ROOT
            / "artifacts"
            / "pgk2_structural_pilot"
            / "boltz_runs"
            / "pgk2-pilot-01-z2286322383"
            / "outputs"
            / "files"
            / "prediction"
        ).glob("*_predicted.cif")
    )
    smiles = "FC(F)(F)C1CCC(Cc2nc(-c3ccnc4[nH]ccc34)no2)CC1"
    pocket, ligand, contacts = build_boltz_complex_graphs(cif, smiles, POCKET_JSON)
    assert pocket.node_features.shape[0] == 49
    assert ligand.node_features.shape[0] == 25
    assert contacts.edge_index.shape[1] > 0
    assert bool(pocket.direct_mask[contacts.edge_index[0]].all())
