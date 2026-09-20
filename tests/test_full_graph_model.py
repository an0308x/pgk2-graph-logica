from __future__ import annotations

from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from logica_binding.full_graph_model import (
    FingerprintMultitaskModel,
    PGK2PocketMultitaskModel,
    PackedLigandBatch,
)
from logica_binding.graph_model import build_pocket_graph, smiles_to_ligand_graph


def packed_ligands() -> PackedLigandBatch:
    graphs = [smiles_to_ligand_graph("CCO"), smiles_to_ligand_graph("c1ccccc1")]
    nodes = torch.cat([graph.node_features for graph in graphs])
    edges = []
    edge_features = []
    offsets = [0]
    for graph in graphs:
        edges.append(graph.edge_index + offsets[-1])
        edge_features.append(graph.edge_features)
        offsets.append(offsets[-1] + graph.node_features.shape[0])
    return PackedLigandBatch(
        node_features=nodes,
        edge_index=torch.cat(edges, dim=1),
        edge_features=torch.cat(edge_features),
        node_offsets=torch.tensor(offsets),
    )


def test_pocket_multitask_model_batches_ligands() -> None:
    definition = ROOT / "artifacts" / "pgk2_inhibitor_pocket" / "inhibitor_pocket.json"
    pockets = [
        build_pocket_graph(
            ROOT / "artifacts" / "pgk2_structural_pilot" / f"PGK2_cmp{compound}_1.pdb",
            definition,
        )
        for compound in (21, 47)
    ]
    model = PGK2PocketMultitaskModel(esm_width=16, output_dim=9, graph_width=32)
    output = model(packed_ligands(), pockets, torch.randn(417, 16))
    assert output.shape == (2, 9)
    assert torch.isfinite(output).all()
    output.sum().backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_ligand_only_and_fingerprint_controls() -> None:
    ligand = PGK2PocketMultitaskModel(
        esm_width=16, output_dim=9, graph_width=32, mode="ligand"
    )
    output = ligand(packed_ligands(), [], torch.empty(417, 16))
    assert output.shape == (2, 9)
    fingerprint = FingerprintMultitaskModel(2048, 9, width=32)
    assert fingerprint(torch.zeros(2, 2048)).shape == (2, 9)
