from pathlib import Path
import sys

import numpy as np
import polars as pl
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(ROOT / "tests")]
from test_pgk2_audit_repairs import tiny_base
from logica_binding.graph_model import GraphLogiCAModel, build_pocket_graph, smiles_to_ligand_graph, score_graph_prepared
from logica_binding.full_graph_model import PackedLigandBatch
from logica_binding.model import PreparedPairs
from logica_binding.streaming_graph_logica import observed_evidence, evidence_pair_loss, packed_logica_scores, ligand_reconstruction_loss
from logica_binding.validation_protocol import structure_identity
from prepare_pgk2_full_logica_manifest import scaffold_key


def fixture():
    torch.manual_seed(12)
    model = GraphLogiCAModel(tiny_base(), 16).eval()
    model.base.pair_alpha_logit = torch.nn.Parameter(torch.zeros(1))
    pocket = build_pocket_graph(ROOT / "data/structures/cache7/PGK2_cmp21.pdb",
                                ROOT / "artifacts/pgk2_inhibitor_pocket/inhibitor_pocket.json")
    smiles = ["CCO", "[Na+]", "c1ccccc1"]
    graphs = [smiles_to_ligand_graph(s) for s in smiles]
    offsets = np.cumsum([0] + [len(g.node_features) for g in graphs])
    packed = PackedLigandBatch(torch.cat([g.node_features for g in graphs]),
        torch.cat([g.edge_index + int(offsets[i]) for i, g in enumerate(graphs)], 1),
        torch.cat([g.edge_features for g in graphs]), torch.tensor(offsets))
    prepared = PreparedPairs(tuple(smiles), torch.randn(1, 419, 16), torch.ones(1, 419),
        torch.randint(0, 10, (1, 419)), torch.zeros(1, 419, dtype=torch.bool),
        torch.randn(3, 7, 16), torch.tensor([[1]*5+[0]*2, [1]*3+[0]*4, [1]*7]),
        torch.randint(0, 10, (3, 7)), torch.tensor([[True]*5+[False]*2, [True]*3+[False]*4, [True]*7]))
    prepared.protein_selected[:, pocket.residue_indices[pocket.direct_mask] + 1] = True
    return model, pocket, graphs, packed, prepared


@pytest.mark.parametrize("graph_on", [True, False])
def test_packed_scores_match_reference(graph_on):
    model, pocket, graphs, batch, p = fixture()
    model.graph_conditioning = graph_on
    reference = score_graph_prepared(model, p, torch.arange(3), [pocket], graphs)
    result = packed_logica_scores(model, p.protein_hidden, p.drug_hidden, p.drug_attention_mask,
        p.drug_original_ids, p.drug_selected, p.protein_original_ids, [pocket], batch)
    torch.testing.assert_close(result, reference, atol=2e-6, rtol=2e-6)
    parameter = model.base.drug_proj_in.weight
    expected_grad = torch.autograd.grad(reference.sum(), parameter, retain_graph=True)[0]
    actual_grad = torch.autograd.grad(result.sum(), parameter)[0]
    torch.testing.assert_close(actual_grad, expected_grad, atol=2e-6, rtol=2e-5)


def test_ligand_auxiliary_does_not_train_protein_as_positive():
    model, _, _, batch, p = fixture()
    loss = ligand_reconstruction_loss(model, p.drug_hidden, p.drug_original_ids, p.drug_selected, batch)
    loss.backward()
    assert model.conditioner.ligand_encoder.input.weight.grad.abs().sum() > 0
    assert model.conditioner.protein_context_in.weight.grad is None
    assert model.base.prot_proj_in.weight.grad is None
    assert all(v.grad is None for v in model.base.cross_attn.parameters())


def test_missing_controls_give_no_ranking_supervision():
    frame = pl.DataFrame({"source_rows": [1, 1], "count_PGK2": [1, 100],
        "count_PGK2_with_inhibitor": [0, 0], "count_NTC_selection": [0, 0],
        "count_NTC_supplement": [0, 0], "ntc_supplement_rows": [0, 0], "historic_hits": [0, 0]})
    proxy, confidence = observed_evidence(frame)
    assert not confidence.any()
    scores = torch.tensor([0., 1.], requires_grad=True)
    loss, stats = evidence_pair_loss(scores, torch.from_numpy(proxy), torch.from_numpy(confidence))
    loss.backward()
    assert loss.item() == 0 and all(s["pairs"] == 0 for s in stats)
    assert not scores.grad.any()


def test_ranking_gradient_and_absolute_confidence():
    scores = torch.tensor([0., 0.], requires_grad=True)
    proxy = torch.tensor([[2., 0., 0.], [0., 0., 0.]])
    confidence = torch.tensor([[1., 0., 0.], [1., 0., 0.]])
    loss, stats = evidence_pair_loss(scores, proxy, confidence)
    weaker, _ = evidence_pair_loss(scores, proxy, confidence * .1)
    torch.testing.assert_close(weaker, loss * .1)
    loss.backward()
    assert scores.grad[0] < 0 < scores.grad[1]
    assert stats[0]["weighted_wins"] == .5


def test_overlay_scaffolds_match_shared_protocol():
    for smiles in ["C[C@H](O)F", "c1ccccc1", "C[C@@H]1CCCCO1"]:
        from rdkit import Chem
        from rdkit.Chem.Scaffolds import MurckoScaffold
        scaffold = Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(Chem.MolFromSmiles(smiles)))
        assert scaffold_key((scaffold or smiles, not bool(scaffold))) == structure_identity(smiles)[1]


def test_real_streaming_trainer_smoke(tmp_path, monkeypatch):
    """Opt-in offline integration check against the actual released checkpoint."""
    import os
    if os.environ.get("PGK2_REAL_SMOKE") != "1":
        pytest.skip("Set PGK2_REAL_SMOKE=1 to run the local pretrained-model integration check")
    import hashlib
    import json
    from logica_binding.validation_protocol import group_split
    from logica_binding.streaming_graph_logica import FULL_PROTOCOL
    from prepare_pgk2_full_logica_manifest import sha256
    from run_pgk2_full_logica import main

    rows, graphs, per_split = [], [], {s: 0 for s in ("train", "dev", "holdout")}
    for n in range(2, 200):
        smi = "C" * n
        split = group_split(structure_identity(smi)[1])
        limit = 12 if split == "train" else 4
        if per_split[split] >= limit:
            continue
        sign = 1 if per_split[split] % 2 else -1
        rows.append({"molecule_id": len(rows), "canonical_smiles": smi, "split": split,
                     **{f"proxy_{i}": float(sign * 2) for i in range(3)},
                     **{f"confidence_{i}": .2 for i in range(3)}})
        graphs.append(smiles_to_ligand_graph(smi)); per_split[split] += 1
        if sum(per_split.values()) == 20:
            break
    assert sum(per_split.values()) == 20
    metadata = tmp_path / "metadata.parquet"
    pl.DataFrame(rows).write_parquet(metadata)
    archive = tmp_path / "graphs.npz"
    node_offsets = np.cumsum([0]+[len(g.node_features) for g in graphs])
    edge_offsets = np.cumsum([0]+[g.edge_index.shape[1] for g in graphs])
    np.savez_compressed(archive, node_offsets=node_offsets, edge_offsets=edge_offsets,
        node_features=torch.cat([g.node_features for g in graphs]).numpy(),
        edge_index=torch.cat([g.edge_index for g in graphs], 1).numpy(),
        edge_features=torch.cat([g.edge_features for g in graphs]).numpy())
    manifest = {"protocol": FULL_PROTOCOL, "completed": True, "graph_mode": "2d",
        "zscore_columns_used": False, "molecules": 20, "synthetic_fixture": True,
        "splits": {s: {"molecules": n} for s,n in per_split.items()},
        "shards": [{"range": [0, 20], "metadata": metadata.name,
            "metadata_sha256": sha256(metadata), "graphs": str(archive), "graphs_sha256": sha256(archive)}]}
    manifest["manifest_sha256"] = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    output = tmp_path / "run"
    monkeypatch.setattr(sys, "argv", ["train", "--manifest", str(manifest_path),
        "--output-dir", str(output), "--device", "cpu", "--batch-size", "4",
        "--rank-batch-size", "4", "--rank-every", "1", "--max-train-batches", "2",
        "--max-eval-batches", "3", "--threads", "2"])
    main()
    report = json.loads((output / "report.json").read_text())
    assert report["smoke_only"] and not report["candidate_scoring_allowed"]
    assert report["counts"]["train_molecule_visits"] == 8
    assert report["counts"]["optimizer_steps"] == 2
    assert report["selected_epoch"] == 1
    assert (output / "last.pt").exists() and (output / "best.pt").exists()
    # Simulate interruption after the final training snapshot, before the report
    # was saved. Resume must evaluate that state, not take an extra update.
    (output / "report.json").rename(output / "before_resume_report.json")
    monkeypatch.setattr(sys, "argv", [*sys.argv, "--resume"])
    main()
    resumed = json.loads((output / "report.json").read_text())
    assert resumed["counts"]["optimizer_steps"] == 2
    assert resumed["counts"]["train_molecule_visits"] == 8
    assert resumed["holdout"] == report["holdout"]


def test_full_launch_guard():
    import copy
    from check_pgk2_full_logica_smoke import check
    from logica_binding.streaming_graph_logica import FULL_PROTOCOL
    phase = {"channels": {"competition": {"pairs": 10, "weight_sum": 1}}, "selection_loss": .01}
    report = {"protocol": FULL_PROTOCOL, "smoke_only": True,
        "identity": {"manifest": "same", "config": {"device": "cuda", "batch_size": 64,
            "rank_batch_size": 32, "rank_every": 4}},
        "counts": {"optimizer_steps": 64, "train_molecule_visits": 4096},
        "training_ranking": phase, "history": [{"dev": phase}], "holdout": phase,
        "train_molecules_per_second": 100., "peak_gpu_memory_bytes": 4 * 2**30}
    manifest = {"manifest_sha256": "same", "splits": {"train": {"molecules": 6026213}}}
    result = check(report, manifest)
    assert result["execution_ready"] and not result["predictive_performance_validated"]
    assert result["recommended_walltime_minutes"] == 1620
    bad = copy.deepcopy(report); bad["identity"]["config"]["device"] = "cpu"
    with pytest.raises(ValueError):
        check(bad, manifest)
    bad = copy.deepcopy(report); bad["counts"]["optimizer_steps"] = 2
    with pytest.raises(ValueError):
        check(bad, manifest)
    bad = copy.deepcopy(report); bad["holdout"]["channels"]["competition"]["pairs"] = 0
    with pytest.raises(ValueError):
        check(bad, manifest)
