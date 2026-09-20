from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest
import torch
from torch import nn
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from logica_binding.model import LogiCAModel, BidirectionalCrossAttention
from logica_binding.graph_model import GraphLogiCAModel, build_pocket_graph, smiles_to_ligand_graph, build_candidate_interaction_graph
from logica_binding.validation_protocol import structure_identity, group_split, validate_pair_splits, count_reliability
from run_pgk2_full_graph_ablation import spearman, roc_auc, average_precision


def tiny_base():
    # Real forward implementation; tiny heads avoid external model downloads.
    base = LogiCAModel.__new__(LogiCAModel)
    nn.Module.__init__(base)
    for name in ("esm", "drug_encoder"):
        head = nn.Module()
        head.config = SimpleNamespace(hidden_size=16)
        head.lm_head = nn.Linear(16, 10)
        setattr(base, name, head)
    for name in ("prot_proj_in", "prot_proj_out", "drug_proj_in", "drug_proj_out"):
        setattr(base, name, nn.Linear(16, 16, bias=False))
    base.prot_gate = nn.Parameter(torch.zeros(1))
    base.drug_gate = nn.Parameter(torch.zeros(1))
    base.cross_attn = BidirectionalCrossAttention(16, 4, 2)
    return base


@pytest.mark.parametrize("conditioning", [True, False])
def test_graph_wrapper_has_no_outside_context_bypass(conditioning):
    torch.manual_seed(7)
    pocket = build_pocket_graph(ROOT / "data/structures/cache7/PGK2_cmp21.pdb",
                                ROOT / "artifacts/pgk2_inhibitor_pocket/inhibitor_pocket.json")
    model = GraphLogiCAModel(tiny_base(), 16, conditioning).eval()
    ligand = smiles_to_ligand_graph("CCO")
    contacts = [build_candidate_interaction_graph(pocket, ligand)]
    protein = torch.randn(1, 419, 16, requires_grad=True)
    drug = torch.randn(1, 5, 16)
    pa, da = torch.ones(1, 419), torch.ones(1, 5)
    context = torch.zeros(419, dtype=torch.bool)
    context[pocket.residue_indices + 1] = True
    direct = pocket.residue_indices[pocket.direct_mask] + 1
    def run(p):
        prot, lig, _, _ = model.forward_from_hidden_with_graph(p, pa, drug, da, pocket, [ligand], contacts)
        return prot[:, direct], lig
    p1, d1 = run(protein)
    changed = protein.detach().clone()
    changed[:, ~context] = torch.randn_like(changed[:, ~context]) * 10
    p2, d2 = run(changed)
    torch.testing.assert_close(p1, p2)
    torch.testing.assert_close(d1, d2)
    (p1.square().sum() + d1.square().sum()).backward()
    assert torch.count_nonzero(protein.grad[:, ~context]) == 0
    assert torch.count_nonzero(protein.grad[:, direct]) > 0


def test_interface_blocks_both_directions_and_preserves_outside_tokens():
    torch.manual_seed(7)
    attention = BidirectionalCrossAttention(16, 4, 2).eval()
    p, d = torch.randn(1, 9, 16), torch.randn(1, 3, 16)
    allowed = torch.zeros(1, 9, dtype=torch.bool); allowed[:, 2:4] = True
    p2, d2 = attention(p, d, torch.zeros_like(allowed), torch.zeros(1, 3, dtype=torch.bool), allowed)
    torch.testing.assert_close(p2[:, ~allowed[0]], p[:, ~allowed[0]])
    changed = p.clone(); changed[:, ~allowed[0]] += torch.randn_like(changed[:, ~allowed[0]])
    _, d3 = attention(changed, d, torch.zeros_like(allowed), torch.zeros(1, 3, dtype=torch.bool), allowed)
    torch.testing.assert_close(d2, d3)


def test_metrics_handle_ties_and_permutation():
    labels = np.array([0, 0, 1, 1]); scores = np.ones(4)
    assert roc_auc(labels, scores) == .5
    assert average_precision(labels, scores) == .5
    x, y = np.array([0, 0, 0, 1, 1, 1]), np.arange(6)
    assert spearman(x, y) == pytest.approx(spearmanr(x, y).statistic)
    rng = np.random.default_rng(8)
    labels, scores = rng.integers(0, 2, 100), rng.integers(0, 4, 100)
    for _ in range(4):
        ix = rng.permutation(100)
        assert roc_auc(labels[ix], scores[ix]) == pytest.approx(roc_auc_score(labels, scores))
        assert average_precision(labels[ix], scores[ix]) == pytest.approx(average_precision_score(labels, scores))


def test_canonical_identity_and_cross_stage_split_do_not_depend_on_seed():
    assert structure_identity("OCC")[0] == structure_identity("CCO")[0]
    assert structure_identity("C[C@H](O)c1ccccc1")[1] == structure_identity("C[C@@H](O)c1ccccc1")[1]
    a, ga = structure_identity("CCO")
    b, gb = structure_identity("CCN")
    bad = "dev" if group_split(ga) != "dev" else "train"
    with pytest.raises(ValueError, match="shared"):
        validate_pair_splits([{"positive_smiles": a, "negative_smiles": b, "split": bad}])


def test_zero_controls_cannot_supply_confidence():
    r = {"positive_count_PGK2": 4, "negative_count_inhibitor": 0, "negative_count_NTC_evidence": 0}
    assert count_reliability(r) == 0
    r["negative_count_inhibitor"] = 1
    weak = count_reliability(r)
    r["positive_count_PGK2"] = 40
    assert 0 < weak < count_reliability(r) < 1
