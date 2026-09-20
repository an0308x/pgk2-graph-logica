from pathlib import Path
import sys
import numpy as np
import polars as pl
import pytest
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"src"),str(ROOT/"scripts"),str(ROOT/"tests")]
from logica_binding.failure_audit import score_components,ligand_only_scores,spearman,pair_diagnostics
from logica_binding.streaming_graph_logica import packed_logica_scores
from test_streaming_graph_logica import fixture
from prepare_pgk2_failure_audit import properties


def test_component_decomposition_matches_production_and_controls_do_not_mutate():
    model,pocket,_,batch,p=fixture()
    arguments=(model,p.protein_hidden,p.drug_hidden,p.drug_attention_mask,p.drug_original_ids,
        p.drug_selected,p.protein_original_ids,[pocket],batch)
    with torch.no_grad():
        original=packed_logica_scores(*arguments)
        parts=score_components(*arguments)
        torch.testing.assert_close(parts["full"],original,atol=2e-6,rtol=2e-6)
        torch.testing.assert_close(parts["full"],.5*(parts["protein_component"]+parts["ligand_component"]))
        for options in ({"drop_contacts":True},{"drop_cross":True},{"use_graph":False}):
            result=score_components(*arguments,**options)
            assert torch.isfinite(result["full"]).all()
        torch.testing.assert_close(packed_logica_scores(*arguments),original)
        assert model.graph_conditioning


def test_no_cross_and_no_contacts_ligand_component_is_protein_independent():
    model,pocket,_,batch,p=fixture()
    with torch.no_grad():
        first=score_components(model,p.protein_hidden,p.drug_hidden,p.drug_attention_mask,
            p.drug_original_ids,p.drug_selected,p.protein_original_ids,[pocket],batch,
            drop_contacts=True,drop_cross=True)
        second=score_components(model,p.protein_hidden*10,p.drug_hidden,p.drug_attention_mask,
            p.drug_original_ids,p.drug_selected,p.protein_original_ids,[pocket],batch,
            drop_contacts=True,drop_cross=True)
        torch.testing.assert_close(first["ligand_component"],second["ligand_component"])
        null=ligand_only_scores(model,p.drug_hidden,p.drug_original_ids,p.drug_selected,batch)
        assert set(null)=={"ligand_lm_only","ligand_graph_only"}
        assert all(torch.isfinite(v).all() for v in null.values())


def test_rank_correlation_handles_ties_and_constants():
    assert spearman([1,1,3],[1,1,3])==pytest.approx(1.)
    assert spearman([1,2,3],[3,2,1])==pytest.approx(-1.)
    assert spearman([1,1,1],[1,2,3]) is None


def test_count_strata_identify_inhibitor_only_discrimination():
    f=pl.DataFrame({"source_rows":[1,1],"count_PGK2":[1,1],"count_PGK2_with_inhibitor":[1,10],
        "proxy_0":[0.,float(np.log(2)-np.log(11))],"confidence_0":[.2,.2]})
    d=pair_diagnostics(f,{"model":np.array([2.,1.])})
    assert d["equal_target_counts"]["pairs"]==1 and d["both_proxy_nonpositive"]["pairs"]==1
    assert d["preferred_target_at_most_one"]["pairs"]==1
    assert d["all"]["weighted_wins"]["target_count_only"]==pytest.approx(.1)
    assert d["all"]["weighted_wins"]["inverse_inhibitor_count_only"]==pytest.approx(.2)
    assert d["both_target_at_least_five"]["pairs"]==0


def test_chemical_descriptors_have_expected_units():
    d=properties("CCO")
    assert d["heavy_atoms"]==3 and 46<d["mol_weight"]<47
    assert d["tpsa"]==pytest.approx(20.23,abs=.01)
