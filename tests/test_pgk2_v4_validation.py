from pathlib import Path
import sys
import copy
import polars as pl
import pytest
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"src"),str(ROOT/"scripts"),str(ROOT/"tests")]
from score_pgk2_v4_validation import pack_smiles,validate_checkpoint
from merge_pgk2_v4_validation import checked_ranking
from logica_binding.development_protocol import PROTOCOL
from test_streaming_graph_logica import fixture


def test_fresh_inference_graphs_match_training_packing():
    _,_,_,expected,_=fixture()
    actual=pack_smiles(["CCO","[Na+]","c1ccccc1"])
    for key in ("node_features","edge_features","edge_index","node_offsets"):
        torch.testing.assert_close(getattr(actual,key),getattr(expected,key))


def test_only_selected_production_checkpoint_allowed():
    identity={"config":{"model":"graph","zero_cap":5,"auxiliary_weight":0.,"seed":2026}}
    saved={"protocol":PROTOCOL,"identity":identity,"selected_step":2198,"counts":{"optimizer_steps":2198}}
    report={"protocol":PROTOCOL,"identity":identity,"selected_step":2198,"smoke_only":False,
        "history":[{"selected":True,"counts":{"optimizer_steps":2198},"dev":{"selection_loss":.0013}}]}
    assert validate_checkpoint(saved,report)=={"selection_loss":.0013}
    bad=copy.deepcopy(report); bad["smoke_only"]=True
    with pytest.raises(ValueError): validate_checkpoint(saved,bad)
    bad=copy.deepcopy(saved); bad["counts"]["optimizer_steps"]=100
    with pytest.raises(ValueError): validate_checkpoint(bad,report)


def test_submission_order_is_highest_score_with_strict_id_coverage():
    candidates=[{"CatalogID":x} for x in ["B","A","C"]]
    frame=pl.DataFrame({"row_index":[2,0,1],"CatalogID":["C","B","A"],"score":[-3.,-1.,-1.]})
    assert checked_ranking(frame,candidates)["CatalogID"].to_list()==["A","B","C"]
    with pytest.raises(ValueError): checked_ranking(frame.head(2),candidates)
    with pytest.raises(ValueError): checked_ranking(frame.with_columns(pl.lit(float("nan")).alias("score")),candidates)
    with pytest.raises(ValueError): checked_ranking(frame.with_columns(pl.lit("A").alias("CatalogID")),candidates)
