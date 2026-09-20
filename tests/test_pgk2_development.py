from pathlib import Path
import sys
import numpy as np
import polars as pl
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/"src"),str(ROOT/"scripts")]
from logica_binding.development_protocol import (evidence_intervals, interval_pair_loss,
    training_batches, select_checkpoint, gradient_comparison)
from logica_binding.streaming_graph_logica import observed_evidence, evidence_pair_loss


def frame():
    f = pl.DataFrame({"source_rows":[1]*6,"ntc_supplement_rows":[1]*6,
        "count_PGK2":[20,20,4,20,20,20],"count_PGK2_with_inhibitor":[10,0,0,0,0,0],
        "count_NTC_selection":[0,0,0,20,0,0],"count_NTC_supplement":[0]*6,
        "historic_hits":[0,0,0,0,9,0],"split":["train"]*5+["dev"]})
    y,w = observed_evidence(f)
    return f.with_columns(*[pl.Series(f"proxy_{i}",y[:,i]) for i in range(3)],
        *[pl.Series(f"confidence_{i}",w[:,i]) for i in range(3)])


def test_zero_policy_is_bounded_and_does_not_rewrite_observed_or_ntc():
    f = frame(); y,w = observed_evidence(f)
    lo,hi,c = evidence_intervals(f,0)
    np.testing.assert_array_equal(lo,y); np.testing.assert_array_equal(c,w)
    lo,hi,c = evidence_intervals(f,5)
    assert lo[1,0] == pytest.approx(np.log(21)-np.log(6))
    assert hi[1,0] == pytest.approx(np.log(21))
    assert 0<c[1,0]<.05 and c[2,0]==0
    assert c[3,0]==pytest.approx(c[1,0]/2)
    assert c[4,0]==pytest.approx(c[1,0]/10)
    np.testing.assert_array_equal(c[:,1:],w[:,1:])
    np.testing.assert_array_equal(c[0],w[0])
    np.testing.assert_array_equal(f["confidence_0"].to_numpy(),w[:,0])
    with pytest.raises(ValueError):
        evidence_intervals(f,3)


def test_interval_point_loss_matches_original_value_and_gradient():
    torch.manual_seed(3)
    y,w = torch.randn(12,3),torch.rand(12,3)
    s = torch.randn(12,requires_grad=True)
    old,old_stats = evidence_pair_loss(s,y,w)
    new,new_stats = interval_pair_loss(s,y,y,w)
    torch.testing.assert_close(new,old)
    torch.testing.assert_close(torch.autograd.grad(new,s,retain_graph=True)[0],torch.autograd.grad(old,s)[0])
    assert old_stats==new_stats


def test_overlapping_intervals_not_ranked_and_disjoint_direction_correct():
    s = torch.zeros(2,requires_grad=True)
    lo = torch.tensor([[2.,0,0],[1.,0,0]])
    hi = torch.tensor([[3.,0,0],[2.5,0,0]])
    w = torch.tensor([[.1,0,0],[.1,0,0]])
    loss,stats = interval_pair_loss(s,lo,hi,w)
    assert loss==0 and stats[0]["pairs"]==0
    hi[1,0]=1.
    loss,stats = interval_pair_loss(s,lo,hi,w)
    loss.backward()
    assert stats[0]["pairs"]==1 and s.grad[0]<0<s.grad[1]


def test_all_training_pool_rows_visited_no_dev_leakage_singleton_tail_retained():
    f = pl.DataFrame({"split":["train"]*65+["dev","holdout"]})
    confidence = np.zeros((67,3)); confidence[:,0]=1
    batches = list(training_batches(f,confidence,2026))
    assert [len(x) for x in batches]==[32,33]
    assert set(np.concatenate(batches))==set(range(65))
    for x,y in zip(batches,training_batches(f,confidence,2026)):
        np.testing.assert_array_equal(x,y)


def test_initial_checkpoint_survives_worse_and_tied_updates():
    best,stale,selected = select_checkpoint(.2,float("inf"),0)
    assert selected and stale==0
    for metric in (.3,.2,None,float("nan")):
        best,stale,selected = select_checkpoint(metric,best,stale)
        assert not selected and best==.2
    assert stale==4
    assert select_checkpoint(.1,best,stale)==(.1,0,True)


def test_gradient_diagnostics_include_shared_conflict_and_leave_backward_intact():
    a,b = torch.nn.Parameter(torch.ones(2)),torch.nn.Parameter(torch.ones(2))
    r = a.sum()+b.sum(); aux = -2*a.sum()
    d = gradient_comparison(r,aux,[a,b])
    assert d["shared_cosine"]==pytest.approx(-1.)
    assert d["shared_auxiliary_norm"]==pytest.approx(2*d["shared_ranking_norm"])
    assert a.grad is None and b.grad is None
    (r+aux).backward()
    torch.testing.assert_close(a.grad,-torch.ones(2))


@pytest.mark.parametrize("model,aux,cap",[("fingerprint",0,0),("graph",0,1),("graph",.01,5)])
def test_real_development_smoke(tmp_path,monkeypatch,model,aux,cap):
    import os
    if os.environ.get("PGK2_REAL_SMOKE")!="1":
        pytest.skip("Opt-in real checkpoint smoke")
    import hashlib,json
    from logica_binding.graph_model import smiles_to_ligand_graph
    from logica_binding.streaming_graph_logica import FULL_PROTOCOL
    from prepare_pgk2_full_logica_manifest import sha256
    from run_pgk2_development import main
    # Synthetic identities and split labels are ONLY for runtime testing.
    f = frame().with_columns(pl.Series("canonical_smiles",["CCO","CCN","CCC","CCCC","CCCO","CCCCO"]),
        pl.Series("molecule_id",list(range(6))),pl.Series("split",["train"]*3+["dev"]*2+["holdout"]))
    # Ensure two separated observed-control dev examples for selection.
    f = f.with_columns(pl.Series("confidence_0",[.2,.2,.2,.2,.2,.2]),
        pl.Series("proxy_0",[2.,-2.,1.,2.,-2.,1.]))
    path=tmp_path/"metadata.parquet"; f.write_parquet(path)
    graphs=[smiles_to_ligand_graph(s) for s in f["canonical_smiles"]]
    archive=tmp_path/"graphs.npz"
    np.savez_compressed(archive,node_offsets=np.cumsum([0]+[len(g.node_features) for g in graphs]),
        edge_offsets=np.cumsum([0]+[g.edge_index.shape[1] for g in graphs]),
        node_features=torch.cat([g.node_features for g in graphs]).numpy(),
        edge_index=torch.cat([g.edge_index for g in graphs],1).numpy(),
        edge_features=torch.cat([g.edge_features for g in graphs]).numpy(),
        fingerprints=np.ones((6,256),dtype=np.uint8))
    manifest={"protocol":FULL_PROTOCOL,"completed":True,"graph_mode":"2d","zscore_columns_used":False,
        "molecules":6,"synthetic_fixture":True,"splits":{s:{"molecules":n} for s,n in [("train",3),("dev",2),("holdout",1)]},
        "shards":[{"range":[0,6],"metadata":path.name,"metadata_sha256":sha256(path),
            "graphs":str(archive),"graphs_sha256":sha256(archive)}]}
    manifest["manifest_sha256"]=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()
    mp=tmp_path/"manifest.json"; mp.write_text(json.dumps(manifest))
    output=tmp_path/"output"
    monkeypatch.setattr(sys,"argv",["dev","--manifest",str(mp),"--output-dir",str(output),"--device","cpu",
        "--model",model,"--auxiliary-weight",str(aux),"--zero-cap",str(cap),"--passes","1",
        "--eval-every-shards","1","--max-batches-per-shard","1"])
    main()
    report=json.loads((output/"report.json").read_text())
    assert report["smoke_only"] and not report["holdout_evaluated"] and not report["candidate_scoring_allowed"]
    assert report["counts"]["optimizer_steps"]==1
    assert len(report["history"])==2 and report["history"][0]["label"]=="initial"
    assert report["selected_step"] in (0,1)
    if aux:
        assert report["gradient_diagnostics"] and report["counts"]["auxiliary_visits"]==12
