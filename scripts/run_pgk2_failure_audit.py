#!/usr/bin/env python3
"""Score decomposition and inference-only controls for failed submission 9780912."""
import argparse
from collections import defaultdict
from dataclasses import replace
import json
from pathlib import Path
import sys
import time
import numpy as np
import polars as pl
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"src"),str(ROOT/"scripts")]
from logica_binding.failure_audit import score_components,ligand_only_scores,spearman,pair_diagnostics
from logica_binding.data import read_fasta
from logica_binding.graph_model import GraphLogiCAModel,build_pocket_graph
from logica_binding.model import LogiCABindingScorer
from logica_binding.pocket import load_pocket_region
from logica_binding.streaming_graph_logica import evidence_pair_loss,CHANNELS,packed_logica_scores
from run_pgk2_full_logica import encode_ligands,summarize_stats
from score_pgk2_v4_validation import pack_smiles,validate_checkpoint
from prepare_pgk2_full_logica_manifest import sha256


def control_summary(frame,columns):
    full=frame["full"].to_numpy(); n=min(50,frame.height)
    selected=set(np.argsort(-full,kind="stable")[:n])
    return {name:{"spearman_with_full":spearman(full,frame[name]),
        "std":float(np.std(frame[name].to_numpy())),
        "mean_absolute_change_from_full":float(np.abs(frame[name].to_numpy()-full).mean()),
        "top50_overlap_within_this_probe":len(selected&set(np.argsort(-frame[name].to_numpy(),kind="stable")[:n]))}
        for name in columns}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--audit-dir",type=Path,required=True)
    p.add_argument("--run-dir",type=Path,required=True)
    p.add_argument("--device",choices=("cpu","cuda"),default="cuda")
    a=p.parse_args(); torch.set_num_threads(4); torch.manual_seed(2026); started=time.monotonic()
    if (a.audit_dir/"report.json").exists(): raise FileExistsError("Completed failure audit exists")
    preparation=json.loads((a.audit_dir/"preparation_report.json").read_text())
    saved=torch.load(a.run_dir/"best.pt",map_location="cpu",weights_only=False)
    original=json.loads((a.run_dir/"report.json").read_text()); expected=validate_checkpoint(saved,original)
    if sha256(a.run_dir/"best.pt")!=preparation["checkpoint_sha256"]: raise ValueError("Wrong submitted checkpoint")
    checkpoint=ROOT/"models/logica-8m/checkpoints/8m/best.pt"
    if sha256(checkpoint)!=saved["identity"]["initial_checkpoint_sha256"]: raise ValueError("Base changed")
    for name in ("src/logica_binding/graph_model.py","src/logica_binding/model.py","src/logica_binding/streaming_graph_logica.py"):
        if sha256(ROOT/name)!=saved["identity"]["source_sha256"][name]: raise ValueError("Production source changed")
    definition=ROOT/"artifacts/pgk2_inhibitor_pocket/inhibitor_pocket.json"
    scorer=LogiCABindingScorer(checkpoint,hf_cache=ROOT/"models/hf-cache",device=a.device,
        score_mode="token_likelihood",protein_mask_positions=load_pocket_region(definition,"inhibitor_pocket"),
        protein_mask_region="inhibitor_pocket",seed=2026)
    pt=scorer.protein_tokenizer([read_fasta(ROOT/"data/proteins/PGK2_P07205.fasta")],return_tensors="pt")
    pi=pt["input_ids"].to(a.device)
    with torch.no_grad():
        ph=scorer.model.esm.base_model(input_ids=pi,attention_mask=pt["attention_mask"].to(a.device),return_dict=True).last_hidden_state
    pockets=[build_pocket_graph(ROOT/f"data/structures/cache7/PGK2_cmp{i}.pdb",definition).to(a.device) for i in (21,47)]
    null_pockets=[replace(p,node_features=torch.zeros_like(p.node_features),edge_features=torch.zeros_like(p.edge_features)) for p in pockets]
    shuffled=[]
    rng=np.random.default_rng(9780912)
    for pocket in pockets:
        order=torch.as_tensor(rng.permutation(len(pocket.edge_features)),device=a.device)
        shuffled.append(replace(pocket,edge_features=pocket.edge_features[order]))
    scorer.configure_finetuning(); model=GraphLogiCAModel(scorer.model).to(a.device)
    if set(saved["state"])!={n for n,v in model.named_parameters() if v.requires_grad}: raise ValueError("Parameter mismatch")
    model.load_state_dict(saved["state"],strict=False); model.eval()
    validation=pl.read_parquet(a.audit_dir/"validation_probe.parquet")
    dev=pl.read_parquet(a.audit_dir/"development_probe.parquet")
    outputs={}; truncated={}; columns=None; same_batch_max_error=0.
    for kind,frame in [("validation",validation),("development",dev)]:
        chunks=[]; truncations=0
        batches=[frame.slice(i,64) for i in range(0,frame.height,64)] if kind=="validation" else frame.partition_by("batch_id",maintain_order=True)
        with torch.no_grad():
            for batch_index,rows in enumerate(batches):
                smiles=rows["canonical_smiles"].to_list(); batch=pack_smiles(smiles).to(a.device)
                dh,da,di,ds,trunc=encode_ligands(scorer,smiles,False,2026); truncations+=trunc
                args=(model,ph,dh,da,di,ds,pi,pockets,batch)
                values=score_components(*args)
                production=packed_logica_scores(*args)
                same_batch_error=float((production-values["full"]).abs().max())
                same_batch_max_error=max(same_batch_max_error,same_batch_error)
                if same_batch_error>1e-6:
                    raise ValueError(f"Same-batch production decomposition mismatch: {same_batch_error}")
                values.update(ligand_only_scores(model,dh,di,ds,batch))
                values["no_graph_contacts"]=score_components(*args,drop_contacts=True)["full"]
                values["no_cross_attention"]=score_components(*args,drop_cross=True)["full"]
                values["graph_off_cross_attention_retained"]=score_components(*args,use_graph=False)["full"]
                values["zero_protein_context"]=score_components(model,torch.zeros_like(ph),dh,da,di,ds,pi,null_pockets,batch)["full"]
                values["shuffled_pocket_edge_features"]=score_components(model,ph,dh,da,di,ds,pi,shuffled,batch)["full"]
                columns=list(values)
                if not all(torch.isfinite(v).all() for v in values.values()): raise ValueError("Nonfinite diagnostic")
                chunks.append(rows.with_columns(*[pl.Series(name,v.cpu().numpy()) for name,v in values.items()],
                    pl.Series("ligand_tokens_capped",da.sum(1).cpu().numpy())))
                if batch_index%40==0: print(json.dumps({"phase":kind,"batch":batch_index,"total_batches":len(batches)}),flush=True)
        outputs[kind]=pl.concat(chunks); truncated[kind]=truncations
        outputs[kind].write_parquet(a.audit_dir/f"{kind}_diagnostic_scores.parquet")
    validation=outputs["validation"]; dev=outputs["development"]
    max_error=float(np.max(np.abs(validation["full"].to_numpy()-validation["score"].to_numpy())))
    # Recorded prior run had one cross-batch/hardware difference of 1.1444e-5,
    # with unchanged selected 50. Require a separate tighter same-batch check.
    if max_error>2e-5: raise ValueError(f"Submitted scores failed to reproduce: {max_error}")
    full_top=set(validation.sort("full",descending=True).head(50)["CatalogID"].to_list())
    saved_top=set(validation.sort("score",descending=True).head(50)["CatalogID"].to_list())
    if full_top!=saved_top: raise ValueError("Submitted top-50 changed during reproduction")
    metrics={name:defaultdict(lambda:defaultdict(float)) for name in columns}
    count_pairs=defaultdict(lambda:{"pairs":0,"weight_sum":0.,"weighted_wins":defaultdict(float)})
    for rows in dev.partition_by("batch_id",maintain_order=True):
        channel=rows["eval_channel"][0]
        y,w=[torch.tensor(rows.select([f"{prefix}_{i}" for i in range(3)]).to_numpy()) for prefix in ("proxy","confidence")]
        for name in columns:
            _,stats=evidence_pair_loss(torch.tensor(rows[name].to_numpy()),y,w)
            for key,value in stats[channel].items(): metrics[name][CHANNELS[channel]][key]+=value
        if channel==0:
            for label,result in pair_diagnostics(rows,{name:rows[name].to_numpy() for name in columns}).items():
                count_pairs[label]["pairs"]+=result["pairs"]; count_pairs[label]["weight_sum"]+=result["weight_sum"]
                for pred,value in result["weighted_wins"].items(): count_pairs[label]["weighted_wins"][pred]+=value
    metrics={name:summarize_stats(total) for name,total in metrics.items()}
    for channel in CHANNELS:
        for key in ("pairs","weighted_win_rate","confidence_scaled_loss"):
            if not np.isclose(metrics["full"]["channels"][channel][key],expected["channels"][channel][key],atol=1e-6,rtol=1e-5):
                raise ValueError("Original selected development metric did not reproduce")
    for row in count_pairs.values():
        row["weighted_accuracy"]={name:w/row["weight_sum"] if row["weight_sum"] else None for name,w in row["weighted_wins"].items()}
    groups={"random_validation4096":validation.filter(pl.col("random_probe")),
        "submitted_top50":validation.filter(pl.col("submitted_model_rank")<=50),
        "validation_probe_combined":validation,"development_unique":dev.unique(subset="molecule_id",maintain_order=True)}
    summary={label:control_summary(frame,columns) for label,frame in groups.items()}
    alpha=float(model.base.pair_alpha().item())
    contributions={label:{"full_std":float(frame["full"].std()),
        "weighted_protein_std":float(alpha*frame["protein_component"].std()),
        "weighted_ligand_std":float((1-alpha)*frame["ligand_component"].std()),
        "protein_ligand_correlation":spearman(frame["protein_component"],frame["ligand_component"]),
        "full_token_length_spearman":spearman(frame["full"],frame["ligand_tokens_capped"])} for label,frame in groups.items()}
    result={"submission_id":9780912,"user_reported_hits":0,"checkpoint_sha256":preparation["checkpoint_sha256"],
        "submitted_score_max_absolute_error":max_error,"same_batch_production_max_absolute_error":same_batch_max_error,
        "submitted_score_absolute_tolerance":2e-5,"submitted_top50_unchanged":True,
        "development_reproduced":True,"pair_alpha":alpha,
        "controls":summary,"weighted_component_variation":contributions,"development_metrics":metrics,
        "competition_pair_strata":dict(count_pairs),"truncated_visits":truncated,
        "elapsed_seconds":time.monotonic()-started,"retrained":False,"challenge_submitted":False,
        "limitations":["Fixed-target training cannot alone demonstrate cross-target biological specificity.",
            "Zero context and shuffled edge features are out-of-distribution sensitivity controls, not alternate biological targets.",
            "No-graph-contacts retains cross-attention; no-cross-attention retains graph contacts.",
            "Ligand-only scores use unmasked inputs, like the submitted scorer; they are not affinity estimates.",
            "Control top-50 overlaps are within the stated probe, not a reranking of the entire validation library.",
            "One fixed perturbation seed; no new submissions or per-compound activity labels inferred."]}
    (a.audit_dir/"report.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2),flush=True)


if __name__=="__main__": main()
