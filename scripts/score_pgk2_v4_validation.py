#!/usr/bin/env python3
"""User-authorized validation-only inference for the selected v4 graph checkpoint."""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import polars as pl
from rdkit import Chem
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"src"),str(ROOT/"scripts")]
from logica_binding.data import read_fasta
from logica_binding.development_protocol import PROTOCOL
from logica_binding.graph_model import GraphLogiCAModel,build_pocket_graph,smiles_to_ligand_graph
from logica_binding.full_graph_model import PackedLigandBatch
from logica_binding.model import LogiCABindingScorer
from logica_binding.pocket import load_pocket_region
from logica_binding.packed_graph_data import load_archive,build_packed_batch
from logica_binding.streaming_graph_logica import CHANNELS,packed_logica_scores,evidence_pair_loss
from run_pgk2_full_logica import encode_ligands,verify_manifest,summarize_stats
from prepare_pgk2_full_logica_manifest import sha256


def pack_smiles(smiles):
    graphs=[smiles_to_ligand_graph(s) for s in smiles]
    offsets=np.cumsum([0]+[len(g.node_features) for g in graphs])
    return PackedLigandBatch(torch.cat([g.node_features for g in graphs]),
        torch.cat([g.edge_index+int(offsets[i]) for i,g in enumerate(graphs)],1),
        torch.cat([g.edge_features for g in graphs]),torch.tensor(offsets))


def validate_checkpoint(saved,report):
    if saved["protocol"]!=PROTOCOL or report["protocol"]!=PROTOCOL or saved["identity"]!=report["identity"]:
        raise ValueError("Checkpoint/report identity mismatch")
    c=report["identity"]["config"]
    if (c["model"],c["zero_cap"],c["auxiliary_weight"],c["seed"])!=("graph",5,0.,2026):
        raise ValueError("Not the user-selected leading model")
    if report["smoke_only"] or saved["selected_step"]!=report["selected_step"]:
        raise ValueError("Smoke or wrong selected checkpoint")
    selected=[h for h in report["history"] if h["selected"]][-1]
    if saved["counts"]["optimizer_steps"]!=selected["counts"]["optimizer_steps"]:
        raise ValueError("Best checkpoint step mismatch")
    return selected["dev"]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input",type=Path,required=True)
    p.add_argument("--run-dir",type=Path,required=True)
    p.add_argument("--output-dir",type=Path,required=True)
    p.add_argument("--manifest",type=Path,required=True)
    p.add_argument("--authorize-validation",action="store_true")
    p.add_argument("--verify-only",action="store_true")
    p.add_argument("--verification",type=Path)
    p.add_argument("--shard-id",type=int,default=0)
    p.add_argument("--num-shards",type=int,default=8)
    p.add_argument("--batch-size",type=int,default=64)
    p.add_argument("--device",choices=("cpu","cuda"),default="cuda")
    a=p.parse_args()
    if not a.authorize_validation:
        raise ValueError("Explicit user validation authorization required; old adapter flags remain unchanged")
    if a.batch_size<1 or a.num_shards<1 or not 0<=a.shard_id<a.num_shards:
        raise ValueError("Invalid inference dimensions")
    torch.set_num_threads(4); torch.manual_seed(2026)
    started=time.monotonic()
    manifest=verify_manifest(a.manifest)
    inp=json.loads(a.input.read_text())
    if inp["panel"]!="validation" or inp["rows"]!=244328 or len(inp["candidates"])!=244328:
        raise ValueError("Only the complete validation panel is authorized")
    if inp["zip_sha256"]!=manifest["candidate_panels_sha256"]:
        raise ValueError("Candidate panel differs from training exclusion source")
    saved=torch.load(a.run_dir/"best.pt",map_location="cpu",weights_only=False)
    report=json.loads((a.run_dir/"report.json").read_text())
    reference=validate_checkpoint(saved,report)
    if saved["identity"]["manifest_sha256"]!=manifest["manifest_sha256"]:
        raise ValueError("Manifest mismatch")
    checkpoint=ROOT/"models/logica-8m/checkpoints/8m/best.pt"
    if sha256(checkpoint)!=saved["identity"]["initial_checkpoint_sha256"]:
        raise ValueError("Base checkpoint changed")
    for name in ("src/logica_binding/graph_model.py","src/logica_binding/model.py","src/logica_binding/streaming_graph_logica.py"):
        if sha256(ROOT/name)!=saved["identity"]["source_sha256"][name]:
            raise ValueError(f"Model source changed: {name}")
    identity={"checkpoint_sha256":sha256(a.run_dir/"best.pt"),"input_sha256":sha256(a.input),
        "training_manifest_sha256":manifest["manifest_sha256"],"selected_step":report["selected_step"],
        "inference_source_sha256":sha256(Path(__file__)),"panel":"validation",
        "authorization":"User explicitly requested validation submission on 2026-09-19; test not authorized."}
    if not a.verify_only:
        if a.verification is None:
            raise ValueError("Successful verification report required")
        verification=json.loads(a.verification.read_text())
        if verification["identity"]!=identity or not verification["verified"]:
            raise ValueError("Inference verification identity mismatch")
    definition=ROOT/"artifacts/pgk2_inhibitor_pocket/inhibitor_pocket.json"
    scorer=LogiCABindingScorer(checkpoint,hf_cache=ROOT/"models/hf-cache",device=a.device,
        score_mode="token_likelihood",protein_mask_positions=load_pocket_region(definition,"inhibitor_pocket"),
        protein_mask_region="inhibitor_pocket",seed=2026)
    pt=scorer.protein_tokenizer([read_fasta(ROOT/"data/proteins/PGK2_P07205.fasta")],return_tensors="pt")
    pi=pt["input_ids"].to(a.device)
    with torch.no_grad():
        ph=scorer.model.esm.base_model(input_ids=pi,attention_mask=pt["attention_mask"].to(a.device),return_dict=True).last_hidden_state
    pockets=[build_pocket_graph(ROOT/f"data/structures/cache7/PGK2_cmp{i}.pdb",definition).to(a.device) for i in (21,47)]
    scorer.configure_finetuning(); model=GraphLogiCAModel(scorer.model).to(a.device)
    if set(saved["state"])!={n for n,v in model.named_parameters() if v.requires_grad}:
        raise ValueError("Adapter parameter keys differ")
    model.load_state_dict(saved["state"],strict=False); model.eval()
    a.output_dir.mkdir(parents=True,exist_ok=True)

    @torch.no_grad()
    def score(smiles,batch):
        dh,da,di,ds,trunc=encode_ligands(scorer,smiles,False,2026)
        s=packed_logica_scores(model,ph,dh,da,di,ds,pi,pockets,batch.to(a.device))
        if not torch.isfinite(s).all():
            raise ValueError("Nonfinite candidate scores")
        return s,trunc

    if a.verify_only:
        total=defaultdict(lambda:defaultdict(float)); parity_checked=False
        for entry in manifest["shards"]:
            mp=a.manifest.parent/entry["metadata"]
            if sha256(mp)!=entry["metadata_sha256"] or sha256(entry["graphs"])!=entry["graphs_sha256"]:
                raise ValueError("Verification inputs changed")
            f=pl.read_parquet(mp); archive=load_archive(Path(entry["graphs"]),True,False)
            for channel,name in enumerate(CHANNELS):
                group=np.flatnonzero((f["split"].to_numpy()=="dev") & (f[f"confidence_{channel}"].to_numpy()>0))
                for offset in range(0,len(group),32):
                    pos=group[offset:offset+32]
                    if len(pos)<2: continue
                    rows=f[pos]; smiles=rows["canonical_smiles"].to_list()
                    batch=build_packed_batch(archive,pos,True,False)
                    s,_=score(smiles,batch)
                    if not parity_checked:
                        fresh=pack_smiles(smiles)
                        for key in ("node_features","edge_features","edge_index","node_offsets"):
                            torch.testing.assert_close(getattr(fresh,key),getattr(batch,key))
                        sf,_=score(smiles,fresh); torch.testing.assert_close(sf,s,atol=2e-6,rtol=2e-6)
                        parity_checked=True
                    y,w=[torch.tensor(rows.select([f"{prefix}_{i}" for i in range(3)]).to_numpy(),device=a.device) for prefix in ("proxy","confidence")]
                    _,stats=evidence_pair_loss(s,y,w)
                    for k,v in stats[channel].items(): total[name][k]+=v
        dev=summarize_stats(total)
        for name in CHANNELS:
            for key in ("pairs","weight_sum","weighted_win_rate","confidence_scaled_loss"):
                if not np.isclose(dev["channels"][name][key],reference["channels"][name][key],atol=1e-6,rtol=1e-5):
                    raise ValueError(f"Selected development score did not reproduce: {name}/{key}")
        # Bounded validation runtime check, not a shortlist or changed ranking.
        sample=[Chem.MolToSmiles(Chem.MolFromSmiles(r["SMILES"]),canonical=True,isomericSmiles=True) for r in inp["candidates"][:128]]
        at=time.monotonic(); ntrunc=0
        for offset in range(0,len(sample),a.batch_size):
            smi=sample[offset:offset+a.batch_size]; _,trunc=score(smi,pack_smiles(smi)); ntrunc+=trunc
        result={"identity":identity,"verified":True,"fresh_graph_parity":parity_checked,"dev":dev,
            "validation_smoke_rows":128,"validation_smoke_truncated":ntrunc,
            "validation_smoke_seconds":time.monotonic()-at,"elapsed_seconds":time.monotonic()-started}
        with (a.output_dir/"verification.json").open("x") as f: json.dump(result,f,indent=2)
        print(json.dumps(result),flush=True); return

    start=244328*a.shard_id//a.num_shards; stop=244328*(a.shard_id+1)//a.num_shards
    output=a.output_dir/f"shard-{a.shard_id:03d}.parquet"
    if output.exists(): raise FileExistsError(output)
    rows=[]; trunc_total=0
    for offset in range(start,stop,a.batch_size):
        selected=inp["candidates"][offset:min(offset+a.batch_size,stop)]
        smiles=[]
        for row in selected:
            mol=Chem.MolFromSmiles(row["SMILES"])
            if mol is None or mol.GetNumAtoms()==0: raise ValueError(f"Invalid molecule {row['CatalogID']}")
            smiles.append(Chem.MolToSmiles(mol,canonical=True,isomericSmiles=True))
        values,trunc=score(smiles,pack_smiles(smiles)); trunc_total+=trunc
        rows.extend({"row_index":offset+i,"CatalogID":r["CatalogID"],"canonical_smiles":smiles[i],"score":float(values[i])} for i,r in enumerate(selected))
        if len(rows)%4096==0:
            print(json.dumps({"shard":a.shard_id,"scored":len(rows),"total":stop-start,"seconds":time.monotonic()-started}),flush=True)
    tmp=output.with_suffix(".tmp"); pl.DataFrame(rows).write_parquet(tmp); tmp.replace(output)
    result={"identity":identity,"completed":True,"range":[start,stop],"num_shards":a.num_shards,
        "rows":len(rows),"score_sha256":sha256(output),"truncated_visits":trunc_total,"elapsed_seconds":time.monotonic()-started}
    output.with_suffix(".json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result),flush=True)


if __name__=="__main__": main()
