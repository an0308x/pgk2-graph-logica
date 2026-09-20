#!/usr/bin/env python3
"""Bounded, development-only objective/zero-read ablation over all shard pools."""
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
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from logica_binding.development_protocol import (PROTOCOL, evidence_intervals,
    interval_pair_loss, training_batches, select_checkpoint, gradient_comparison)
from logica_binding.streaming_graph_logica import (CHANNELS, evidence_pair_loss,
    packed_logica_scores, ligand_reconstruction_loss)
from logica_binding.packed_graph_data import load_archive, build_packed_batch
from logica_binding.model import LogiCABindingScorer
from logica_binding.graph_model import GraphLogiCAModel, build_pocket_graph
from logica_binding.pocket import load_pocket_region
from logica_binding.data import read_fasta
from run_pgk2_full_logica import verify_manifest, encode_ligands, summarize_stats
from prepare_pgk2_full_logica_manifest import sha256


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--model", choices=("graph", "fingerprint"), required=True)
    p.add_argument("--zero-cap", type=int, choices=(0,1,5), default=0)
    p.add_argument("--auxiliary-weight", type=float, choices=(0.,.01), default=0.)
    p.add_argument("--passes", type=int, default=2)
    p.add_argument("--eval-every-shards", type=int, default=16)
    p.add_argument("--patience", type=int, default=4)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--device", choices=("cpu","cuda"), default="cuda")
    p.add_argument("--max-batches-per-shard", type=int, default=0, help="Synthetic smoke only; not a scientific run")
    return p


def main():
    args = parser().parse_args()
    if min(args.passes,args.eval_every_shards,args.patience) < 1 or args.max_batches_per_shard < 0:
        raise ValueError("Invalid development budget")
    if args.model == "fingerprint" and args.auxiliary_weight:
        raise ValueError("Fingerprint arm has no auxiliary objective")
    manifest = verify_manifest(args.manifest)
    if manifest.get("synthetic_fixture") and not args.max_batches_per_shard:
        raise ValueError("Synthetic data requires smoke mode")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4); torch.manual_seed(args.seed)
    started = time.monotonic()
    graph = args.model == "graph"
    config = {k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    identity = {"manifest_sha256":manifest["manifest_sha256"],"config":config,
        "source_sha256":{name:sha256(ROOT/name) for name in [
            "scripts/run_pgk2_development.py","src/logica_binding/development_protocol.py",
            "src/logica_binding/streaming_graph_logica.py","src/logica_binding/graph_model.py",
            "src/logica_binding/model.py"]}}
    if graph:
        definition = ROOT / "artifacts/pgk2_inhibitor_pocket/inhibitor_pocket.json"
        checkpoint = ROOT / "models/logica-8m/checkpoints/8m/best.pt"
        identity["initial_checkpoint_sha256"] = sha256(checkpoint)
        scorer = LogiCABindingScorer(checkpoint, hf_cache=ROOT/"models/hf-cache", device=args.device,
            score_mode="token_likelihood", protein_mask_positions=load_pocket_region(definition,"inhibitor_pocket"),
            protein_mask_region="inhibitor_pocket",seed=args.seed)
        pt = scorer.protein_tokenizer([read_fasta(ROOT/"data/proteins/PGK2_P07205.fasta")],return_tensors="pt")
        pi = pt["input_ids"].to(args.device)
        with torch.no_grad():
            ph = scorer.model.esm.base_model(input_ids=pi,
                attention_mask=pt["attention_mask"].to(args.device),return_dict=True).last_hidden_state
        pockets = [build_pocket_graph(ROOT/f"data/structures/cache7/PGK2_cmp{i}.pdb",definition).to(args.device) for i in (21,47)]
        scorer.configure_finetuning()
        model = GraphLogiCAModel(scorer.model).to(args.device)
    else:
        model = torch.nn.Linear(2048,1,bias=False).to(args.device)
        torch.nn.init.zeros_(model.weight)
    parameters = {n:p for n,p in model.named_parameters() if p.requires_grad}
    optimizer = torch.optim.AdamW(parameters.values(),lr=3e-5 if graph else .001,weight_decay=.01)
    verified = set()
    counts = {"optimizer_steps":0,"ranking_visits":0,"auxiliary_visits":0,
        "ranking_truncated_visits":0,"auxiliary_truncated_visits":0,"shards_completed":0}
    history, diagnostics, coverage = [], [], []
    best, stale, selected_step = float("inf"), 0, None

    def load(entry):
        meta = args.manifest.parent / entry["metadata"]
        if entry["metadata"] not in verified:
            if sha256(meta) != entry["metadata_sha256"] or sha256(entry["graphs"]) != entry["graphs_sha256"]:
                raise ValueError("Source data checksum changed")
            verified.add(entry["metadata"])
        frame = pl.read_parquet(meta)
        archive = load_archive(Path(entry["graphs"]),graph,not graph)
        return frame,archive

    def scores(frame,archive,pos,training=False):
        batch = build_packed_batch(archive,pos,graph,not graph).to(args.device)
        if not graph:
            return model(batch.fingerprints).flatten()
        dh,da,di,ds,trunc = encode_ligands(scorer,frame[pos]["canonical_smiles"].to_list(),False,args.seed)
        if training:
            counts["ranking_truncated_visits"] += trunc
        return packed_logica_scores(model,ph,dh,da,di,ds,pi,pockets,batch)

    @torch.no_grad()
    def evaluate():
        # Every arm is judged on EXACTLY the original observed-control dev pairs.
        # Zero-cap training labels never enter this evaluator; holdout is untouched.
        model.eval(); total = defaultdict(lambda:defaultdict(float)); digest = hashlib.sha256()
        batches,visits = 0,0
        for entry in manifest["shards"]:
            frame,archive = load(entry)
            dev = frame["split"].to_numpy() == "dev"
            for channel,name in enumerate(CHANNELS):
                group = np.flatnonzero(dev & (frame[f"confidence_{channel}"].to_numpy()>0))
                for offset in range(0,len(group),32):
                    pos = group[offset:offset+32]
                    if len(pos)<2:
                        continue
                    rows = frame[pos]
                    y,w = [torch.tensor(rows.select([f"{prefix}_{i}" for i in range(3)]).to_numpy(),
                        device=args.device,dtype=torch.float32) for prefix in ("proxy","confidence")]
                    _,stats = evidence_pair_loss(scores(frame,archive,pos),y,w)
                    for key,value in stats[channel].items():
                        total[name][key] += value
                    a,b = np.triu_indices(len(pos),1)
                    yy,ww = y.cpu().numpy(),w.cpu().numpy()
                    valid = (np.abs(yy[a,channel]-yy[b,channel])>=np.log(2)) & (ww[a,channel]>0) & (ww[b,channel]>0)
                    ids = rows["molecule_id"].to_numpy()
                    for ia,ib in zip(a[valid],b[valid]):
                        digest.update(f"{channel}:{ids[ia]}:{ids[ib]}\n".encode())
                    batches += 1; visits += len(pos)
        return {**summarize_stats(total),"batches":batches,"visits":visits,"pair_identity_sha256":digest.hexdigest()}

    def save_checkpoint(name):
        payload = {"protocol":PROTOCOL,"candidate_scoring_allowed":False,"identity":identity,
            "state":{n:p.detach().cpu().clone() for n,p in parameters.items()},"counts":dict(counts),
            "selected_step":selected_step,"best_dev_loss":best}
        tmp = args.output_dir / (name+".tmp")
        torch.save(payload,tmp); tmp.replace(args.output_dir/name)

    def checkpoint(label):
        nonlocal best,stale,selected_step
        rng = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state_all() if args.device=="cuda" else None
        dev = evaluate()
        torch.set_rng_state(rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
        if history and dev["pair_identity_sha256"] != history[0]["dev"]["pair_identity_sha256"]:
            raise ValueError("Development pairs changed")
        reference_path = ROOT/"artifacts/pgk2_v3_baselines/report.json"
        if not history and reference_path.exists() and not manifest.get("synthetic_fixture"):
            reference = json.loads(reference_path.read_text())
            if reference["manifest_sha256"] != manifest["manifest_sha256"] or dev["pair_identity_sha256"] != reference["pair_identity_sha256"]:
                raise ValueError("Development identity does not match the completed baseline audit")
            if graph and args.seed==2026:
                expected = reference["results"]["initial_graph"]["channels"]["competition"]["weighted_win_rate"]
                actual = dev["channels"]["competition"]["weighted_win_rate"]
                if not np.isclose(actual,expected,atol=1e-5):
                    raise ValueError("Starting graph scores did not reproduce")
        best,stale,improved = select_checkpoint(dev["selection_loss"],best,stale)
        if improved:
            selected_step = counts["optimizer_steps"]; save_checkpoint("best.pt")
        history.append({"label":label,"counts":dict(counts),"dev":dev,"selected":improved})
        print(json.dumps(history[-1]),flush=True)
        (args.output_dir/"progress.json").write_text(json.dumps({"identity":identity,"history":history},indent=2)+"\n")

    checkpoint("initial")
    if selected_step is None:
        raise ValueError("No finite development competition loss")
    stop = False
    for epoch in range(args.passes):
        order = np.random.default_rng(args.seed+epoch).permutation(len(manifest["shards"]))
        for shard_position,shard_id in enumerate(order):
            frame,archive = load(manifest["shards"][int(shard_id)])
            lower,upper,confidence = evidence_intervals(frame,args.zero_cap)
            tensor = [torch.tensor(x,device=args.device) for x in (lower,upper,confidence)]
            train = np.flatnonzero(frame["split"].to_numpy()=="train")
            batch_seed = args.seed+epoch*100000+shard_position
            aux_rng = np.random.default_rng(batch_seed)
            aux_order = aux_rng.permutation(train); aux_cursor = 0
            seen = set()
            for batch_index,pos in enumerate(training_batches(frame,confidence,batch_seed)):
                if args.max_batches_per_shard and batch_index>=args.max_batches_per_shard:
                    break
                # Match ranking dropout draws across objective arms despite the
                # auxiliary branch consuming additional random numbers.
                torch.manual_seed(batch_seed*10000+batch_index)
                model.train()
                if graph:
                    model.base.esm.eval(); model.base.drug_encoder.eval()
                # Match the old rank-update multiplier. The experiment changes
                # update scheduling, so it is NOT a replay of the old full epoch.
                ranking,_ = interval_pair_loss(scores(frame,archive,pos,True),*[t[pos] for t in tensor])
                ranking = 4*ranking
                auxiliary = None
                if args.auxiliary_weight:
                    # 256 auxiliary molecules per ranking batch = old 4 x 64
                    # exposure ratio, accumulated into ONE optimizer update.
                    # Draws span every shard; no tiny fixed molecule subset.
                    pieces = []
                    for _ in range(4):
                        if aux_cursor>=len(aux_order):
                            aux_order=aux_rng.permutation(train); aux_cursor=0
                        ap = aux_order[aux_cursor:aux_cursor+64]; aux_cursor += len(ap)
                        dh,_,di,selected,trunc = encode_ligands(scorer,frame[ap]["canonical_smiles"].to_list(),True,batch_seed)
                        packed = build_packed_batch(archive,ap,True,False).to(args.device)
                        pieces.append(ligand_reconstruction_loss(model,dh,di,selected,packed))
                        counts["auxiliary_visits"] += len(ap); counts["auxiliary_truncated_visits"] += trunc
                    auxiliary = args.auxiliary_weight*torch.stack(pieces).sum()
                    if counts["optimizer_steps"] % 200 == 0:
                        diagnostics.append({"step":counts["optimizer_steps"],
                            **gradient_comparison(ranking,auxiliary,list(parameters.values()))})
                loss = ranking if auxiliary is None else ranking+auxiliary
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite loss")
                optimizer.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(list(parameters.values()),1.,error_if_nonfinite=True)
                optimizer.step()
                counts["optimizer_steps"] += 1; counts["ranking_visits"] += len(pos)
                seen.update(map(int,pos))
            coverage.append({"pass":epoch+1,"shard":int(shard_id),"eligible_train":int(
                ((frame["split"].to_numpy()=="train") & (confidence>0).any(axis=1)).sum()),
                "unique_ranked_train":len(seen),"added_zero_competitor_train":int(
                ((frame["split"].to_numpy()=="train") & (confidence[:,0]>0) & (frame["confidence_0"].to_numpy()==0)).sum())})
            counts["shards_completed"] += 1
            if (shard_position+1)%args.eval_every_shards==0 or shard_position+1==len(order):
                checkpoint(f"pass{epoch+1}_shard{shard_position+1}")
                if stale>=args.patience:
                    stop=True; break
        if stop:
            break
    save_checkpoint("last.pt")
    report = {"protocol":PROTOCOL,"identity":identity,"candidate_scoring_allowed":False,
        "holdout_evaluated":False,"smoke_only":bool(args.max_batches_per_shard),"counts":counts,
        "history":history,"selected_step":selected_step,"early_stopped":stop,"coverage":coverage,
        "gradient_diagnostics":diagnostics,"elapsed_seconds":time.monotonic()-started,
        "limitations":["Zero caps 1 and 5 are assumed count ranges, not statistical confidence intervals.",
            "No sequencing-depth calibration or kinase-assay labels; training metadata unchanged.",
            "All eligible training pools visited per completed pass, not full-library auxiliary coverage.",
            "Auxiliary microbatches accumulated at ranking updates; not an exact v3 trajectory replay.",
            "One seed, development selection only; no challenge inference or holdout evaluation.",
            "Fixed observed-control dev pairs cannot validate newly included zero-read compounds.",
            "128-token truncation remains; complete ligand graphs retained."]}
    (args.output_dir/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report),flush=True)


if __name__=="__main__":
    main()
