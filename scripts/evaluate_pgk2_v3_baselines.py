#!/usr/bin/env python3
"""Matched development-only baselines; no challenge scoring or holdout evaluation."""
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
from logica_binding.data import read_fasta
from logica_binding.graph_model import GraphLogiCAModel, build_pocket_graph
from logica_binding.model import LogiCABindingScorer
from logica_binding.pocket import load_pocket_region
from logica_binding.packed_graph_data import load_archive, build_packed_batch
from logica_binding.streaming_graph_logica import CHANNELS, FULL_PROTOCOL, evidence_pair_loss, packed_logica_scores
from run_pgk2_full_logica import verify_manifest, encode_ligands, summarize_stats
from prepare_pgk2_full_logica_manifest import sha256


def ranking_schedule(frame, shard_position, seed=2026):
    """Replay exactly the v3 full epoch's supervised batch selection (no aux fit)."""
    train = frame["split"].to_numpy() == "train"
    available = [np.flatnonzero(train & (frame[f"confidence_{i}"].to_numpy() > 0)) for i in range(3)]
    available = [g for g in available if len(g) > 1]
    if not available:
        return
    for batch_index in range(0, (int(train.sum())+63)//64, 4):
        pool = available[(batch_index//4) % len(available)]
        rng = np.random.default_rng(seed + shard_position*10000 + batch_index)
        yield rng.choice(pool, min(32, len(pool)), replace=False)


def tensors(frame, device):
    return [torch.tensor(frame.select([f"{prefix}_{i}" for i in range(3)]).to_numpy(),
                         dtype=torch.float32, device=device) for prefix in ("proxy", "confidence")]


def pair_details(scores, rows, channel):
    proxy = rows[f"proxy_{channel}"].to_numpy()
    confidence = rows[f"confidence_{channel}"].to_numpy()
    a, b = np.triu_indices(len(rows), 1)
    valid = (np.abs(proxy[a]-proxy[b]) >= np.log(2)) & (confidence[a] > 0) & (confidence[b] > 0)
    a, b = a[valid], b[valid]
    weight = np.minimum(confidence[a], confidence[b])
    return a, b, weight


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--trained-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = p.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    args.output_dir.mkdir(parents=True)
    torch.set_num_threads(4); torch.manual_seed(2026)
    start = time.monotonic()
    manifest = verify_manifest(args.manifest)
    completed = json.loads((args.trained_dir / "report.json").read_text())
    if not completed["full_training_coverage"] or completed["identity"]["manifest"] != manifest["manifest_sha256"]:
        raise ValueError("Checkpoint data identity mismatch")
    config = completed["identity"]["config"]
    for name, expected in {"epochs":1,"seed":2026,"batch_size":64,"rank_batch_size":32,"rank_every":4}.items():
        if config[name] != expected:
            raise ValueError(f"Unsupported matched recipe: {name}")
    pretrained = Path(config["checkpoint"])
    if sha256(pretrained) != completed["identity"]["initial_checkpoint_sha256"]:
        raise ValueError("Starting checkpoint changed")
    definition = ROOT / "artifacts/pgk2_inhibitor_pocket/inhibitor_pocket.json"
    scorer = LogiCABindingScorer(pretrained, hf_cache=Path(config["hf_cache"]), device=args.device,
        score_mode="token_likelihood", protein_mask_positions=load_pocket_region(definition,"inhibitor_pocket"),
        protein_mask_region="inhibitor_pocket", seed=2026)
    sequence = read_fasta(ROOT / "data/proteins/PGK2_P07205.fasta")
    pt = scorer.protein_tokenizer([sequence], return_tensors="pt")
    with torch.no_grad():
        ph = scorer.model.esm.base_model(input_ids=pt["input_ids"].to(args.device),
            attention_mask=pt["attention_mask"].to(args.device), return_dict=True).last_hidden_state
    protein_ids = pt["input_ids"].to(args.device)
    pockets = [build_pocket_graph(ROOT/f"data/structures/cache7/PGK2_cmp{i}.pdb",definition).to(args.device) for i in (21,47)]
    scorer.configure_finetuning()
    model = GraphLogiCAModel(scorer.model).to(args.device).eval()
    initial = {name:v.detach().clone() for name,v in model.named_parameters() if v.requires_grad}
    trained = torch.load(args.trained_dir / "best.pt", map_location=args.device, weights_only=False)
    if trained["protocol"] != FULL_PROTOCOL or trained["identity"] != completed["identity"] or set(trained["state"]) != set(initial):
        raise ValueError("Adapter identity or trainable state differs")
    fp = torch.nn.Linear(2048, 1, bias=False).to(args.device)
    torch.nn.init.zeros_(fp.weight)
    optimizer = torch.optim.AdamW(fp.parameters(), lr=.001, weight_decay=.01)
    visits, steps = 0, 0

    def load(entry, graph=False, fingerprint=False):
        path = args.manifest.parent / entry["metadata"]
        if sha256(path) != entry["metadata_sha256"]:
            raise ValueError("Metadata changed")
        if sha256(entry["graphs"]) != entry["graphs_sha256"]:
            raise ValueError("Graph archive changed")
        return pl.read_parquet(path), load_archive(Path(entry["graphs"]), graph, fingerprint)

    order = np.random.default_rng(2026).permutation(len(manifest["shards"]))
    for shard_position, shard_id in enumerate(order):
        frame, archive = load(manifest["shards"][int(shard_id)], fingerprint=True)
        for positions in ranking_schedule(frame, shard_position):
            features = build_packed_batch(archive, positions, False, True).fingerprints.to(args.device)
            loss, _ = evidence_pair_loss(fp(features).flatten(), *tensors(frame[positions], args.device))
            optimizer.zero_grad(set_to_none=True); (4*loss).backward()
            torch.nn.utils.clip_grad_norm_(fp.parameters(), 1., error_if_nonfinite=True)
            optimizer.step(); visits += len(positions); steps += 1
        if shard_position % 8 == 0:
            print(json.dumps({"phase":"fingerprint_fit","shard_position":shard_position,"rank_visits":visits}),flush=True)
    if visits != completed["counts"]["rank_molecule_visits"]:
        raise ValueError("Supervised exposure did not match Graph LogiCA")
    torch.save({"state":fp.state_dict(), "manifest":manifest["manifest_sha256"],
                "candidate_scoring_allowed":False}, args.output_dir / "morgan_linear.pt")

    names = ("trained_graph", "initial_graph", "initial_masked_logica", "morgan_linear")
    metrics = {name:defaultdict(lambda:defaultdict(float)) for name in names}
    diagnostics = {c:{"weight_sum":0., "weight_square_sum":0., "molecule_weight":defaultdict(float)} for c in CHANNELS}
    pair_hash = hashlib.sha256()
    exported, batches, eval_visits = [], 0, 0
    with torch.no_grad():
        for shard_id, entry in enumerate(manifest["shards"]):
            frame, archive = load(entry, graph=True, fingerprint=True)
            dev = frame["split"].to_numpy() == "dev"
            for channel, cname in enumerate(CHANNELS):
                group = np.flatnonzero(dev & (frame[f"confidence_{channel}"].to_numpy() > 0))
                for offset in range(0,len(group),32):
                    positions = group[offset:offset+32]
                    if len(positions) < 2:
                        continue
                    rows = frame[positions]
                    dh, da, di, ds, _ = encode_ligands(scorer, rows["canonical_smiles"].to_list(), False, 2026)
                    batch = build_packed_batch(archive, positions, True, True).to(args.device)
                    predictions = {}
                    for name, state, graph_on in [("trained_graph",trained["state"],True),
                            ("initial_graph",initial,True),("initial_masked_logica",initial,False)]:
                        model.load_state_dict(state,strict=False); model.graph_conditioning=graph_on
                        predictions[name] = packed_logica_scores(model,ph,dh,da,di,ds,protein_ids,pockets,batch)
                    predictions["morgan_linear"] = fp(batch.fingerprints).flatten()
                    label, confidence = tensors(rows,args.device)
                    for name, scores in predictions.items():
                        _, stats = evidence_pair_loss(scores,label,confidence)
                        for key,value in stats[channel].items():
                            metrics[name][cname][key] += value
                    a,b,w = pair_details(None,rows,channel)
                    ids = rows["molecule_id"].to_numpy()
                    for ia,ib,weight in zip(a,b,w):
                        pair_hash.update(f"{channel}:{ids[ia]}:{ids[ib]}\n".encode())
                        diagnostics[cname]["molecule_weight"][str(ids[ia])] += float(weight)
                        diagnostics[cname]["molecule_weight"][str(ids[ib])] += float(weight)
                    diagnostics[cname]["weight_sum"] += float(w.sum())
                    diagnostics[cname]["weight_square_sum"] += float((w*w).sum())
                    for j,mid in enumerate(ids):
                        exported.append({"molecule_id":int(mid),"channel":cname,"batch_id":batches,
                            "scaffold":rows["scaffold_key"][j], "token_length_capped":int(da[j].sum()),
                            **{name:float(values[j]) for name,values in predictions.items()}})
                    batches += 1; eval_visits += len(positions)
            if shard_id % 16 == 0:
                print(json.dumps({"phase":"matched_dev","shard":shard_id,"batches":batches}),flush=True)
    results = {name:summarize_stats(total) for name,total in metrics.items()}
    expected = completed["history"][-1]["dev"]
    for cname in CHANNELS:
        actual = results["trained_graph"]["channels"][cname]
        reference = expected["channels"][cname]
        if actual["pairs"] != reference["pairs"] or not np.isclose(actual["weighted_win_rate"],reference["weighted_win_rate"],atol=1e-5):
            raise ValueError(f"Trained development results did not reproduce: {cname}")
    concentration = {}
    for cname,row in diagnostics.items():
        weights = sorted(row["molecule_weight"].values(),reverse=True)
        total = row["weight_sum"]
        concentration[cname] = {"pair_weight_ess":total*total/row["weight_square_sum"] if total else None,
            "unique_pair_molecules":len(weights),"top10_molecule_incidence_weight_fraction":sum(weights[:10])/(2*total) if total else None,
            "note":"Weight-only ESS is not an independent sample size; pairs share molecules/scaffolds."}
    pl.DataFrame(exported).write_parquet(args.output_dir / "development_scores.parquet")
    report = {"protocol":FULL_PROTOCOL,"manifest_sha256":manifest["manifest_sha256"],"split":"dev",
        "holdout_evaluated":False,"candidate_scoring_allowed":False,"results":results,
        "dev_batches":batches,"dev_molecule_visits":eval_visits,"pair_identity_sha256":pair_hash.hexdigest(),
        "trained_dev_reproduced":True,"supervised_rank_visits":visits,"fingerprint_steps":steps,
        "weight_concentration":concentration,"elapsed_seconds":time.monotonic()-start,
        "baseline_recipe":{"model":"zero-initialized linear Morgan-2048 ranker","learning_rate":.001,
            "epochs":1,"weight_decay":.01,"supervised_batches":"exact v3 replay",
            "molecular_auxiliary":False,"hyperparameter_search":False},
        "limitations":["Development-only comparison; do not interpret these DEL proxies as kinase hit rates.",
            "Fingerprint baseline has the same ranking exposure but no molecular auxiliary objective.",
            "Initial graph baseline includes the exact seed-2026 randomly initialized graph branch."]}
    (args.output_dir / "report.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2),flush=True)


if __name__ == "__main__":
    main()
