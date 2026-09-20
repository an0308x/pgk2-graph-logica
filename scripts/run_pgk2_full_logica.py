#!/usr/bin/env python3
"""Stream full-library Graph LogiCA with observed-control ranking and ligand auxiliary learning.

This is a PGK2-specific adaptation of LogiCA, not the paper's original training
recipe. All train molecules receive ligand-only reconstruction supervision;
only observed DEL controls support the separate protein-conditioned ranking.
"""
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
from logica_binding.data import read_fasta
from logica_binding.graph_model import GraphLogiCAModel, build_pocket_graph
from logica_binding.model import LogiCABindingScorer, _make_masked_inputs
from logica_binding.pocket import load_pocket_region
from logica_binding.streaming_graph_logica import (
    FULL_PROTOCOL, CHANNELS, CHANNEL_WEIGHTS, evidence_pair_loss,
    ligand_reconstruction_loss, packed_logica_scores,
)
from prepare_pgk2_full_logica_manifest import sha256
from logica_binding.packed_graph_data import load_archive, build_packed_batch


def args_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, default=ROOT / "models/logica-8m/checkpoints/8m/best.pt")
    p.add_argument("--hf-cache", type=Path, default=ROOT / "models/hf-cache")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--rank-batch-size", type=int, default=32)
    p.add_argument("--rank-every", type=int, default=4)
    p.add_argument("--auxiliary-weight", type=float, default=.01)
    p.add_argument("--learning-rate", type=float, default=3e-5)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--max-train-batches", type=int, default=0, help="Nonzero marks run smoke-only")
    p.add_argument("--max-eval-batches", type=int, default=0, help="Nonzero marks run smoke-only")
    p.add_argument("--resume", action="store_true")
    return p


def verify_manifest(path):
    manifest = json.loads(path.read_text())
    expected = manifest.pop("manifest_sha256")
    actual = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    manifest["manifest_sha256"] = expected
    if actual != expected or manifest["protocol"] != FULL_PROTOCOL or not manifest["completed"]:
        raise ValueError("Invalid manifest")
    if manifest["graph_mode"] != "2d" or manifest["zscore_columns_used"]:
        raise ValueError("Incorrect full-data protocol")
    if sum(row["molecules"] for row in manifest["splits"].values()) != manifest["molecules"]:
        raise ValueError("Split coverage does not match the library")
    expected_start = 0
    for entry in manifest["shards"]:
        start, stop = entry["range"]
        if start != expected_start or stop <= start:
            raise ValueError("Noncontiguous shard coverage")
        expected_start = stop
    if expected_start != manifest["molecules"]:
        raise ValueError("Incomplete shard coverage")
    return manifest


def encode_ligands(scorer, smiles, masked, seed):
    selfies = [scorer.smiles_to_selfies(s) for s in smiles]
    full = scorer.drug_tokenizer(selfies, truncation=False, padding=False)
    truncated = sum(len(ids) > 128 for ids in full["input_ids"])
    tokens = scorer.drug_tokenizer(selfies, truncation=True, max_length=128,
        padding=True, return_special_tokens_mask=True, return_tensors="pt")
    ids = tokens["input_ids"]
    attention = tokens["attention_mask"]
    if masked:
        inputs, selected = _make_masked_inputs(ids, attention, tokens["special_tokens_mask"],
            int(scorer.drug_tokenizer.mask_token_id), smiles, .15, seed)
    else:
        inputs = ids
        selected = attention.bool() & ~tokens["special_tokens_mask"].bool()
    with torch.no_grad():
        scorer.model.drug_encoder.eval()
        hidden = scorer.model.drug_encoder.base_model(input_ids=inputs.to(scorer.device),
            attention_mask=attention.to(scorer.device), return_dict=True).last_hidden_state
    return hidden, attention.to(scorer.device), ids.to(scorer.device), selected.to(scorer.device), truncated


def add_stats(total, stats):
    for name, row in zip(CHANNELS, stats):
        for key, value in row.items():
            total[name][key] += value


def summarize_stats(total):
    result = {}
    for name in CHANNELS:
        row = dict(total[name])
        n, w = row.get("pairs", 0), row.get("weight_sum", 0)
        row["weighted_win_rate"] = row.get("weighted_wins", 0) / w if w else None
        row["confidence_scaled_loss"] = row.get("loss_sum", 0) / n if n else None
        result[name] = row
    losses = [result[n]["confidence_scaled_loss"] for n in CHANNELS]
    # Selection requires genuine observed competition comparisons, not only NTC.
    metric = sum(w*x for w, x in zip(CHANNEL_WEIGHTS, losses) if x is not None) / sum(
        w for w, x in zip(CHANNEL_WEIGHTS, losses) if x is not None) if losses[0] is not None else None
    return {"channels": result, "selection_loss": metric}


def main():
    args = args_parser().parse_args()
    if min(args.epochs, args.batch_size, args.rank_batch_size, args.rank_every, args.threads) < 1:
        raise ValueError("Invalid training dimensions")
    if args.auxiliary_weight <= 0 or min(args.max_train_batches, args.max_eval_batches) < 0:
        raise ValueError("Invalid objective weight or smoke limit")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    if args.output_dir.exists() and not args.resume:
        raise FileExistsError(args.output_dir)
    if (args.output_dir / "report.json").exists():
        raise FileExistsError("Already complete")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    manifest = verify_manifest(args.manifest)
    smoke = bool(args.max_train_batches or args.max_eval_batches)
    if manifest.get("synthetic_fixture") and not smoke:
        raise ValueError("Synthetic fixtures are restricted to smoke runs")
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items() if k != "resume"}
    identity = {"manifest": manifest["manifest_sha256"], "config": config,
                "initial_checkpoint_sha256": sha256(args.checkpoint)}
    definition = ROOT / "artifacts/pgk2_inhibitor_pocket/inhibitor_pocket.json"
    scorer = LogiCABindingScorer(args.checkpoint, hf_cache=args.hf_cache, device=args.device,
        score_mode="token_likelihood", protein_mask_positions=load_pocket_region(definition, "inhibitor_pocket"),
        protein_mask_region="inhibitor_pocket", seed=args.seed)
    sequence = read_fasta(ROOT / "data/proteins/PGK2_P07205.fasta")
    pt = scorer.protein_tokenizer([sequence], return_tensors="pt")
    with torch.no_grad():
        ph = scorer.model.esm.base_model(input_ids=pt["input_ids"].to(args.device),
            attention_mask=pt["attention_mask"].to(args.device), return_dict=True).last_hidden_state
    protein_ids = pt["input_ids"].to(args.device)
    pockets = [build_pocket_graph(ROOT / f"data/structures/cache7/PGK2_cmp{i}.pdb", definition).to(args.device)
               for i in (21, 47)]
    scorer.configure_finetuning()
    model = GraphLogiCAModel(scorer.model).to(args.device)
    trainable = {name: p for name, p in model.named_parameters() if p.requires_grad}
    optimizer = torch.optim.AdamW(trainable.values(), lr=args.learning_rate, weight_decay=.01)
    cursor = [0, 0, 0]
    history, best_dev, selected_epoch = [], float("inf"), None
    counts = {"train_molecule_visits": 0, "rank_molecule_visits": 0, "truncated_aux_visits": 0,
              "truncated_rank_visits": 0, "optimizer_steps": 0, "training_seconds": 0.0}
    saved_seconds = 0.0
    train_stats = defaultdict(lambda: defaultdict(float))
    if args.resume:
        saved = torch.load(args.output_dir / "last.pt", map_location="cpu", weights_only=False)
        if saved["identity"] != identity:
            raise ValueError("Resume identity differs")
        if set(saved["state"]) != set(trainable):
            raise ValueError("Trainable parameters differ")
        model.load_state_dict(saved["state"], strict=False)
        optimizer.load_state_dict(saved["optimizer"])
        cursor, counts, history = saved["cursor"], saved["counts"], saved["history"]
        best_dev, selected_epoch = saved["best_dev"], saved["selected_epoch"]
        saved_seconds = saved["elapsed_seconds"]
        for name, row in saved["train_stats"].items():
            train_stats[name].update(row)
        torch.set_rng_state(saved["rng"])
        if args.device == "cuda":
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
    started = time.monotonic()
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    def state():
        return {name: p.detach().cpu().clone() for name, p in trainable.items()}

    def save():
        payload = {"protocol": FULL_PROTOCOL, "candidate_scoring_allowed": False, "identity": identity,
            "state": state(), "optimizer": optimizer.state_dict(), "cursor": cursor,
            "counts": counts, "history": history, "best_dev": best_dev, "selected_epoch": selected_epoch,
            "train_stats": {k: dict(v) for k, v in train_stats.items()},
            "rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all() if args.device == "cuda" else [],
            "elapsed_seconds": saved_seconds + time.monotonic() - started}
        tmp = args.output_dir / "last.pt.tmp"
        torch.save(payload, tmp); tmp.replace(args.output_dir / "last.pt")

    def load_shard(entry, need_graph=True):
        metadata = args.manifest.parent / entry["metadata"]
        if sha256(metadata) != entry["metadata_sha256"]:
            raise ValueError("Metadata changed")
        frame = pl.read_parquet(metadata)
        if need_graph:
            if sha256(entry["graphs"]) != entry["graphs_sha256"]:
                raise ValueError("Graphs changed")
            archive = load_archive(Path(entry["graphs"]), True, False)
            if len(archive["node_offsets"]) != frame.height + 1:
                raise ValueError("Graph/metadata mismatch")
        else:
            archive = None
        return frame, archive

    def rank(frame, archive, positions):
        rows = frame[positions]
        smiles = rows["canonical_smiles"].to_list()
        dh, da, di, ds, trunc = encode_ligands(scorer, smiles, False, args.seed)
        batch = build_packed_batch(archive, positions, True, False).to(args.device)
        scores = packed_logica_scores(model, ph, dh, da, di, ds, protein_ids, pockets, batch)
        proxy = torch.as_tensor(rows.select([f"proxy_{i}" for i in range(3)]).to_numpy().copy(), device=args.device)
        confidence = torch.as_tensor(rows.select([f"confidence_{i}" for i in range(3)]).to_numpy().copy(), device=args.device)
        loss, stats = evidence_pair_loss(scores, proxy, confidence)
        return loss, stats, trunc

    @torch.no_grad()
    def evaluate(split):
        model.eval()
        total = defaultdict(lambda: defaultdict(float))
        batches, visits = 0, 0
        for entry in manifest["shards"]:
            frame, _ = load_shard(entry, False)
            base = frame["split"].to_numpy() == split
            groups = [np.flatnonzero(base & (frame[f"confidence_{i}"].to_numpy() > 0)) for i in range(3)]
            if not any(len(g) > 1 for g in groups):
                continue
            _, archive = load_shard(entry)
            for channel, group in enumerate(groups):
                # Same deterministic batches for all epochs and models.
                for offset in range(0, len(group), args.rank_batch_size):
                    pos = group[offset:offset+args.rank_batch_size]
                    if len(pos) < 2:
                        continue
                    _, stats, _ = rank(frame, archive, pos)
                    # Avoid counting a channel again when its rows enter another pool.
                    add_stats(total, [r if i == channel else {"pairs": 0, "loss_sum": 0,
                        "weight_sum": 0, "weighted_wins": 0} for i, r in enumerate(stats)])
                    batches += 1; visits += len(pos)
                    if args.max_eval_batches and batches >= args.max_eval_batches:
                        return {**summarize_stats(total), "batches": batches, "molecule_visits": visits}
        return {**summarize_stats(total), "batches": batches, "molecule_visits": visits}

    stop_smoke = bool(args.max_train_batches and counts["optimizer_steps"] >= args.max_train_batches)
    for epoch in range(cursor[0], args.epochs):
        order = np.random.default_rng(args.seed + epoch).permutation(len(manifest["shards"]))
        for shard_position in range(cursor[1] if epoch == cursor[0] else 0, len(order)):
            if stop_smoke:
                break
            entry = manifest["shards"][int(order[shard_position])]
            frame, archive = load_shard(entry)
            rng = np.random.default_rng(args.seed + epoch * 100000 + shard_position)
            train = frame["split"].to_numpy() == "train"
            indices = rng.permutation(np.flatnonzero(train))
            pools = [np.flatnonzero(train & (frame[f"confidence_{i}"].to_numpy() > 0)) for i in range(3)]
            first_batch = cursor[2] if epoch == cursor[0] and shard_position == cursor[1] else 0
            for batch_index, offset in enumerate(range(0, len(indices), args.batch_size)):
                if batch_index < first_batch:
                    continue
                batch_started = time.monotonic()
                model.train(); model.base.esm.eval(); model.base.drug_encoder.eval()
                pos = indices[offset:offset+args.batch_size]
                dh, _, di, selected, trunc = encode_ligands(scorer, frame[pos]["canonical_smiles"].to_list(), True, args.seed+epoch)
                batch = build_packed_batch(archive, pos, True, False).to(args.device)
                auxiliary = ligand_reconstruction_loss(model, dh, di, selected, batch)
                loss = args.auxiliary_weight * auxiliary
                rank_value = 0.0
                if batch_index % args.rank_every == 0:
                    available = [g for g in pools if len(g) > 1]
                    if available:
                        pool = available[(batch_index // args.rank_every) % len(available)]
                        # Independent of traversal history so restart is reproducible.
                        draw = np.random.default_rng(args.seed + epoch*10000000 + shard_position*10000 + batch_index)
                        ranked = draw.choice(pool, min(args.rank_batch_size, len(pool)), replace=False)
                        rloss, stats, rt = rank(frame, archive, ranked)
                        loss = loss + args.rank_every * rloss
                        rank_value = float(rloss.detach())
                        counts["rank_molecule_visits"] += len(ranked)
                        counts["truncated_rank_visits"] += rt
                        add_stats(train_stats, stats)
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(list(trainable.values()), 1.0, error_if_nonfinite=True)
                optimizer.step()
                if args.device == "cuda":
                    torch.cuda.synchronize()
                counts["training_seconds"] += time.monotonic() - batch_started
                counts["optimizer_steps"] += 1; counts["train_molecule_visits"] += len(pos)
                counts["truncated_aux_visits"] += trunc
                cursor = [epoch, shard_position, batch_index+1]
                if counts["optimizer_steps"] % 100 == 0 or counts["optimizer_steps"] == 1:
                    print(json.dumps({"epoch": epoch+1, "shard_position": shard_position,
                        **counts, "auxiliary_loss": float(auxiliary.detach()), "ranking_loss": rank_value,
                        "gradient_norm": float(norm), "elapsed_seconds": saved_seconds+time.monotonic()-started}), flush=True)
                if counts["optimizer_steps"] % 1000 == 0:
                    save()
                if args.max_train_batches and counts["optimizer_steps"] >= args.max_train_batches:
                    stop_smoke = True; save(); break
            if stop_smoke:
                break
            cursor = [epoch, shard_position+1, 0]; save()
        dev = evaluate("dev")
        metric = dev["selection_loss"]
        if metric is not None and metric < best_dev:
            best_dev, selected_epoch = metric, epoch+1
            torch.save({"protocol": FULL_PROTOCOL, "candidate_scoring_allowed": False,
                        "identity": identity, "state": state(), "selected_epoch": selected_epoch},
                       args.output_dir / "best.pt")
        history.append({"epoch": epoch+1, "dev": dev, "counts": dict(counts)})
        print(json.dumps(history[-1]), flush=True)
        if stop_smoke:
            break
        cursor = [epoch+1, 0, 0]; save()
    if selected_epoch is None and not smoke:
        raise ValueError("No development competition evidence; cannot select a checkpoint")
    if selected_epoch is not None:
        best = torch.load(args.output_dir / "best.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(best["state"], strict=False)
    holdout = evaluate("holdout")
    expected = manifest["splits"]["train"]["molecules"] * args.epochs
    if not smoke and counts["train_molecule_visits"] != expected:
        raise ValueError("Incomplete full-library coverage")
    report = {"protocol": FULL_PROTOCOL, "identity": identity, "smoke_only": smoke,
        "candidate_scoring_allowed": False, "selected_epoch": selected_epoch,
        "counts": counts, "expected_full_train_visits": expected,
        "train_molecules_per_second": counts["train_molecule_visits"] / max(counts["training_seconds"], 1e-6),
        "full_training_coverage": not smoke and counts["train_molecule_visits"] == expected,
        "library_splits": manifest["splits"], "history": history, "holdout": holdout,
        "training_ranking": summarize_stats(train_stats),
        "elapsed_seconds": saved_seconds+time.monotonic()-started,
        "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated() if args.device == "cuda" else None,
        "limitations": ["Custom Graph LogiCA adaptation, not the original paper training recipe.",
            "All train molecules receive ligand-only self-supervision, not kinase labels.",
            "Zero controls are uncertain; only observed controls support ranking.",
            "Count contrasts and reliability weights are heuristics with unknown sequencing depths.",
            "Direct-pocket edges are candidate interactions, not experimental ligand poses.",
            "SELFormer has a 128-token cap; truncation visits are reported, full ligand graphs retained.",
            "DEL validation is not kinase validation; review before any challenge scoring."]}
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
