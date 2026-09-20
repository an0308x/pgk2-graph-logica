#!/usr/bin/env python3
"""Train pose-free, pocket-restricted Graph LogiCA on PGK2 ranking pairs.

Ligands use RDKit atom/bond graphs by default, with ETKDG as an optional control.
Every ligand atom is connected to all 26 direct inhibitor-pocket
residues in each of the compound-21 and compound-47 protein conformations.
Those cross edges are learned candidate interactions and never claim a known
protein--ligand distance.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from logica_binding.data import read_fasta
from logica_binding.graph_model import (
    GraphLogiCAModel,
    build_pocket_graph,
    score_graph_prepared,
    smiles_to_rdkit_3d_graph,
    smiles_to_ligand_graph,
)
from logica_binding.model import LogiCABindingScorer
from logica_binding.pocket import load_pocket_region
from logica_binding.validation_protocol import PROTOCOL, validate_pair_splits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pairs",
        type=Path,
        default=ROOT / "artifacts" / "pgk2_corrected_pairs_v2" / "pretrain.parquet",
    )
    parser.add_argument(
        "--protein-fasta",
        type=Path,
        default=ROOT / "data" / "proteins" / "PGK2_P07205.fasta",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "models" / "logica-8m" / "checkpoints" / "8m" / "best.pt",
    )
    parser.add_argument("--hf-cache", type=Path, default=ROOT / "models" / "hf-cache")
    parser.add_argument(
        "--initial-graph-adapter",
        type=Path,
        default=None,
        help="Optional Graph LogiCA adapter from a broader first-stage pair fit.",
    )
    parser.add_argument(
        "--pocket-json",
        type=Path,
        default=ROOT / "artifacts" / "pgk2_inhibitor_pocket" / "inhibitor_pocket.json",
    )
    parser.add_argument(
        "--cmp21-pdb",
        type=Path,
        default=ROOT / "data" / "structures" / "cache7" / "PGK2_cmp21.pdb",
    )
    parser.add_argument(
        "--cmp47-pdb",
        type=Path,
        default=ROOT / "data" / "structures" / "cache7" / "PGK2_cmp47.pdb",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "artifacts" / "pgk2_graph_logica_corrected",
    )
    parser.add_argument("--smoke-pairs-per-split", type=int, default=0)
    parser.add_argument("--ligand-mode", choices=("2d", "etkdg"), default="2d")
    parser.add_argument("--graph-conditioning", choices=("on", "off"), default="on")
    parser.add_argument("--run-purpose", choices=("diagnostic", "production"), default="diagnostic")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--encoder-batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--graph-width", type=int, default=128)
    parser.add_argument("--ligand-radius", type=float, default=4.5)
    parser.add_argument(
        "--graph-workers",
        type=int,
        default=1,
        help="Threads used for optional deterministic RDKit conformer generation.",
    )
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def murcko_group(smiles: str) -> str:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return smiles
    scaffold = MurckoScaffold.GetScaffoldForMol(molecule)
    return Chem.MolToSmiles(scaffold, canonical=True) if scaffold.GetNumAtoms() else smiles


def group_holdout_indices(
    groups: np.ndarray, test_fraction: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    unique = np.unique(groups)
    if unique.size < 2:
        raise ValueError("At least two scaffold groups are required")
    generator = np.random.default_rng(seed)
    generator.shuffle(unique)
    count = min(max(1, int(round(unique.size * test_fraction))), unique.size - 1)
    heldout_groups = set(unique[:count])
    heldout = np.flatnonzero(np.isin(groups, list(heldout_groups)))
    train = np.flatnonzero(~np.isin(groups, list(heldout_groups)))
    return train, heldout


def weighted_pair_loss(
    scores: torch.Tensor, weights: torch.Tensor, temperature: float
) -> torch.Tensor:
    midpoint = scores.shape[0] // 2
    losses = torch.nn.functional.softplus(
        -(scores[:midpoint] - scores[midpoint:]) / temperature
    )
    return (losses * weights).sum() / weights.sum().clamp_min(torch.finfo(losses.dtype).eps)


def build_ligand_graph_task(task: tuple[str, int, float]):
    smiles, seed, radius = task
    return smiles_to_rdkit_3d_graph(smiles, seed=seed, spatial_cutoff=radius)


def main() -> int:
    args = parse_args()
    for path in (
        args.pairs,
        args.protein_fasta,
        args.checkpoint,
        args.pocket_json,
        args.cmp21_pdb,
        args.cmp47_pdb,
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.initial_graph_adapter is not None and not args.initial_graph_adapter.exists():
        raise FileNotFoundError(args.initial_graph_adapter)
    if (
        args.smoke_pairs_per_split < 0
        or args.epochs <= 0
        or args.temperature <= 0
        or args.graph_workers <= 0
    ):
        raise ValueError("Invalid smoke limit, epochs, temperature, or graph-workers")
    if (args.output_dir / "report.json").exists() or (args.output_dir / "graph_logica_adapter.pt").exists():
        raise FileExistsError(f"Refusing to overwrite a completed fit: {args.output_dir}")
    started = time.perf_counter()
    torch.manual_seed(args.seed)
    generator = np.random.default_rng(args.seed)
    pairs = pl.read_parquet(args.pairs)
    required = {"positive_smiles", "negative_smiles", "pair_weight", "split", "protocol", "split_manifest_sha256", "count_reliability"}
    if not required.issubset(pairs.columns):
        raise ValueError(f"Missing pair columns: {sorted(required - set(pairs.columns))}")
    if pairs["protocol"].unique().to_list() != [PROTOCOL] or pairs["split_manifest_sha256"].n_unique() != 1:
        raise ValueError("A corrected shared split manifest is required")
    manifest_id = pairs["split_manifest_sha256"][0]
    validate_pair_splits(pairs.to_dicts())
    if args.smoke_pairs_per_split:
        pairs = pl.concat([pairs.filter(pl.col("split") == split).head(args.smoke_pairs_per_split)
                           for split in ("train", "dev", "holdout")])
    rows = pairs.to_dicts()
    smiles = list(
        dict.fromkeys(
            [str(row["positive_smiles"]) for row in rows]
            + [str(row["negative_smiles"]) for row in rows]
        )
    )
    smiles_index = {value: index for index, value in enumerate(smiles)}
    positive = np.asarray([smiles_index[str(row["positive_smiles"])] for row in rows])
    negative = np.asarray([smiles_index[str(row["negative_smiles"])] for row in rows])
    weights = np.asarray([float(row["pair_weight"]) for row in rows], dtype=np.float32)
    splits = np.asarray([row["split"] for row in rows])
    train_indices = np.flatnonzero(splits == "train")
    dev_indices = np.flatnonzero(splits == "dev")
    heldout_indices = np.flatnonzero(splits == "holdout")
    if any(len(ix) == 0 for ix in (train_indices, dev_indices, heldout_indices)):
        raise ValueError("All three independent splits must contain pairs")

    device = (
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu"
        if args.device == "auto"
        else args.device
    )
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested but CUDA is unavailable")
    pocket_positions = load_pocket_region(args.pocket_json, "inhibitor_pocket")
    scorer = LogiCABindingScorer(
        checkpoint=args.checkpoint,
        hf_cache=args.hf_cache,
        alpha=args.alpha,
        seed=args.seed,
        protein_mask_positions=pocket_positions,
        protein_mask_region="inhibitor_pocket",
        score_mode="token_likelihood",
        device=device,
    )
    prepared, diagnostics = scorer.prepare(
        read_fasta(args.protein_fasta), smiles, encoder_batch_size=args.encoder_batch_size
    )
    graph_tasks = [(value, args.seed, args.ligand_radius) for value in smiles]
    if args.ligand_mode == "2d":
        ligand_graphs = [smiles_to_ligand_graph(value) for value in smiles]
    elif args.graph_workers == 1:
        ligand_graphs = [build_ligand_graph_task(task) for task in graph_tasks]
    else:
        with ThreadPoolExecutor(max_workers=args.graph_workers) as executor:
            ligand_graphs = list(executor.map(build_ligand_graph_task, graph_tasks))
    pocket_graphs = [
        build_pocket_graph(args.cmp21_pdb, args.pocket_json),
        build_pocket_graph(args.cmp47_pdb, args.pocket_json),
    ]
    graph_model = GraphLogiCAModel(scorer.model, graph_width=args.graph_width,
                                  graph_conditioning=args.graph_conditioning == "on").to(device)
    if args.initial_graph_adapter is not None:
        initial = torch.load(args.initial_graph_adapter, map_location="cpu", weights_only=False)
        if initial.get("protocol") != PROTOCOL or initial.get("split_manifest_sha256") != manifest_id:
            raise ValueError("Initial adapter must use the same corrected cross-stage split")
        for key in ("graph_conditioning", "ligand_mode"):
            if initial["training_args"].get(key) != getattr(args, key):
                raise ValueError(f"Initial adapter differs in {key}")
        if initial["training_args"].get("smoke_pairs_per_split", 0):
            raise ValueError("A smoke checkpoint cannot initialize a production fit")
        if initial.get("target") != "PGK2" or initial.get("uniprot") != "P07205":
            raise ValueError("Initial graph adapter does not identify canonical human PGK2")
        if int(initial["training_args"]["graph_width"]) != args.graph_width:
            raise ValueError("Initial graph adapter uses a different graph width")
        base_incompatible = scorer.model.load_state_dict(
            initial["base_state_dict"], strict=False
        )
        graph_incompatible = graph_model.conditioner.load_state_dict(
            initial["conditioner_state_dict"], strict=True
        )
        if base_incompatible.unexpected_keys or graph_incompatible.unexpected_keys:
            raise ValueError("Unexpected state in initial Graph LogiCA adapter")
    base_trainable = scorer.configure_finetuning()
    for parameter in graph_model.conditioner.parameters():
        parameter.requires_grad_(args.graph_conditioning == "on")
    trainable = [p for p in graph_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=args.learning_rate, weight_decay=args.weight_decay
    )
    history: list[dict[str, float | int]] = []
    best_dev = -float("inf")
    best_state = None
    selected_epoch = None
    trainable_names = {name for name, parameter in graph_model.named_parameters() if parameter.requires_grad}

    @torch.no_grad()
    def evaluate_pairs(pair_indices):
        graph_model.eval()
        correct, total, weighted_correct, total_weight = 0.0, 0, 0.0, 0.0
        for offset in range(0, len(pair_indices), args.batch_size):
            ix = pair_indices[offset:offset + args.batch_size]
            scores = score_graph_prepared(graph_model, prepared,
                torch.as_tensor(np.concatenate([positive[ix], negative[ix]])), pocket_graphs, ligand_graphs).cpu().numpy()
            wins = (scores[:len(ix)] > scores[len(ix):]).astype(float) + 0.5 * (scores[:len(ix)] == scores[len(ix):])
            correct += float(wins.sum()); total += len(ix)
            weighted_correct += float((wins * weights[ix]).sum()); total_weight += float(weights[ix].sum())
        return {"pairs": total, "win_rate": correct / total, "weighted_win_rate": weighted_correct / total_weight}

    for epoch in range(1, args.epochs + 1):
        graph_model.train()
        shuffled = generator.permutation(train_indices)
        losses: list[float] = []
        for offset in range(0, len(shuffled), args.batch_size):
            batch_pairs = shuffled[offset : offset + args.batch_size]
            compound_indices = torch.as_tensor(
                np.concatenate([positive[batch_pairs], negative[batch_pairs]]),
                dtype=torch.long,
            )
            batch_weights = torch.as_tensor(
                weights[batch_pairs], dtype=torch.float32, device=device
            )
            scores = score_graph_prepared(
                graph_model,
                prepared,
                compound_indices,
                pocket_graphs,
                ligand_graphs,
            )
            loss = weighted_pair_loss(scores, batch_weights, args.temperature)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        dev = evaluate_pairs(dev_indices)
        if dev["weighted_win_rate"] > best_dev:
            best_dev = dev["weighted_win_rate"]
            best_state = {name: tensor.detach().cpu().clone() for name, tensor in graph_model.state_dict().items()
                          if name in trainable_names}
            selected_epoch = epoch
        item = {"epoch": epoch, "weighted_pairwise_loss": float(np.mean(losses)), "dev": dev}
        history.append(item)
        print(json.dumps(item))

    assert best_state is not None
    graph_model.load_state_dict(best_state, strict=False)
    heldout_metrics = evaluate_pairs(heldout_indices)
    graph_model.eval()
    base_prefixes = (
        "prot_gate",
        "drug_gate",
        "prot_proj_",
        "drug_proj_",
        "cross_attn.",
        "pair_alpha_logit",
    )
    base_state = {
        name: tensor.detach().cpu()
        for name, tensor in scorer.model.state_dict().items()
        if name.startswith(base_prefixes)
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "protocol": PROTOCOL,
            "candidate_scoring_allowed": args.run_purpose == "production" and not args.smoke_pairs_per_split,
            "split_manifest_sha256": manifest_id,
            "selected_epoch": selected_epoch,
            "base_state_dict": base_state,
            "conditioner_state_dict": {
                name: tensor.detach().cpu()
                for name, tensor in graph_model.conditioner.state_dict().items()
            },
            "target": "PGK2",
            "uniprot": "P07205",
            "graph_mode": f"{args.ligand_mode}_strict_direct_pocket",
            "initial_graph_adapter": (
                str(args.initial_graph_adapter) if args.initial_graph_adapter else None
            ),
            "training_args": vars(args),
            "history": history,
        },
        args.output_dir / "graph_logica_adapter.pt",
    )
    report = {
        "protocol": PROTOCOL,
        "run_purpose": args.run_purpose,
        "split_manifest_sha256": manifest_id,
        "selected_epoch": selected_epoch,
        "graph_conditioning": args.graph_conditioning,
        "ligand_mode": args.ligand_mode,
        "smoke_only": bool(args.smoke_pairs_per_split),
        "target": "PGK2",
        "graph_mode": f"{args.ligand_mode}_strict_direct_pocket",
        "initial_graph_adapter": (
            str(args.initial_graph_adapter) if args.initial_graph_adapter else None
        ),
        "pairs": {
            "used": len(rows),
            "train": len(train_indices),
            "dev": len(dev_indices),
            "heldout": len(heldout_indices),
            "unique_molecules": len(smiles),
        },
        "graphs": {
            "protein_templates": [args.cmp21_pdb.stem, args.cmp47_pdb.stem],
            "protein_context_nodes": int(pocket_graphs[0].node_features.shape[0]),
            "direct_cross_edge_residues": int(pocket_graphs[0].direct_mask.sum()),
            "ligand_geometry": args.ligand_mode,
            "base_attention_interface_restricted": True,
            "intermolecular_distance_used": False,
            "protein_language_context": "ESM-2 over the complete 417-residue PGK2 sequence",
        },
        "heldout_scaffold_pair_win_rate": heldout_metrics["win_rate"],
        "holdout": heldout_metrics,
        "training": {
            "device": device,
            "history": history,
            "trainable_parameters": int(sum(parameter.numel() for parameter in trainable)),
            "learned_pair_alpha": float(scorer.model.pair_alpha().detach().cpu()),
            "protein_graph_gate": float(
                torch.sigmoid(graph_model.conditioner.protein_graph_gate).detach().cpu()
            ),
            "drug_graph_gate": float(
                torch.sigmoid(graph_model.conditioner.drug_graph_gate).detach().cpu()
            ),
        },
        "diagnostics": diagnostics,
        "limitations": [
            "Count reliability is a prespecified heuristic, not a calibrated binding probability; sequencing depths remain unknown.",
            "No normalized z-score aggregation is used by this corrected pair objective.",
            "Cross edges are candidate interactions, not known physical contacts.",
            "Training preferences are DEL-derived and not kinase-assay labels.",
            "Graph capacity can overfit small pseudo-label sets; use broader pretraining, active-site refinement, and scaffold-held-out diagnostics.",
        ],
        "elapsed_seconds": time.perf_counter() - started,
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
