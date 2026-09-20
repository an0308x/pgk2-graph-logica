#!/usr/bin/env python3
"""Fine-tune the LogiCA adapter on weighted PGK2 ranking pairs.

This is deliberately not scored with AUROC/AP: the pair preferences are
derived DEL evidence, while the challenge endpoint is blinded ASMS retrieval.
A scaffold-grouped held-out pair win rate is recorded only as a training
diagnostic.  Select final settings exclusively by challenge validation results.
"""

from __future__ import annotations

import argparse
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
from logica_binding.model import LogiCABindingScorer
from logica_binding.pocket import REGION_CHOICES, load_pocket_region


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, default=ROOT / "artifacts" / "pgk2_pairs" / "pgk2_ranking_pairs.parquet")
    parser.add_argument("--protein-fasta", type=Path, default=ROOT / "data" / "proteins" / "PGK2_P07205.fasta")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "models" / "logica-8m" / "checkpoints" / "8m" / "best.pt")
    parser.add_argument(
        "--initial-adapter",
        type=Path,
        default=None,
        help="Optional PGK2 adapter to refine instead of starting from the base checkpoint alone.",
    )
    parser.add_argument("--hf-cache", type=Path, default=ROOT / "models" / "hf-cache")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts" / "pgk2_pairwise_finetune")
    parser.add_argument("--max-pairs", type=int, default=1_000, help="Bounded default; increase explicitly on GPU compute.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--encoder-batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--mask-fraction", type=float, default=0.15)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument(
        "--score-mode",
        choices=("masked_gain", "token_likelihood"),
        default="masked_gain",
        help=(
            "masked_gain preserves the project's historical reconstruction; "
            "token_likelihood reproduces the official binding score."
        ),
    )
    parser.add_argument(
        "--protein-mask-region",
        choices=REGION_CHOICES,
        default="random",
        help="Optional structure-derived set of PGK2 residues read by the protein score.",
    )
    parser.add_argument(
        "--pocket-json",
        type=Path,
        default=ROOT / "artifacts" / "pgk2_inhibitor_pocket" / "inhibitor_pocket.json",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Use CUDA when available by default; set cpu only for debugging.",
    )
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def murcko_group(smiles: str) -> str:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return smiles
    scaffold = MurckoScaffold.GetScaffoldForMol(molecule)
    return Chem.MolToSmiles(scaffold, canonical=True) if scaffold.GetNumAtoms() else smiles


def weighted_pair_loss(scores: torch.Tensor, weights: torch.Tensor, temperature: float) -> torch.Tensor:
    midpoint = scores.shape[0] // 2
    raw = torch.nn.functional.softplus(-(scores[:midpoint] - scores[midpoint:]) / temperature)
    return (raw * weights).sum() / weights.sum().clamp_min(torch.finfo(raw.dtype).eps)


def group_holdout_indices(groups: np.ndarray, test_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic group holdout without a scikit-learn runtime dependency."""

    unique_groups = np.unique(groups)
    if unique_groups.size < 2:
        raise ValueError("At least two positive-scaffold groups are required for a holdout")
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_groups)
    test_group_count = min(
        max(1, int(round(unique_groups.size * test_fraction))), unique_groups.size - 1
    )
    heldout_groups = set(unique_groups[:test_group_count])
    heldout = np.flatnonzero(np.isin(groups, list(heldout_groups)))
    train = np.flatnonzero(~np.isin(groups, list(heldout_groups)))
    return train, heldout


def main() -> int:
    args = parse_args()
    if not args.pairs.exists() or not args.protein_fasta.exists() or not args.checkpoint.exists():
        raise FileNotFoundError("Pairs, protein FASTA, or LogiCA checkpoint is missing")
    if args.initial_adapter is not None and not args.initial_adapter.exists():
        raise FileNotFoundError(args.initial_adapter)
    if args.protein_mask_region != "random" and not args.pocket_json.exists():
        raise FileNotFoundError(args.pocket_json)
    if not 0 < args.test_fraction < 1 or args.temperature <= 0 or args.max_pairs <= 0:
        raise ValueError("invalid split, temperature, or max-pairs setting")
    started = time.perf_counter()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    pairs = pl.read_parquet(args.pairs)
    required = {"positive_smiles", "negative_smiles", "pair_weight"}
    if not required.issubset(pairs.columns):
        raise ValueError(f"Missing pair columns: {sorted(required - set(pairs.columns))}")
    if pairs.height > args.max_pairs:
        pairs = pairs.sample(n=args.max_pairs, seed=args.seed)
    pair_rows = pairs.to_dicts()
    groups = np.asarray([murcko_group(str(row["positive_smiles"])) for row in pair_rows])
    indices = np.arange(len(pair_rows))
    train_idx, heldout_idx = group_holdout_indices(groups, args.test_fraction, args.seed)
    all_smiles = list(dict.fromkeys([str(row["positive_smiles"]) for row in pair_rows] + [str(row["negative_smiles"]) for row in pair_rows]))
    smiles_to_index = {smiles: index for index, smiles in enumerate(all_smiles)}
    positive_index = np.asarray([smiles_to_index[str(row["positive_smiles"])] for row in pair_rows])
    negative_index = np.asarray([smiles_to_index[str(row["negative_smiles"])] for row in pair_rows])
    weights = np.asarray([float(row["pair_weight"]) for row in pair_rows], dtype=np.float32)

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested but CUDA is unavailable")
    protein_mask_positions = load_pocket_region(
        args.pocket_json, args.protein_mask_region
    )
    scorer = LogiCABindingScorer(
        checkpoint=args.checkpoint,
        hf_cache=args.hf_cache,
        mask_fraction=args.mask_fraction,
        alpha=args.alpha,
        seed=args.seed,
        protein_mask_key="PGK2_P07205",
        protein_mask_positions=protein_mask_positions or None,
        protein_mask_region=args.protein_mask_region,
        score_mode=args.score_mode,
        device=device,
    )
    if args.initial_adapter is not None:
        payload = torch.load(args.initial_adapter, map_location="cpu", weights_only=False)
        if payload.get("target") != "PGK2" or payload.get("uniprot") != "P07205":
            raise ValueError("Initial adapter metadata does not identify canonical human PGK2")
        incompatible = scorer.model.load_state_dict(payload["model_state_dict"], strict=False)
        if incompatible.unexpected_keys:
            raise ValueError(
                f"Unexpected initial-adapter weights: {incompatible.unexpected_keys}"
            )
    prepared, diagnostics = scorer.prepare(read_fasta(args.protein_fasta), all_smiles, encoder_batch_size=args.encoder_batch_size)
    trainable = scorer.configure_finetuning()
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=args.weight_decay)
    history: list[dict[str, float | int]] = []
    for epoch in range(1, args.epochs + 1):
        scorer.model.train()
        shuffled = rng.permutation(train_idx)
        losses: list[float] = []
        for offset in range(0, len(shuffled), args.batch_size):
            batch_pairs = shuffled[offset : offset + args.batch_size]
            if not len(batch_pairs):
                continue
            compound_indices = torch.as_tensor(np.concatenate([positive_index[batch_pairs], negative_index[batch_pairs]]), dtype=torch.long)
            batch_weights = torch.as_tensor(
                weights[batch_pairs], dtype=torch.float32, device=scorer.device
            )
            loss = weighted_pair_loss(scorer.score_tensor(prepared, compound_indices), batch_weights, args.temperature)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        history.append({"epoch": epoch, "weighted_pairwise_loss": float(np.mean(losses))})
        print(json.dumps(history[-1]))
    scorer.model.eval()
    final_scores = scorer.score_prepared(prepared, batch_size=args.batch_size)
    heldout_wins = final_scores[positive_index[heldout_idx]] > final_scores[negative_index[heldout_idx]]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    trainable_prefixes = (
        "prot_gate",
        "drug_gate",
        "prot_proj_",
        "drug_proj_",
        "cross_attn.",
    )
    if args.score_mode == "token_likelihood":
        trainable_prefixes += ("pair_alpha_logit",)
    trainable_state = {
        name: tensor.detach().cpu()
        for name, tensor in scorer.model.state_dict().items()
        if name.startswith(trainable_prefixes)
    }
    torch.save({"model_state_dict": trainable_state, "base_checkpoint": str(args.checkpoint), "initial_adapter": str(args.initial_adapter) if args.initial_adapter else None, "target": "PGK2", "uniprot": "P07205", "training_args": vars(args), "history": history}, args.output_dir / "finetuned_adapter.pt")
    report = {
        "target": "PGK2",
        "pairs": {"used": len(pair_rows), "train": len(train_idx), "heldout": len(heldout_idx), "weight_sum": float(weights.sum())},
        "heldout_scaffold_pair_win_rate": float(np.mean(heldout_wins)),
        "training": {
            "device": device,
            "history": history,
            "trainable_parameters": int(sum(parameter.numel() for parameter in trainable)),
            "score_mode": args.score_mode,
            "protein_mask_region": args.protein_mask_region,
            "learned_pair_alpha": float(scorer.model.pair_alpha().detach().cpu()),
        },
        "initial_adapter": str(args.initial_adapter) if args.initial_adapter else None,
        "diagnostics": diagnostics,
        "limitations": [
            "Pair labels are DEL-derived preferences, not assay-confirmed ASMS binding labels.",
            "The held-out pair win rate is a training diagnostic; choose configurations only using blinded ASMS validation retrieval.",
            "The default max-pairs is a bounded smoke run, not a final production fit.",
            "This runner freezes the two language-model encoders and tunes only the interaction adapter; it reproduces the official score when token_likelihood is selected, but not the official downstream LoRA recipe.",
        ],
        "elapsed_seconds": time.perf_counter() - started,
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
