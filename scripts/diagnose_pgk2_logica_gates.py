#!/usr/bin/env python3
"""Diagnose whether LogiCA's cross-modal gates are a live conditioning pathway.

Every checkpoint in this project reports sigmoid(gate) ~= 0.0033 against an
initialization of sigmoid(-6.0) = 0.00247, including the released BindingDB
checkpoint after 100 epochs. All cross-modal information reaches the token
heads through that multiplier, so this asks three separate questions:

1. Magnitude: how large is the gated conditioning perturbation relative to the
   unconditional hidden state and logits?
2. Trainability: is the raw gate parameter receiving usable gradient, or is it
   initialized into the saturated tail of the sigmoid where its derivative
   vanishes?
3. Utility: as the gate is forced open, does DEL competition-pair
   discrimination improve, degrade, or stay flat?

Question 3 is the decisive one. Improvement means the closed gate is an
optimization failure worth fixing before any graph branch is injected through
the same pathway. Degradation means the model has learned that this
cross-modal formulation actively hurts, which is a different and more serious
problem. Flatness means the score is near-independent of the conditioning.

Pair win rates here are DEL-derived training diagnostics, never ASMS or
kinase-assay performance.
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from logica_binding.data import read_fasta
from logica_binding.model import LogiCABindingScorer
from logica_binding.pocket import REGION_CHOICES, load_pocket_region


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pairs",
        type=Path,
        default=ROOT / "artifacts" / "pgk2_competition_pairs" / "pgk2_competition_pairs.parquet",
    )
    parser.add_argument(
        "--protein-fasta", type=Path, default=ROOT / "data" / "proteins" / "PGK2_P07205.fasta"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "models" / "logica-8m" / "checkpoints" / "8m" / "best.pt",
    )
    parser.add_argument(
        "--adapter",
        type=Path,
        default=ROOT / "artifacts" / "pgk2_competition_finetune" / "finetuned_adapter.pt",
    )
    parser.add_argument("--hf-cache", type=Path, default=ROOT / "models" / "hf-cache")
    parser.add_argument(
        "--pocket-json",
        type=Path,
        default=ROOT / "artifacts" / "pgk2_inhibitor_pocket" / "inhibitor_pocket.json",
    )
    parser.add_argument(
        "--regions", nargs="+", choices=REGION_CHOICES, default=["random", "inhibitor_pocket"]
    )
    parser.add_argument(
        "--gate-logits",
        nargs="+",
        type=float,
        default=[-6.0, -4.0, -2.0, -1.0, 0.0, 1.0, 2.0],
        help="Raw gate values to force. The trained value is near -5.7.",
    )
    parser.add_argument("--max-pairs", type=int, default=279)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--encoder-batch-size", type=int, default=32)
    parser.add_argument("--mask-fraction", type=float, default=0.15)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "artifacts" / "pgk2_gate_diagnostic"
    )
    return parser.parse_args()


def load_scorer(args: argparse.Namespace, region: str, device: str) -> LogiCABindingScorer:
    positions = load_pocket_region(args.pocket_json, region)
    scorer = LogiCABindingScorer(
        checkpoint=args.checkpoint,
        hf_cache=args.hf_cache,
        mask_fraction=args.mask_fraction,
        alpha=args.alpha,
        seed=args.seed,
        protein_mask_key="PGK2_P07205",
        protein_mask_positions=positions or None,
        protein_mask_region=region,
        device=device,
    )
    if args.adapter is not None:
        payload = torch.load(args.adapter, map_location="cpu", weights_only=False)
        incompatible = scorer.model.load_state_dict(payload["model_state_dict"], strict=False)
        if incompatible.unexpected_keys:
            raise ValueError(f"Unexpected adapter weights: {incompatible.unexpected_keys}")
    return scorer


@torch.no_grad()
def perturbation_norms(scorer: LogiCABindingScorer, prepared, index: int) -> dict[str, float]:
    """Relative size of the gated conditioning update on hidden states and logits."""
    model = scorer.model
    device = scorer.device
    protein_hidden = prepared.protein_hidden.to(device)
    protein_attention = prepared.protein_attention_mask.to(device)
    drug_hidden = prepared.drug_hidden[index : index + 1].to(device)
    drug_attention = prepared.drug_attention_mask[index : index + 1].to(device)

    protein_interaction_0 = model.prot_proj_in(protein_hidden)
    drug_interaction_0 = model.drug_proj_in(drug_hidden)
    protein_interaction, _ = model.cross_attn(
        protein_interaction_0,
        drug_interaction_0,
        protein_attention == 0,
        drug_attention == 0,
    )
    delta = model.prot_proj_out(protein_interaction - protein_interaction_0)
    gate = torch.sigmoid(model.prot_gate)
    gated = gate * delta

    unconditional_logits = model.esm.lm_head(protein_hidden)
    conditional_logits = model.esm.lm_head(protein_hidden + gated)

    def rel(numerator: torch.Tensor, denominator: torch.Tensor) -> float:
        return float(numerator.norm() / denominator.norm().clamp_min(1e-12))

    return {
        "gate_value": float(gate),
        "ungated_delta_rel_hidden": rel(delta, protein_hidden),
        "gated_delta_rel_hidden": rel(gated, protein_hidden),
        "logit_shift_rel": rel(conditional_logits - unconditional_logits, unconditional_logits),
    }


def gate_gradient_report(
    scorer: LogiCABindingScorer, prepared, positive_index: np.ndarray, negative_index: np.ndarray
) -> dict[str, float]:
    """Compare the raw gate gradient with an adapter weight gradient.

    A gate initialized at -6.0 sits in the saturated tail of the sigmoid, where
    d sigmoid/d g = s(1-s) ~= 0.0025. Gradient reaching the raw parameter is
    scaled by that factor, so a vanishing gate gradient alongside healthy
    adapter gradients means the gate is effectively frozen by construction
    rather than converged.
    """
    scorer.configure_finetuning()
    scorer.model.zero_grad(set_to_none=True)
    take = min(64, len(positive_index))
    indices = torch.tensor(
        np.concatenate([positive_index[:take], negative_index[:take]]), dtype=torch.long
    )
    scores = scorer.score_tensor(prepared, indices)
    positive_scores, negative_scores = scores[:take], scores[take:]
    loss = -torch.nn.functional.logsigmoid(positive_scores - negative_scores).mean()
    loss.backward()

    gate = scorer.model.prot_gate
    sigmoid_value = float(torch.sigmoid(gate))
    weight = scorer.model.prot_proj_out.weight
    return {
        "loss": float(loss),
        "prot_gate_raw": float(gate.detach()),
        "prot_gate_sigmoid": sigmoid_value,
        "sigmoid_derivative": sigmoid_value * (1.0 - sigmoid_value),
        # Signed, not norm: the sign says whether descent would open or close the gate.
        "prot_gate_grad_signed": float(gate.grad.reshape(-1)[0]) if gate.grad is not None else float("nan"),
        "drug_gate_grad_signed": (
            float(scorer.model.drug_gate.grad.reshape(-1)[0])
            if scorer.model.drug_gate.grad is not None
            else float("nan")
        ),
        "prot_proj_out_grad_norm": float(weight.grad.norm()) if weight.grad is not None else float("nan"),
        # Raw-gate units needed to reach sigmoid(gate)=0.5, for context on the sign.
        "raw_units_to_half_open": -float(gate.detach().reshape(-1)[0]),
    }


def main() -> int:
    args = parse_args()
    started = time.perf_counter()
    pairs = pl.read_parquet(args.pairs).head(args.max_pairs)
    positives = pairs["positive_smiles"].to_list()
    negatives = pairs["negative_smiles"].to_list()

    unique: dict[str, int] = {}
    for smiles in positives + negatives:
        unique.setdefault(smiles, len(unique))
    compounds = list(unique)
    positive_index = np.array([unique[s] for s in positives])
    negative_index = np.array([unique[s] for s in negatives])

    protein = read_fasta(args.protein_fasta)
    device = (
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu"
        if args.device == "auto"
        else args.device
    )

    records: list[dict[str, object]] = []
    gradients: dict[str, dict[str, float]] = {}
    baseline_norms: dict[str, dict[str, float]] = {}

    for region in args.regions:
        scorer = load_scorer(args, region, device)
        prepared, diagnostics = scorer.prepare(
            protein, compounds, encoder_batch_size=args.encoder_batch_size
        )
        trained_gate = float(scorer.model.prot_gate.detach())
        baseline_norms[region] = perturbation_norms(scorer, prepared, 0)
        gradients[region] = gate_gradient_report(
            scorer, prepared, positive_index, negative_index
        )

        for gate_logit in args.gate_logits:
            with torch.no_grad():
                scorer.model.prot_gate.fill_(gate_logit)
                scorer.model.drug_gate.fill_(gate_logit)
            scores = scorer.score_prepared(prepared, batch_size=args.batch_size)
            positive_scores = scores[positive_index]
            negative_scores = scores[negative_index]
            wins = positive_scores > negative_scores
            norms = perturbation_norms(scorer, prepared, 0)
            records.append(
                {
                    "region": region,
                    "gate_logit": gate_logit,
                    "gate_sigmoid": norms["gate_value"],
                    "pair_win_rate": float(np.mean(wins)),
                    "mean_score": float(np.mean(scores)),
                    "score_std": float(np.std(scores)),
                    "mean_positive_minus_negative": float(
                        np.mean(positive_scores - negative_scores)
                    ),
                    "gated_delta_rel_hidden": norms["gated_delta_rel_hidden"],
                    "logit_shift_rel": norms["logit_shift_rel"],
                }
            )
            print(
                f"{region:22s} gate_logit={gate_logit:+.1f} "
                f"sigmoid={norms['gate_value']:.6f} "
                f"win_rate={float(np.mean(wins)):.4f} "
                f"delta/hidden={norms['gated_delta_rel_hidden']:.4f} "
                f"logit_shift={norms['logit_shift_rel']:.4f}",
                flush=True,
            )
        print(f"  trained prot_gate raw = {trained_gate:.4f}", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    table = pl.DataFrame(records)
    table.write_csv(args.output_dir / "gate_sweep.csv")
    report = {
        "target": "PGK2",
        "uniprot": "P07205",
        "adapter": str(args.adapter),
        "pairs_used": len(pairs),
        "unique_compounds": len(compounds),
        "device": device,
        "alpha": args.alpha,
        "regions": args.regions,
        "baseline_perturbation_norms": baseline_norms,
        "gate_gradients": gradients,
        "elapsed_seconds": time.perf_counter() - started,
        "limitations": [
            "Pair win rates are DEL-derived training diagnostics, not ASMS or kinase-assay performance.",
            "Forcing a gate value is an inference-time intervention, not a retrained model.",
            "Gate values are shared by the protein and drug branches in this sweep.",
        ],
    }
    (args.output_dir / "gate_diagnostic.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in report.items() if k != "baseline_perturbation_norms"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
