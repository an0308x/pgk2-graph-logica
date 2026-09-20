#!/usr/bin/env python3
"""Measure whether LogiCA's gated residual behaves as a residual.

Paper Eq. 5 is H_c = H + sigma(g) * phi(Z_N - Z_0), and the paper states that
"the near-zero gate initialization keeps H_c close to H at the start of
training". That premise requires sigma(g) * ||phi(Z_N - Z_0)|| to stay small
relative to ||H||. If it does not, the native token head receives a vector
dominated by adapter output rather than a nudged encoder representation, which
breaks the premise that makes the context-conditioned log-likelihood in Eq. 2
meaningful.

This reports, per branch, the RMS-per-element norms of the hidden state, the
raw tower update, the projected update, and the gated update relative to the
hidden state, for the released checkpoint and for each supplied adapter.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from logica_binding.data import read_fasta
from logica_binding.model import LogiCABindingScorer

PROBE_SMILES = "CNC(=O)C[C@H](c1cccnc1)n1c(-c2ccccc2)nc2ccccc21"


def rms(tensor: torch.Tensor) -> float:
    """Root-mean-square per element, so norms are comparable across shapes."""
    return float(tensor.norm() / tensor.numel() ** 0.5)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "models" / "logica-8m" / "checkpoints" / "8m" / "best.pt",
    )
    parser.add_argument(
        "--adapters",
        nargs="*",
        type=Path,
        default=[ROOT / "artifacts" / "pgk2_competition_finetune" / "finetuned_adapter.pt"],
    )
    parser.add_argument(
        "--protein-fasta", type=Path, default=ROOT / "data" / "proteins" / "PGK2_P07205.fasta"
    )
    parser.add_argument("--hf-cache", type=Path, default=ROOT / "models" / "hf-cache")
    parser.add_argument("--smiles", default=PROBE_SMILES)
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "artifacts" / "pgk2_gate_diagnostic"
    )
    return parser.parse_args()


@torch.no_grad()
def measure(args: argparse.Namespace, adapter: Path | None) -> dict[str, object]:
    scorer = LogiCABindingScorer(
        checkpoint=args.checkpoint,
        hf_cache=args.hf_cache,
        protein_mask_key="PGK2_P07205",
        device="cpu",
    )
    if adapter is not None:
        payload = torch.load(adapter, map_location="cpu", weights_only=False)
        incompatible = scorer.model.load_state_dict(payload["model_state_dict"], strict=False)
        if incompatible.unexpected_keys:
            raise ValueError(f"Unexpected adapter weights: {incompatible.unexpected_keys}")
    sequence = read_fasta(args.protein_fasta)
    prepared, _ = scorer.prepare(sequence, [args.smiles], encoder_batch_size=1)
    model = scorer.model

    protein_hidden = prepared.protein_hidden
    drug_hidden = prepared.drug_hidden
    z0 = model.prot_proj_in(protein_hidden)
    d0 = model.drug_proj_in(drug_hidden)
    zn, dn = model.cross_attn(
        z0, d0, prepared.protein_attention_mask == 0, prepared.drug_attention_mask == 0
    )
    branches = {}
    for name, hidden, update, project, gate in (
        ("protein", protein_hidden, zn - z0, model.prot_proj_out, model.prot_gate),
        ("drug", drug_hidden, dn - d0, model.drug_proj_out, model.drug_gate),
    ):
        projected = project(update)
        gate_value = float(torch.sigmoid(gate))
        branches[name] = {
            "gate_sigmoid": gate_value,
            "rms_hidden": rms(hidden),
            "rms_tower_update": rms(update),
            "rms_projected_update": rms(projected),
            "tower_amplification_vs_hidden": rms(update) / rms(hidden),
            "projected_update_over_hidden": rms(projected) / rms(hidden),
            # The quantity Eq. 5 needs to be small for H_c to stay close to H.
            "gated_update_over_hidden": gate_value * rms(projected) / rms(hidden),
            "proj_out_weight_fro": float(project.weight.norm()),
        }
    return {"adapter": str(adapter) if adapter else None, "branches": branches}


def main() -> int:
    args = parse_args()
    results = [measure(args, None)]
    results.extend(measure(args, adapter) for adapter in args.adapters)

    print(f"{'checkpoint':34s} {'branch':8s} {'|H|':>8s} {'|dZ|':>10s} "
          f"{'|phi|/|H|':>10s} {'gated/|H|':>10s}")
    print("-" * 86)
    for entry in results:
        label = Path(entry["adapter"]).parent.name if entry["adapter"] else "RELEASED 8m (no adapter)"
        for branch, values in entry["branches"].items():
            print(
                f"{label:34s} {branch:8s} {values['rms_hidden']:8.4f} "
                f"{values['rms_tower_update']:10.2f} "
                f"{values['projected_update_over_hidden']:10.1f} "
                f"{values['gated_update_over_hidden']:10.3f}"
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "probe_smiles": args.smiles,
        "paper_premise": "Eq. 5 assumes sigma(g)*phi(Z_N - Z_0) is small relative to H",
        "results": results,
        "limitations": [
            "Single protein and single probe ligand; norms are RMS per element.",
            "CrossAttentionBlock internals are reconstructed, so an official normalization step absent here would change these numbers.",
        ],
    }
    (args.output_dir / "adapter_norms.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
