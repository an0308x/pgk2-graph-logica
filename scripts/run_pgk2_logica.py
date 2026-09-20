#!/usr/bin/env python3
"""Score a bounded PGK2 challenge-panel batch with the pretrained LogiCA model.

The PGK2 validation and test files have no public labels. This script produces
rankable scores only; it intentionally does not calculate AUROC/AP or use the
legacy WDR91 adapter. Pair-specific LogiCA scoring is expensive, so the default
is a small smoke batch. Use a scheduler/GPU workflow for larger bounded runs.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
import zipfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from logica_binding.data import read_fasta
from logica_binding.model import LogiCABindingScorer
from logica_binding.pocket import REGION_CHOICES, load_pocket_region


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", choices=("validation", "test"), default="validation")
    parser.add_argument(
        "--candidate-panels",
        type=Path,
        default=ROOT / "DREAM_challenge_2026" / "Val-Test-set.zip",
    )
    parser.add_argument(
        "--protein-fasta", type=Path, default=ROOT / "data" / "proteins" / "PGK2_P07205.fasta"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "models" / "logica-8m" / "checkpoints" / "8m" / "best.pt",
    )
    parser.add_argument("--hf-cache", type=Path, default=ROOT / "models" / "hf-cache")
    parser.add_argument(
        "--adapter",
        type=Path,
        default=None,
        help="Optional PGK2 pairwise-finetuned adapter produced by run_pgk2_pairwise_finetune.py.",
    )
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument(
        "--max-compounds",
        type=int,
        default=8,
        help="Bounded scoring batch; use an explicit larger value only on adequate compute.",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--encoder-batch-size", type=int, default=8)
    parser.add_argument("--mask-fraction", type=float, default=0.15)
    parser.add_argument(
        "--score-mode",
        choices=("masked_gain", "token_likelihood"),
        default="masked_gain",
        help=(
            "masked_gain reproduces every earlier submission: mask a fraction of "
            "residues and score the conditional-minus-unconditional gain at those "
            "positions. token_likelihood matches the reference implementation's "
            "pair_scores_from_tokens, the score the official code selects for the "
            "binding task: unmasked inputs, own-token log-probability at every "
            "valid position, no unconditional term."
        ),
    )
    parser.add_argument(
        "--protein-mask-region",
        choices=REGION_CHOICES,
        default="random",
        help=(
            "Residues that define the protein-side score. 'random' keeps the seeded "
            "mask-fraction behaviour used by every submission so far; the pocket "
            "regions score only structure-derived compound-21/47 residues and ignore "
            "--mask-fraction."
        ),
    )
    parser.add_argument(
        "--pocket-json",
        type=Path,
        default=ROOT / "artifacts" / "pgk2_inhibitor_pocket" / "inhibitor_pocket.json",
    )
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Use CUDA when available by default; set cpu only for local debugging.",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts" / "pgk2_logica")
    return parser.parse_args()


def read_panel(path: Path, panel: str, offset: int, limit: int) -> list[dict[str, str]]:
    if offset < 0 or limit <= 0:
        raise ValueError("--offset must be non-negative and --max-compounds must be positive")
    member = f"Val-Test-set/PGK2_{panel.capitalize()}_split.csv"
    with zipfile.ZipFile(path) as archive, archive.open(member) as handle:
        rows = list(csv.DictReader(line.decode("utf-8") for line in handle))
    selected = rows[offset : offset + limit]
    if not selected:
        raise ValueError(f"No {panel} rows in requested range offset={offset}, limit={limit}")
    if set(selected[0]) != {"CatalogID", "SMILES"}:
        raise ValueError(f"Unexpected columns in {member}: {list(selected[0])}")
    return selected


def main() -> int:
    args = parse_args()
    started = time.perf_counter()
    rows = read_panel(args.candidate_panels, args.panel, args.offset, args.max_compounds)
    protein = read_fasta(args.protein_fasta)
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested but CUDA is unavailable")
    pocket_positions = load_pocket_region(args.pocket_json, args.protein_mask_region)
    scorer = LogiCABindingScorer(
        checkpoint=args.checkpoint,
        hf_cache=args.hf_cache,
        mask_fraction=args.mask_fraction,
        alpha=args.alpha,
        seed=args.seed,
        protein_mask_key="PGK2_P07205",
        protein_mask_positions=pocket_positions or None,
        protein_mask_region=args.protein_mask_region,
        score_mode=args.score_mode,
        device=device,
    )
    if args.adapter is not None:
        if not args.adapter.exists():
            raise FileNotFoundError(args.adapter)
        payload = torch.load(args.adapter, map_location="cpu", weights_only=False)
        if payload.get("target") != "PGK2" or payload.get("uniprot") != "P07205":
            raise ValueError("Adapter metadata does not identify the canonical PGK2 target")
        incompatible = scorer.model.load_state_dict(payload["model_state_dict"], strict=False)
        if incompatible.unexpected_keys:
            raise ValueError(f"Unexpected adapter weights: {incompatible.unexpected_keys}")
    prepared, diagnostics = scorer.prepare(
        protein,
        [row["SMILES"] for row in rows],
        encoder_batch_size=args.encoder_batch_size,
    )
    scores = scorer.score_prepared(prepared, batch_size=args.batch_size)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "" if args.protein_mask_region == "random" else f"_{args.protein_mask_region}"
    if args.score_mode != "masked_gain":
        suffix += f"_{args.score_mode}"
    output = args.output_dir / f"{args.panel}_offset{args.offset}_n{len(rows)}{suffix}.csv"
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["CatalogID", "SMILES", "logica_score"])
        writer.writeheader()
        for row, score in zip(rows, scores, strict=True):
            writer.writerow({**row, "logica_score": f"{score:.10g}"})
    print(
        {
            "target": "PGK2",
            "uniprot": "P07205",
            "panel": args.panel,
            "protein_mask_region": args.protein_mask_region,
            "score_mode": args.score_mode,
            "protein_mask_residues": len(pocket_positions) or None,
            "scored_rows": len(rows),
            "output": str(output),
            "adapter": str(args.adapter) if args.adapter else None,
            "device": device,
            "diagnostics": diagnostics,
            "elapsed_seconds": time.perf_counter() - started,
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
