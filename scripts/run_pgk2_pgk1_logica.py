#!/usr/bin/env python3
"""Score one challenge-panel chunk against PGK2 and its PGK1 off-target.

The two 417-residue human paralogues use the same deterministic protein mask
key so their LogiCA context-gain difference is not confounded by selecting
different masked residue positions.  The resulting delta is a sequence-model
feature, not a calibrated biochemical selectivity estimate.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import zipfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from logica_binding.data import read_fasta
from logica_binding.model import LogiCABindingScorer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", choices=("validation", "test"), default="validation")
    parser.add_argument(
        "--candidate-panels",
        type=Path,
        default=ROOT / "DREAM_challenge_2026" / "Val-Test-set.zip",
    )
    parser.add_argument(
        "--pgk2-fasta",
        type=Path,
        default=ROOT / "data" / "proteins" / "PGK2_P07205.fasta",
    )
    parser.add_argument(
        "--pgk1-fasta",
        type=Path,
        default=ROOT / "data" / "proteins" / "PGK1_P00558.fasta",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "models" / "logica-8m" / "checkpoints" / "8m" / "best.pt",
    )
    parser.add_argument("--hf-cache", type=Path, default=ROOT / "models" / "hf-cache")
    parser.add_argument("--adapter", type=Path, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--max-compounds", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--encoder-batch-size", type=int, default=8)
    parser.add_argument("--mask-fraction", type=float, default=0.15)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--protein-mask-key",
        default="PGK1_PGK2_ALIGNED_SELECTIVITY",
        help="Shared key that forces identical masked positions in the aligned proteins.",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "artifacts" / "pgk2_selectivity_logica"
    )
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


def validate_aligned_paralogues(pgk2: str, pgk1: str) -> None:
    if len(pgk2) != 417 or len(pgk1) != 417:
        raise ValueError(f"Expected 417-residue PGK paralogues; observed {len(pgk2)} and {len(pgk1)}")
    if (pgk2[241], pgk1[241]) != ("Y", "F"):
        raise ValueError("Expected PGK2 Y242 / PGK1 F242 selectivity substitution")
    if (pgk2[254], pgk1[254]) != ("A", "T"):
        raise ValueError("Expected PGK2 A255 / PGK1 T255 selectivity substitution")


def main() -> int:
    args = parse_args()
    started = time.perf_counter()
    rows = read_panel(args.candidate_panels, args.panel, args.offset, args.max_compounds)
    smiles = [row["SMILES"] for row in rows]
    pgk2 = read_fasta(args.pgk2_fasta)
    pgk1 = read_fasta(args.pgk1_fasta)
    validate_aligned_paralogues(pgk2, pgk1)
    device = (
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested but CUDA is unavailable")

    scorer = LogiCABindingScorer(
        checkpoint=args.checkpoint,
        hf_cache=args.hf_cache,
        mask_fraction=args.mask_fraction,
        alpha=args.alpha,
        seed=args.seed,
        protein_mask_key=args.protein_mask_key,
        device=device,
    )
    if args.adapter is not None:
        if not args.adapter.exists():
            raise FileNotFoundError(args.adapter)
        payload = torch.load(args.adapter, map_location="cpu", weights_only=False)
        if payload.get("target") != "PGK2" or payload.get("uniprot") != "P07205":
            raise ValueError("Adapter metadata does not identify canonical PGK2")
        incompatible = scorer.model.load_state_dict(payload["model_state_dict"], strict=False)
        if incompatible.unexpected_keys:
            raise ValueError(f"Unexpected adapter weights: {incompatible.unexpected_keys}")

    pgk2_prepared, diagnostics = scorer.prepare(
        pgk2, smiles, encoder_batch_size=args.encoder_batch_size
    )
    pgk2_scores = scorer.score_prepared(pgk2_prepared, batch_size=args.batch_size)
    pgk1_prepared, _ = scorer.prepare(pgk1, smiles, encoder_batch_size=args.encoder_batch_size)
    pgk1_scores = scorer.score_prepared(pgk1_prepared, batch_size=args.batch_size)
    deltas = pgk2_scores - pgk1_scores

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / f"{args.panel}_offset{args.offset}_n{len(rows)}.csv"
    with output.open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "CatalogID",
            "SMILES",
            "pgk2_logica_score",
            "pgk1_logica_score",
            "logica_selectivity_delta",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row, pgk2_score, pgk1_score, delta in zip(
            rows, pgk2_scores, pgk1_scores, deltas, strict=True
        ):
            writer.writerow(
                {
                    **row,
                    "pgk2_logica_score": f"{pgk2_score:.10g}",
                    "pgk1_logica_score": f"{pgk1_score:.10g}",
                    "logica_selectivity_delta": f"{delta:.10g}",
                }
            )
    report = {
        "target": "PGK2 P07205",
        "off_target": "PGK1 P00558",
        "panel": args.panel,
        "scored_rows": len(rows),
        "output": str(output),
        "adapter": str(args.adapter) if args.adapter else None,
        "shared_protein_mask_key": args.protein_mask_key,
        "selectivity_substitutions": {"242": "Y/F", "255": "A/T"},
        "device": device,
        "diagnostics": diagnostics,
        "limitations": [
            "The score difference is a sequence-model feature, not measured PGK2/PGK1 selectivity.",
            "The PGK2-trained adapter is applied unchanged to both paralogues.",
        ],
        "elapsed_seconds": time.perf_counter() - started,
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
