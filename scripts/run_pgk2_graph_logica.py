#!/usr/bin/env python3
"""Score a PGK2 challenge-panel slice with trained Graph LogiCA."""

from __future__ import annotations

import argparse
import csv
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from logica_binding.data import read_fasta
from logica_binding.graph_model import (
    GraphLogiCAModel,
    build_pocket_graph,
    score_graph_prepared,
    smiles_to_rdkit_3d_graph,
)
from logica_binding.model import LogiCABindingScorer
from logica_binding.pocket import load_pocket_region
from logica_binding.validation_protocol import PROTOCOL
from logica_binding.graph_model import smiles_to_ligand_graph


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", choices=("validation", "test"), default="validation")
    parser.add_argument(
        "--candidate-panels",
        type=Path,
        default=ROOT / "DREAM_challenge_2026" / "Val-Test-set.zip",
    )
    parser.add_argument(
        "--adapter",
        type=Path,
        default=ROOT / "artifacts" / "pgk2_graph_logica_competition_finetune" / "graph_logica_adapter.pt",
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
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--max-compounds", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--encoder-batch-size", type=int, default=32)
    parser.add_argument("--graph-workers", type=int, default=1)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "artifacts" / "pgk2_graph_logica"
    )
    return parser.parse_args()


def read_panel(path: Path, panel: str, offset: int, limit: int) -> list[dict[str, str]]:
    if offset < 0 or limit <= 0:
        raise ValueError("offset must be non-negative and max-compounds positive")
    member = f"Val-Test-set/PGK2_{panel.capitalize()}_split.csv"
    with zipfile.ZipFile(path) as archive, archive.open(member) as handle:
        rows = list(csv.DictReader(line.decode("utf-8") for line in handle))
    selected = rows[offset : offset + limit]
    if not selected:
        raise ValueError("Requested panel slice is empty")
    return selected


def build_graph_task(task: tuple[str, int, float]):
    smiles, seed, radius = task
    return smiles_to_rdkit_3d_graph(smiles, seed=seed, spatial_cutoff=radius)


def main() -> int:
    args = parse_args()
    started = time.perf_counter()
    for path in (
        args.candidate_panels,
        args.adapter,
        args.protein_fasta,
        args.checkpoint,
        args.pocket_json,
        args.cmp21_pdb,
        args.cmp47_pdb,
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.graph_workers <= 0:
        raise ValueError("graph-workers must be positive")
    payload = torch.load(args.adapter, map_location="cpu", weights_only=False)
    if payload.get("protocol") != PROTOCOL:
        raise ValueError("Historical adapter lacks the corrected interface/split protocol; retraining is required")
    if not payload.get("candidate_scoring_allowed", False):
        raise ValueError("Diagnostic adapters are not eligible for challenge candidate scoring")
    if payload.get("target") != "PGK2" or payload.get("uniprot") != "P07205":
        raise ValueError("Graph adapter does not identify canonical human PGK2")
    training = payload["training_args"]
    if training.get("smoke_pairs_per_split", 0):
        raise ValueError("Smoke checkpoints cannot score challenge candidates")
    graph_width = int(training["graph_width"])
    ligand_radius = float(training["ligand_radius"])
    seed = int(training["seed"])
    rows = read_panel(args.candidate_panels, args.panel, args.offset, args.max_compounds)
    smiles = [row["SMILES"] for row in rows]
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
        alpha=float(training["alpha"]),
        seed=seed,
        protein_mask_positions=pocket_positions,
        protein_mask_region="inhibitor_pocket",
        score_mode="token_likelihood",
        device=device,
    )
    scorer.model.load_state_dict(payload["base_state_dict"], strict=False)
    prepared, diagnostics = scorer.prepare(
        read_fasta(args.protein_fasta), smiles, encoder_batch_size=args.encoder_batch_size
    )
    tasks = [(value, seed, ligand_radius) for value in smiles]
    if training["ligand_mode"] == "2d":
        ligand_graphs = [smiles_to_ligand_graph(value) for value in smiles]
    elif args.graph_workers == 1:
        ligand_graphs = [build_graph_task(task) for task in tasks]
    else:
        with ThreadPoolExecutor(max_workers=args.graph_workers) as executor:
            ligand_graphs = list(executor.map(build_graph_task, tasks))
    pocket_graphs = [
        build_pocket_graph(args.cmp21_pdb, args.pocket_json),
        build_pocket_graph(args.cmp47_pdb, args.pocket_json),
    ]
    model = GraphLogiCAModel(scorer.model, graph_width=graph_width,
                            graph_conditioning=training["graph_conditioning"] == "on").to(device)
    model.conditioner.load_state_dict(payload["conditioner_state_dict"], strict=True)
    model.eval()
    scores: list[float] = []
    with torch.inference_mode():
        for offset in range(0, len(rows), args.batch_size):
            indices = torch.arange(offset, min(offset + args.batch_size, len(rows)))
            scores.extend(
                float(value)
                for value in score_graph_prepared(
                    model, prepared, indices, pocket_graphs, ligand_graphs
                ).cpu()
            )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / f"{args.panel}_offset{args.offset}_n{len(rows)}.csv"
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["CatalogID", "SMILES", "logica_score"])
        writer.writeheader()
        for row, score in zip(rows, scores, strict=True):
            writer.writerow({**row, "logica_score": f"{score:.10g}"})
    print(
        {
            "panel": args.panel,
            "rows": len(rows),
            "output": str(output),
            "device": device,
            "graph_mode": payload["graph_mode"],
            "diagnostics": diagnostics,
            "elapsed_seconds": time.perf_counter() - started,
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
