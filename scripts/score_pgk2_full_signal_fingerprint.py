#!/usr/bin/env python3
"""Score a challenge panel with an ensemble of full-signal Morgan models."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from logica_binding.full_graph_model import FingerprintMultitaskModel  # noqa: E402


FP_BITS = 2048
TARGET_NAMES = (
    "log_count_PGK2",
    "log_count_PGK2_with_inhibitor",
    "log_count_NTC_selection",
    "log_count_NTC_supplement",
    "log_historic_hits",
    "zscore_PGK2",
    "zscore_PGK2_with_inhibitor",
    "zscore_NTC_selection",
    "zscore_NTC_supplement",
)
PANEL_MEMBERS = {
    "validation": "Val-Test-set/PGK2_Validation_split.csv",
    "test": "Val-Test-set/PGK2_Test_split.csv",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate-panels",
        type=Path,
        default=ROOT / "DREAM_challenge_2026" / "Val-Test-set.zip",
    )
    parser.add_argument("--panel", choices=tuple(PANEL_MEMBERS), default="validation")
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument(
        "--compare-submission",
        type=Path,
        action="append",
        default=[],
        help="Prior 50-ID submission whose overlap should be audited.",
    )
    return parser.parse_args()


def read_panel(path: Path, panel: str) -> list[dict[str, str]]:
    with zipfile.ZipFile(path) as archive, archive.open(PANEL_MEMBERS[panel]) as handle:
        rows = list(csv.DictReader(line.decode("utf-8") for line in handle))
    if not rows or set(rows[0]) != {"CatalogID", "SMILES"}:
        raise ValueError("Challenge panel must contain exactly CatalogID and SMILES")
    ids = [row["CatalogID"].strip() for row in rows]
    smiles = [row["SMILES"].strip() for row in rows]
    if any(not value for value in ids + smiles):
        raise ValueError("Challenge panel contains blank CatalogID or SMILES")
    if len(set(ids)) != len(ids):
        raise ValueError("Challenge panel contains duplicate CatalogIDs")
    if len(set(smiles)) != len(smiles):
        raise ValueError("Challenge panel contains duplicate SMILES")
    return [{"CatalogID": catalog_id, "SMILES": smile} for catalog_id, smile in zip(ids, smiles)]


def fingerprint_matrix(smiles: list[str]) -> np.ndarray:
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=FP_BITS)
    matrix = np.empty((len(smiles), FP_BITS), dtype=np.float32)
    for index, value in enumerate(smiles):
        molecule = Chem.MolFromSmiles(value)
        if molecule is None:
            raise ValueError(f"Invalid SMILES at batch row {index}: {value!r}")
        DataStructs.ConvertToNumpyArray(generator.GetFingerprint(molecule), matrix[index])
    return matrix


def active_score(values: np.ndarray) -> np.ndarray:
    target = values[:, 0] + 0.25 * values[:, 5]
    inhibitor = values[:, 1] + 0.25 * values[:, 6]
    ntc_selection = values[:, 2] + 0.25 * values[:, 7]
    ntc_supplement = values[:, 3] + 0.25 * values[:, 8]
    return (
        target
        - inhibitor
        - 0.5 * np.maximum(ntc_selection, ntc_supplement)
        - 0.25 * values[:, 4]
    )


def load_model(path: Path) -> tuple[FingerprintMultitaskModel, np.ndarray, np.ndarray, int]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("model_mode") != "fingerprint" or checkpoint.get("graph_mode") != "2d":
        raise ValueError(f"Not a fingerprint/2d checkpoint: {path}")
    if tuple(checkpoint.get("target_names", ())) != TARGET_NAMES:
        raise ValueError(f"Unexpected target ordering in {path}")
    state = checkpoint["model_state_dict"]
    width = int(state["network.0.weight"].shape[0])
    if int(state["network.0.weight"].shape[1]) != FP_BITS:
        raise ValueError(f"Unexpected fingerprint width in {path}")
    model = FingerprintMultitaskModel(FP_BITS, len(TARGET_NAMES), width=width)
    model.load_state_dict(state)
    model.eval()
    scaler = checkpoint["scaler"]
    mean = np.asarray(scaler["mean"], dtype=np.float32)
    scale = np.asarray(scaler["scale"], dtype=np.float32)
    seed = int(checkpoint["training_args"]["seed"])
    return model, mean, scale, seed


def read_submission(path: Path) -> list[str]:
    ids = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(ids) != 50 or len(set(ids)) != 50:
        raise ValueError(f"Prior submission is not 50 unique IDs: {path}")
    return ids


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    if len(args.checkpoint) < 1:
        raise ValueError("At least one checkpoint is required")
    started = time.perf_counter()
    panel = read_panel(args.candidate_panels, args.panel)
    loaded = [load_model(path) for path in args.checkpoint]
    seeds = [item[3] for item in loaded]
    if len(set(seeds)) != len(seeds):
        raise ValueError("Checkpoint seeds must be unique")
    score_columns = [np.empty(len(panel), dtype=np.float32) for _ in loaded]
    with torch.inference_mode():
        for start in range(0, len(panel), args.batch_size):
            stop = min(start + args.batch_size, len(panel))
            fingerprints = torch.from_numpy(
                fingerprint_matrix([row["SMILES"] for row in panel[start:stop]])
            )
            for column, (model, mean, scale, _) in zip(score_columns, loaded):
                standardized = model(fingerprints).numpy()
                column[start:stop] = active_score(standardized * scale + mean)
            print(json.dumps({"scored": stop, "total": len(panel)}), flush=True)
    ensemble = np.mean(np.vstack(score_columns), axis=0)
    order = sorted(range(len(panel)), key=lambda i: (-float(ensemble[i]), panel[i]["CatalogID"]))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ranked_path = args.output_dir / f"{args.panel}_ranked_scores.csv"
    top_path = args.output_dir / f"{args.panel}_top50.csv"
    submission_path = args.output_dir / f"{args.panel}_submission.txt"
    fields = ["rank", "CatalogID", "SMILES", "ensemble_active_score"] + [
        f"seed_{seed}_active_score" for seed in seeds
    ]
    with ranked_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for rank, index in enumerate(order, 1):
            row: dict[str, object] = {
                "rank": rank,
                "CatalogID": panel[index]["CatalogID"],
                "SMILES": panel[index]["SMILES"],
                "ensemble_active_score": float(ensemble[index]),
            }
            row.update(
                {f"seed_{seed}_active_score": float(scores[index]) for seed, scores in zip(seeds, score_columns)}
            )
            writer.writerow(row)
    with ranked_path.open(newline="", encoding="utf-8") as source, top_path.open(
        "w", newline="", encoding="utf-8"
    ) as target:
        reader = csv.DictReader(source)
        writer = csv.DictWriter(target, fieldnames=reader.fieldnames)
        writer.writeheader()
        writer.writerows([next(reader) for _ in range(50)])
    top_ids = [panel[index]["CatalogID"] for index in order[:50]]
    submission_path.write_text("\r\n".join(top_ids) + "\r\n", encoding="utf-8", newline="")
    comparisons = {}
    for path in args.compare_submission:
        prior = read_submission(path)
        overlap = sorted(set(top_ids) & set(prior))
        comparisons[path.name] = {"overlap_count": len(overlap), "overlap_ids": overlap}
    report = {
        "target": "human PGK2",
        "panel": args.panel,
        "panel_source": f"{args.candidate_panels}:{PANEL_MEMBERS[args.panel]}",
        "panel_rows": len(panel),
        "unique_catalog_ids": len({row["CatalogID"] for row in panel}),
        "unique_smiles": len({row["SMILES"] for row in panel}),
        "model": "three-seed full-signal Morgan radius-2/2048-bit multitask ensemble",
        "checkpoint_seeds": seeds,
        "checkpoint_sha256": {str(path): sha256(path) for path in args.checkpoint},
        "top50_unique": len(set(top_ids)),
        "prior_submission_comparisons": comparisons,
        "outputs": {
            "ranked": str(ranked_path),
            "top50": str(top_path),
            "submission": str(submission_path),
        },
        "controls": {
            "challenge_labels_used": False,
            "blind_test_scored": args.panel == "test",
            "ranking_direction": "descending predicted active-site competition score",
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    report_path = args.output_dir / f"{args.panel}_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
