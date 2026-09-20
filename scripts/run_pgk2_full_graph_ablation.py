#!/usr/bin/env python3
"""Train and evaluate full-signal PGK2 graph models on the fixed ablation set."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import polars as pl
import torch
from torch import nn
from scipy.stats import rankdata as scipy_rankdata
from sklearn.metrics import average_precision_score, roc_auc_score
from transformers import AutoModelForMaskedLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from logica_binding.data import read_fasta  # noqa: E402
from logica_binding.full_graph_model import (  # noqa: E402
    FingerprintMultitaskModel,
    PGK2PocketMultitaskModel,
    PackedLigandBatch,
)
from logica_binding.graph_model import PocketGraph, build_pocket_graph  # noqa: E402


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
TASK_WEIGHTS = np.asarray([1.0, 1.5, 1.0, 1.0, 0.5, 1.0, 1.5, 1.0, 1.0], dtype=np.float32)


@dataclass(frozen=True)
class TargetScaler:
    names: tuple[str, ...]
    mean: tuple[float, ...]
    scale: tuple[float, ...]

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (values - np.asarray(self.mean)) / np.asarray(self.scale)

    def inverse(self, values: np.ndarray) -> np.ndarray:
        return values * np.asarray(self.scale) + np.asarray(self.mean)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument(
        "--index",
        type=Path,
        default=ROOT
        / "artifacts"
        / "pgk2_full_canonical_index"
        / "canonical_training_index.parquet",
    )
    parser.add_argument(
        "--model-mode", choices=("fingerprint", "ligand", "pocket"), required=True
    )
    parser.add_argument("--graph-mode", choices=("2d", "etkdg"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--protein-fasta",
        type=Path,
        default=ROOT / "data" / "proteins" / "PGK2_P07205.fasta",
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
    parser.add_argument("--esm-model", default="facebook/esm2_t6_8M_UR50D")
    parser.add_argument("--hf-cache", type=Path, default=ROOT / "models" / "hf-cache")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=192)
    parser.add_argument("--graph-width", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--data-scope",
        choices=("ablation", "all"),
        default="ablation",
        help="Use the preregistered comparison subset or every row in each scaffold split.",
    )
    parser.add_argument(
        "--selection-metric",
        choices=(
            "active_score_spearman",
            "competition_roc_auc",
            "competition_average_precision",
            "neg_loss",
        ),
        default="competition_average_precision",
    )
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-eval-batches", type=int, default=0)
    parser.add_argument("--max-shards", type=int, default=0)
    return parser.parse_args()


def raw_targets(frame: pl.DataFrame) -> np.ndarray:
    def values(name: str) -> np.ndarray:
        return frame[name].to_numpy().astype(np.float32, copy=False)

    return np.column_stack(
        [
            np.log1p(values("count_PGK2")),
            np.log1p(values("count_PGK2_with_inhibitor")),
            np.log1p(values("count_NTC_selection")),
            np.log1p(values("count_NTC_supplement")),
            np.log1p(values("historic_hits")),
            np.clip(values("zscore_PGK2"), -20, 20),
            np.clip(values("zscore_PGK2_with_inhibitor"), -20, 20),
            np.clip(values("zscore_NTC_selection"), -20, 20),
            np.clip(values("zscore_NTC_supplement"), -20, 20),
        ]
    ).astype(np.float32, copy=False)


def scope_filter(split: str, data_scope: str) -> pl.Expr:
    selected = pl.col("scaffold_split") == split
    if data_scope == "ablation":
        selected &= pl.col("informative") | (pl.col("sample_bucket") < 14)
    return selected


def fit_scaler(index: Path, data_scope: str) -> TargetScaler:
    frame = (
        pl.scan_parquet(index)
        .filter(scope_filter("train", data_scope))
        .collect(engine="streaming")
    )
    targets = raw_targets(frame)
    mean = targets.mean(axis=0, dtype=np.float64)
    scale = targets.std(axis=0, dtype=np.float64)
    scale[scale < 1e-6] = 1.0
    return TargetScaler(TARGET_NAMES, tuple(mean.tolist()), tuple(scale.tolist()))


def evidence_weights(frame: pl.DataFrame) -> np.ndarray:
    total = (
        frame["count_PGK2"].to_numpy()
        + frame["count_PGK2_with_inhibitor"].to_numpy()
        + frame["count_NTC_selection"].to_numpy()
        + frame["count_NTC_supplement"].to_numpy()
        + frame["historic_hits"].to_numpy()
    )
    return np.clip(1.0 + np.log1p(total), 1.0, 5.0).astype(np.float32)


def discover_shards(data_dir: Path, max_shards: int) -> list[tuple[Path, Path, Path]]:
    reports = sorted(data_dir.glob("*-report.json"))
    if max_shards > 0:
        reports = reports[:max_shards]
    result: list[tuple[Path, Path, Path]] = []
    for report_path in reports:
        stem = report_path.name.removesuffix("-report.json")
        metadata = data_dir / f"{stem}-metadata.parquet"
        graphs = data_dir / f"{stem}-graphs.npz"
        if not metadata.exists() or not graphs.exists():
            raise FileNotFoundError(f"Incomplete graph shard {stem}")
        result.append((metadata, graphs, report_path))
    if not result:
        raise ValueError(f"No complete graph shards in {data_dir}")
    return result


def validate_shards(shards: list[tuple[Path, Path, Path]], graph_mode: str) -> None:
    for _, _, report_path in shards:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if not report.get("completed") or report.get("graph_mode") != graph_mode:
            raise ValueError(f"Unexpected report in {report_path}")


def select_positions(frame: pl.DataFrame, split: str, data_scope: str) -> np.ndarray:
    mask = (frame["scaffold_split"].to_numpy() == split) & (
        frame["graph_status"].to_numpy() == "ok"
    )
    if data_scope == "ablation":
        mask &= frame["informative"].to_numpy() | (frame["sample_bucket"].to_numpy() < 14)
    return np.flatnonzero(mask)


def build_packed_batch(
    archive: dict[str, np.ndarray], indices: np.ndarray, need_graph: bool, need_fp: bool
) -> PackedLigandBatch:
    fingerprints: torch.Tensor | None = None
    if need_fp:
        packed = archive["fingerprints"][indices]
        bits = np.unpackbits(packed, axis=1, bitorder="little").astype(np.float32)
        fingerprints = torch.from_numpy(bits)
    if not need_graph:
        return PackedLigandBatch(
            node_features=torch.empty((0, 20)),
            edge_index=torch.empty((2, 0), dtype=torch.long),
            edge_features=torch.empty((0, 7)),
            node_offsets=torch.zeros(len(indices) + 1, dtype=torch.long),
            fingerprints=fingerprints,
        )
    nodes: list[np.ndarray] = []
    edges: list[np.ndarray] = []
    edge_features: list[np.ndarray] = []
    offsets = [0]
    for index in indices:
        node_start = int(archive["node_offsets"][index])
        node_stop = int(archive["node_offsets"][index + 1])
        edge_start = int(archive["edge_offsets"][index])
        edge_stop = int(archive["edge_offsets"][index + 1])
        nodes.append(archive["node_features"][node_start:node_stop])
        edges.append(archive["edge_index"][:, edge_start:edge_stop] + offsets[-1])
        edge_features.append(archive["edge_features"][edge_start:edge_stop])
        offsets.append(offsets[-1] + node_stop - node_start)
    return PackedLigandBatch(
        node_features=torch.from_numpy(np.concatenate(nodes, axis=0)),
        edge_index=torch.from_numpy(np.concatenate(edges, axis=1)).long(),
        edge_features=torch.from_numpy(np.concatenate(edge_features, axis=0)),
        node_offsets=torch.as_tensor(offsets, dtype=torch.long),
        fingerprints=fingerprints,
    )


def load_archive(path: Path, need_graph: bool, need_fp: bool) -> dict[str, np.ndarray]:
    keys = ["fingerprints"] if need_fp else []
    if need_graph:
        keys.extend(
            ["node_features", "edge_index", "edge_features", "node_offsets", "edge_offsets"]
        )
    with np.load(path) as archive:
        return {key: archive[key] for key in keys}


def iter_batches(
    shards: list[tuple[Path, Path, Path]],
    split: str,
    batch_size: int,
    seed: int,
    shuffle: bool,
    need_graph: bool,
    need_fp: bool,
    max_batches: int,
    data_scope: str,
) -> Iterator[tuple[PackedLigandBatch, pl.DataFrame]]:
    rng = np.random.default_rng(seed)
    order = np.arange(len(shards))
    if shuffle:
        rng.shuffle(order)
    yielded = 0
    for shard_position in order:
        metadata_path, graph_path, _ = shards[int(shard_position)]
        frame = pl.read_parquet(metadata_path)
        positions = select_positions(frame, split, data_scope)
        if shuffle:
            rng.shuffle(positions)
        if not len(positions):
            continue
        archive = load_archive(graph_path, need_graph, need_fp)
        for start in range(0, len(positions), batch_size):
            selected = positions[start : start + batch_size]
            yield build_packed_batch(archive, selected, need_graph, need_fp), frame[selected]
            yielded += 1
            if max_batches > 0 and yielded >= max_batches:
                return


def encode_full_sequence_esm(
    sequence: str, model_name: str, cache_dir: Path, device: torch.device
) -> torch.Tensor:
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, cache_dir=cache_dir, local_files_only=True
    )
    model = AutoModelForMaskedLM.from_pretrained(
        model_name, cache_dir=cache_dir, local_files_only=True
    ).to(device)
    tokens = tokenizer(sequence, return_tensors="pt")
    with torch.inference_mode():
        hidden = model.base_model(
            input_ids=tokens["input_ids"].to(device),
            attention_mask=tokens["attention_mask"].to(device),
            return_dict=True,
        ).last_hidden_state[0, 1 : len(sequence) + 1]
    result = hidden.detach().cpu()
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def multitask_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    evidence: torch.Tensor,
    task_weights: torch.Tensor,
) -> torch.Tensor:
    per_task = torch.nn.functional.smooth_l1_loss(
        predictions, targets, reduction="none", beta=1.0
    )
    per_row = (per_task * task_weights[None, :]).sum(dim=1) / task_weights.sum()
    return (per_row * evidence).sum() / evidence.sum().clamp_min(1e-8)


def rankdata(values: np.ndarray) -> np.ndarray:
    return scipy_rankdata(values, method="average")


def spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(rankdata(x), rankdata(y))[0, 1])


def roc_auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    positive = labels == 1
    n_positive = int(positive.sum())
    n_negative = int((~positive).sum())
    if not n_positive or not n_negative:
        return None
    return float(roc_auc_score(labels, scores))


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float | None:
    positives = int(labels.sum())
    if positives == 0 or positives == len(labels):
        return None
    return float(average_precision_score(labels, scores))


def active_score(values: np.ndarray) -> np.ndarray:
    target = values[:, 0] + 0.25 * values[:, 5]
    inhibitor = values[:, 1] + 0.25 * values[:, 6]
    ntc_selection = values[:, 2] + 0.25 * values[:, 7]
    ntc_supplement = values[:, 3] + 0.25 * values[:, 8]
    return target - inhibitor - 0.5 * np.maximum(ntc_selection, ntc_supplement) - 0.25 * values[:, 4]


def evaluation_labels(frame: pl.DataFrame) -> np.ndarray:
    target = frame["count_PGK2"].to_numpy()
    inhibitor = frame["count_PGK2_with_inhibitor"].to_numpy()
    ntc = np.maximum(
        frame["count_NTC_selection"].to_numpy(),
        frame["count_NTC_supplement"].to_numpy(),
    )
    historic = frame["historic_hits"].to_numpy()
    clean = (target >= 4) & (ntc < 0.1 * target) & (historic < 5)
    positive = clean & (inhibitor < 0.1 * target)
    negative = clean & (inhibitor >= 0.5 * target)
    labels = np.full(frame.height, -1, dtype=np.int8)
    labels[positive] = 1
    labels[negative] = 0
    return labels


@torch.no_grad()
def evaluate(
    model: nn.Module,
    model_mode: str,
    shards: list[tuple[Path, Path, Path]],
    split: str,
    scaler: TargetScaler,
    batch_size: int,
    seed: int,
    device: torch.device,
    pocket_graphs: list[PocketGraph],
    esm: torch.Tensor,
    max_batches: int,
    data_scope: str,
) -> dict[str, object]:
    model.eval()
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    losses: list[float] = []
    task_weights = torch.as_tensor(TASK_WEIGHTS, device=device)
    for batch, frame in iter_batches(
        shards,
        split,
        batch_size,
        seed,
        False,
        model_mode != "fingerprint",
        model_mode == "fingerprint",
        max_batches,
        data_scope,
    ):
        batch = batch.to(device)
        raw = raw_targets(frame)
        target = torch.as_tensor(scaler.transform(raw), dtype=torch.float32, device=device)
        evidence = torch.as_tensor(evidence_weights(frame), device=device)
        if model_mode == "fingerprint":
            assert batch.fingerprints is not None
            prediction = model(batch.fingerprints)
        else:
            prediction = model(batch, pocket_graphs, esm)
        losses.append(float(multitask_loss(prediction, target, evidence, task_weights).cpu()))
        predictions.append(prediction.cpu().numpy())
        targets.append(raw)
        labels.append(evaluation_labels(frame))
    standardized = np.concatenate(predictions)
    predicted_raw = scaler.inverse(standardized)
    true_raw = np.concatenate(targets)
    class_labels = np.concatenate(labels)
    predicted_score = active_score(predicted_raw)
    true_score = active_score(true_raw)
    classified = class_labels >= 0
    return {
        "split": split,
        "rows": int(len(true_raw)),
        "weighted_multitask_loss": float(np.mean(losses)),
        "active_score_spearman": spearman(predicted_score, true_score),
        "competition_classification_rows": int(classified.sum()),
        "competition_sensitive_rows": int((class_labels[classified] == 1).sum()),
        "competition_retained_rows": int((class_labels[classified] == 0).sum()),
        "competition_roc_auc": roc_auc(class_labels[classified], predicted_score[classified]),
        "competition_average_precision": average_precision(
            class_labels[classified], predicted_score[classified]
        ),
        "task_spearman": {
            name: spearman(predicted_raw[:, index], true_raw[:, index])
            for index, name in enumerate(TARGET_NAMES)
        },
    }


def main() -> int:
    args = parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or args.graph_width <= 0:
        raise ValueError("epochs, batch-size, and graph-width must be positive")
    for path in (
        args.data_dir,
        args.index,
        args.protein_fasta,
        args.pocket_json,
        args.cmp21_pdb,
        args.cmp47_pdb,
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "report.json"
    checkpoint_path = args.output_dir / "model.pt"
    if report_path.exists() or checkpoint_path.exists():
        raise FileExistsError(f"Refusing to overwrite completed output in {args.output_dir}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu"
        if args.device == "auto"
        else args.device
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    started = time.perf_counter()
    shards = discover_shards(args.data_dir, args.max_shards)
    validate_shards(shards, args.graph_mode)
    scaler = fit_scaler(args.index, args.data_scope)
    sequence = read_fasta(args.protein_fasta)
    esm = encode_full_sequence_esm(
        sequence, args.esm_model, args.hf_cache, device
    ).to(device)
    pocket_graphs = [
        build_pocket_graph(args.cmp21_pdb, args.pocket_json).to(device),
        build_pocket_graph(args.cmp47_pdb, args.pocket_json).to(device),
    ]
    if args.model_mode == "fingerprint":
        model: nn.Module = FingerprintMultitaskModel(2048, len(TARGET_NAMES)).to(device)
    else:
        model = PGK2PocketMultitaskModel(
            esm_width=esm.shape[1],
            output_dim=len(TARGET_NAMES),
            graph_width=args.graph_width,
            mode=args.model_mode,
        ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    task_weights = torch.as_tensor(TASK_WEIGHTS, device=device)
    history: list[dict[str, object]] = []
    best_metric = -math.inf
    best_state: dict[str, torch.Tensor] | None = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses: list[float] = []
        rows_seen = 0
        for batch, frame in iter_batches(
            shards,
            "train",
            args.batch_size,
            args.seed + epoch,
            True,
            args.model_mode != "fingerprint",
            args.model_mode == "fingerprint",
            args.max_train_batches,
            args.data_scope,
        ):
            batch = batch.to(device)
            target = torch.as_tensor(
                scaler.transform(raw_targets(frame)), dtype=torch.float32, device=device
            )
            evidence = torch.as_tensor(evidence_weights(frame), device=device)
            optimizer.zero_grad(set_to_none=True)
            if args.model_mode == "fingerprint":
                assert batch.fingerprints is not None
                prediction = model(batch.fingerprints)
            else:
                prediction = model(batch, pocket_graphs, esm)
            loss = multitask_loss(prediction, target, evidence, task_weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            rows_seen += frame.height
        dev = evaluate(
            model,
            args.model_mode,
            shards,
            "dev",
            scaler,
            args.batch_size,
            args.seed,
            device,
            pocket_graphs,
            esm,
            args.max_eval_batches,
            args.data_scope,
        )
        metric = (
            -float(dev["weighted_multitask_loss"])
            if args.selection_metric == "neg_loss"
            else dev[args.selection_metric]
        )
        selected_value = -float(dev["weighted_multitask_loss"]) if metric is None else float(metric)
        if selected_value > best_metric:
            best_metric = selected_value
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
        item = {
            "epoch": epoch,
            "train_rows": rows_seen,
            "train_loss": float(np.mean(losses)),
            "dev": dev,
        }
        history.append(item)
        print(json.dumps(item), flush=True)
    assert best_state is not None
    model.load_state_dict(best_state)
    holdout = evaluate(
        model,
        args.model_mode,
        shards,
        "holdout",
        scaler,
        args.batch_size,
        args.seed,
        device,
        pocket_graphs,
        esm,
        args.max_eval_batches,
        args.data_scope,
    )
    checkpoint = {
        "target": "PGK2",
        "uniprot": "P07205",
        "model_mode": args.model_mode,
        "graph_mode": args.graph_mode,
        "model_state_dict": best_state,
        "scaler": asdict(scaler),
        "target_names": TARGET_NAMES,
        "task_weights": TASK_WEIGHTS.tolist(),
        "training_args": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "holdout": holdout,
    }
    torch.save(checkpoint, checkpoint_path)
    report = {
        "definition": "Paired full-signal PGK2 ligand representation ablation",
        "model_mode": args.model_mode,
        "graph_mode": args.graph_mode,
        "seed": args.seed,
        "shards": len(shards),
        "scaler": asdict(scaler),
        "task_weights": dict(zip(TARGET_NAMES, TASK_WEIGHTS.tolist())),
        "history": history,
        "holdout": holdout,
        "checkpoint": str(checkpoint_path),
        "elapsed_seconds": time.perf_counter() - started,
        "controls": {
            "data_scope": args.data_scope,
            "selection_metric": args.selection_metric,
            "candidate_panel_feedback_used": False,
            "scaffold_holdout_used_for_model_selection": False,
            "protein_sequence_encoding": "full 417-residue PGK2 ESM; frozen",
            "direct_pocket_edges": "compound-21/47 26-residue envelope only",
            "protein_templates": [str(args.cmp21_pdb), str(args.cmp47_pdb)],
            "protein_ligand_distance": "not used; coordinate frames are independent",
        },
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"completed": True, "holdout": holdout, "output": str(report_path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
