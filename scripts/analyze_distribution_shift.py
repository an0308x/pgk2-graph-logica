#!/usr/bin/env python3
"""Measure chemical distribution shift between challenge train and test sets."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, Lipinski, rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


DESCRIPTORS = (
    "molecular_weight",
    "logp",
    "tpsa",
    "hbond_donors",
    "hbond_acceptors",
    "rotatable_bonds",
    "ring_count",
    "fraction_csp3",
    "heavy_atoms",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train",
        type=Path,
        default=ROOT / "DREAM_challenge" / "DREAM_Challenge_1_TrainSet.parquet",
    )
    parser.add_argument(
        "--test",
        type=Path,
        default=ROOT / "DREAM_challenge" / "DREAM_Target2035_Challenge_test_data.csv",
    )
    parser.add_argument(
        "--task2-scores", type=Path, default=ROOT / "artifacts" / "task2" / "split_scores.csv"
    )
    parser.add_argument(
        "--zero-shot-test",
        type=Path,
        default=ROOT / "artifacts" / "task1" / "test_predictions.csv",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "artifacts" / "distribution_shift"
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--domain-classifier-train-sample", type=int, default=10_000)
    return parser.parse_args()


def descriptor_vector(mol: Chem.Mol) -> np.ndarray:
    return np.asarray(
        [
            Descriptors.MolWt(mol),
            Descriptors.MolLogP(mol),
            Descriptors.TPSA(mol),
            Lipinski.NumHDonors(mol),
            Lipinski.NumHAcceptors(mol),
            Lipinski.NumRotatableBonds(mol),
            Lipinski.RingCount(mol),
            Descriptors.FractionCSP3(mol),
            Lipinski.HeavyAtomCount(mol),
        ],
        dtype=np.float64,
    )


def canonical_scaffold(mol: Chem.Mol) -> str:
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    return Chem.MolToSmiles(scaffold, canonical=True) if scaffold.GetNumAtoms() else ""


def population_stability_index(train: np.ndarray, test: np.ndarray, bins: int = 10) -> float:
    edges = np.unique(np.quantile(train, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    train_counts = np.histogram(train, bins=edges)[0].astype(float)
    test_counts = np.histogram(test, bins=edges)[0].astype(float)
    train_fraction = np.maximum(train_counts / train_counts.sum(), 1e-6)
    test_fraction = np.maximum(test_counts / test_counts.sum(), 1e-6)
    return float(np.sum((test_fraction - train_fraction) * np.log(test_fraction / train_fraction)))


def descriptor_shift_rows(train: np.ndarray, test: np.ndarray) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for column, name in enumerate(DESCRIPTORS):
        train_values, test_values = train[:, column], test[:, column]
        train_sd = float(np.std(train_values, ddof=1)) or 1.0
        lower, upper = np.quantile(train_values, [0.01, 0.99])
        ks = ks_2samp(train_values, test_values, alternative="two-sided", method="asymp")
        rows.append(
            {
                "descriptor": name,
                "train_mean": float(np.mean(train_values)),
                "train_sd": train_sd,
                "train_median": float(np.median(train_values)),
                "test_mean": float(np.mean(test_values)),
                "test_sd": float(np.std(test_values, ddof=1)),
                "test_median": float(np.median(test_values)),
                "standardized_mean_difference": float(
                    (np.mean(test_values) - np.mean(train_values)) / train_sd
                ),
                "normalized_wasserstein": float(
                    wasserstein_distance(train_values, test_values) / train_sd
                ),
                "ks_statistic": float(ks.statistic),
                "ks_p_value": float(ks.pvalue),
                "psi": population_stability_index(train_values, test_values),
                "test_outside_train_1_99pct": float(
                    np.mean((test_values < lower) | (test_values > upper))
                ),
            }
        )
    return rows


def fp_to_array(fp) -> np.ndarray:
    result = np.zeros((fp.GetNumBits(),), dtype=np.float64)
    DataStructs.ConvertToNumpyArray(fp, result)
    return result


def domain_classifier_auc(
    source_a_descriptors: np.ndarray,
    source_a_fps: list,
    source_b_descriptors: np.ndarray,
    source_b_fps: list,
    source_a_sample: int,
    seed: int,
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    size = min(source_a_sample, len(source_a_fps))
    indices = rng.choice(len(source_a_fps), size=size, replace=False)
    descriptors = np.vstack([source_a_descriptors[indices], source_b_descriptors])
    fingerprints = np.vstack(
        [
            *[fp_to_array(source_a_fps[int(i)]) for i in indices],
            *[fp_to_array(fp) for fp in source_b_fps],
        ]
    )
    labels = np.concatenate([np.zeros(size, dtype=int), np.ones(len(source_b_fps), dtype=int)])
    features = np.hstack([fingerprints, descriptors])
    classifier = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=0.1,
            class_weight="balanced",
            max_iter=2_000,
            solver="liblinear",
            random_state=seed,
        ),
    )
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    scores = cross_val_score(classifier, features, labels, cv=splitter, scoring="roc_auc")
    return {
        "method": "5-fold logistic regression on Morgan fingerprints plus descriptors",
        "source_a_rows": size,
        "source_b_rows": len(source_b_fps),
        "fold_aurocs": [float(value) for value in scores],
        "mean_auroc": float(np.mean(scores)),
        "sd_auroc": float(np.std(scores, ddof=1)),
    }


def max_similarities(query_fps: list, reference_fps: list) -> tuple[np.ndarray, np.ndarray]:
    if not reference_fps:
        return np.full(len(query_fps), np.nan), np.full(len(query_fps), -1, dtype=int)
    values, indices = [], []
    for fp in query_fps:
        similarities = np.asarray(DataStructs.BulkTanimotoSimilarity(fp, reference_fps))
        position = int(np.argmax(similarities))
        values.append(float(similarities[position]))
        indices.append(position)
    return np.asarray(values), np.asarray(indices, dtype=int)


def similarity_summary(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p10": float(np.quantile(values, 0.10)),
        "p90": float(np.quantile(values, 0.90)),
        "fraction_below_0_4": float(np.mean(values < 0.4)),
        "fraction_at_least_0_6": float(np.mean(values >= 0.6)),
        "fraction_at_least_0_8": float(np.mean(values >= 0.8)),
    }


def read_task2_data(path: Path, generator) -> dict[str, object]:
    result: dict[str, list] = {
        "train_fps": [],
        "train_positive_fps": [],
        "train_scaffolds": [],
        "test_fps": [],
        "test_descriptors": [],
        "test_scaffolds": [],
        "test_labels": [],
        "test_scores": [],
    }
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            mol = Chem.MolFromSmiles(row["smiles"])
            if mol is None:
                continue
            fp = generator.GetFingerprint(mol)
            scaffold = canonical_scaffold(mol)
            if row["split"] == "train":
                result["train_fps"].append(fp)
                result["train_scaffolds"].append(scaffold)
                if int(row["label"]) == 1:
                    result["train_positive_fps"].append(fp)
            else:
                result["test_fps"].append(fp)
                result["test_descriptors"].append(descriptor_vector(mol))
                result["test_scaffolds"].append(scaffold)
                result["test_labels"].append(int(row["label"]))
                result["test_scores"].append(float(row["finetuned_score"]))
    return result


def performance_subset(labels: np.ndarray, scores: np.ndarray, selected: np.ndarray) -> dict[str, object]:
    subset_labels, subset_scores = labels[selected], scores[selected]
    result: dict[str, object] = {
        "n": int(selected.sum()),
        "positives": int(subset_labels.sum()),
    }
    if len(np.unique(subset_labels)) == 2:
        result.update(
            {
                "auroc": float(roc_auc_score(subset_labels, subset_scores)),
                "average_precision": float(average_precision_score(subset_labels, subset_scores)),
            }
        )
    else:
        result.update({"auroc": None, "average_precision": None})
    return result


def main() -> int:
    args = parse_args()
    started = time.perf_counter()
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

    train_ids: list[str] = []
    train_smiles: list[str] = []
    train_labels: list[int] = []
    train_fps: list = []
    train_descriptors: list[np.ndarray] = []
    train_scaffolds: set[str] = set()
    train_canonical: set[str] = set()
    invalid_train = 0

    parquet = pq.ParquetFile(args.train)
    for batch in parquet.iter_batches(
        batch_size=8_192, columns=["SMILES", "RandomID", "LABEL"]
    ):
        frame = batch.to_pydict()
        for smiles, compound_id, label in zip(
            frame["SMILES"], frame["RandomID"], frame["LABEL"], strict=True
        ):
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                invalid_train += 1
                continue
            canonical = Chem.MolToSmiles(mol, canonical=True)
            scaffold = canonical_scaffold(mol)
            train_ids.append(str(compound_id))
            train_smiles.append(str(smiles))
            train_labels.append(int(label))
            train_fps.append(generator.GetFingerprint(mol))
            train_descriptors.append(descriptor_vector(mol))
            train_canonical.add(canonical)
            if scaffold:
                train_scaffolds.add(scaffold)

    test_rows: list[dict[str, str]] = []
    test_fps: list = []
    test_descriptors: list[np.ndarray] = []
    test_canonical: list[str] = []
    test_scaffolds: list[str] = []
    invalid_test = 0
    with args.test.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            mol = Chem.MolFromSmiles(row["SMILES"])
            if mol is None:
                invalid_test += 1
                continue
            test_rows.append(row)
            test_fps.append(generator.GetFingerprint(mol))
            test_descriptors.append(descriptor_vector(mol))
            test_canonical.append(Chem.MolToSmiles(mol, canonical=True))
            test_scaffolds.append(canonical_scaffold(mol))

    train_descriptor_array = np.vstack(train_descriptors)
    test_descriptor_array = np.vstack(test_descriptors)
    labels = np.asarray(train_labels, dtype=int)
    shift_rows = descriptor_shift_rows(train_descriptor_array, test_descriptor_array)

    nearest_all, nearest_all_idx = max_similarities(test_fps, train_fps)
    positive_indices = np.flatnonzero(labels == 1)
    positive_fps = [train_fps[int(index)] for index in positive_indices]
    nearest_positive, _ = max_similarities(test_fps, positive_fps)
    task2 = read_task2_data(args.task2_scores, generator)
    task2_fps = task2["train_fps"]
    task2_positive_fps = task2["train_positive_fps"]
    nearest_task2, _ = max_similarities(test_fps, task2_fps)
    nearest_task2_positive, _ = max_similarities(test_fps, task2_positive_fps)
    internal_fps = task2["test_fps"]
    internal_descriptors = np.vstack(task2["test_descriptors"])
    internal_nearest, _ = max_similarities(internal_fps, task2_fps)
    internal_nearest_positive, _ = max_similarities(internal_fps, task2_positive_fps)
    internal_labels = np.asarray(task2["test_labels"], dtype=int)
    internal_scores = np.asarray(task2["test_scores"], dtype=float)
    task2_train_scaffolds = set(filter(None, task2["train_scaffolds"]))
    internal_scaffold_seen = np.asarray(
        [bool(value) and value in task2_train_scaffolds for value in task2["test_scaffolds"]],
        dtype=bool,
    )

    zero_shot_scores: dict[str, float] = {}
    if args.zero_shot_test.exists():
        with args.zero_shot_test.open(newline="", encoding="utf-8") as handle:
            zero_shot_scores = {
                row["compound_id"]: float(row["logica_score"]) for row in csv.DictReader(handle)
            }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "descriptor_shift.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(shift_rows[0]))
        writer.writeheader()
        writer.writerows(shift_rows)

    per_test_fields = [
        "compound_id",
        "smiles",
        *DESCRIPTORS,
        "canonical_smiles_in_train",
        "scaffold_in_train",
        "nearest_full_train_similarity",
        "nearest_full_train_id",
        "nearest_full_train_smiles",
        "nearest_full_train_label",
        "nearest_any_training_positive_similarity",
        "nearest_task2_train_similarity",
        "nearest_task2_positive_similarity",
        "zero_shot_logica_score",
    ]
    with (args.output_dir / "test_domain_membership.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=per_test_fields)
        writer.writeheader()
        for i, row in enumerate(test_rows):
            neighbor = int(nearest_all_idx[i])
            output = {
                "compound_id": row["Ccompound_ID"],
                "smiles": row["SMILES"],
                **{
                    name: f"{test_descriptor_array[i, column]:.10g}"
                    for column, name in enumerate(DESCRIPTORS)
                },
                "canonical_smiles_in_train": int(test_canonical[i] in train_canonical),
                "scaffold_in_train": int(
                    bool(test_scaffolds[i]) and test_scaffolds[i] in train_scaffolds
                ),
                "nearest_full_train_similarity": f"{nearest_all[i]:.10g}",
                "nearest_full_train_id": train_ids[neighbor],
                "nearest_full_train_smiles": train_smiles[neighbor],
                "nearest_full_train_label": train_labels[neighbor],
                "nearest_any_training_positive_similarity": f"{nearest_positive[i]:.10g}",
                "nearest_task2_train_similarity": f"{nearest_task2[i]:.10g}",
                "nearest_task2_positive_similarity": f"{nearest_task2_positive[i]:.10g}",
                "zero_shot_logica_score": (
                    f"{zero_shot_scores[row['Ccompound_ID']]:.10g}"
                    if row["Ccompound_ID"] in zero_shot_scores
                    else ""
                ),
            }
            writer.writerow(output)

    scaffold_matches = np.asarray(
        [bool(value) and value in train_scaffolds for value in test_scaffolds], dtype=bool
    )
    exact_matches = np.asarray([value in train_canonical for value in test_canonical], dtype=bool)
    classifier = domain_classifier_auc(
        train_descriptor_array,
        train_fps,
        test_descriptor_array,
        test_fps,
        args.domain_classifier_train_sample,
        args.seed,
    )
    internal_external_classifier = domain_classifier_auc(
        internal_descriptors,
        internal_fps,
        test_descriptor_array,
        test_fps,
        len(internal_fps),
        args.seed + 1,
    )
    auc = float(classifier["mean_auroc"])
    median_task2 = float(np.median(nearest_task2))
    if auc >= 0.90 or median_task2 < 0.30:
        severity = "severe"
    elif auc >= 0.80 or median_task2 < 0.40:
        severity = "large"
    elif auc >= 0.70 or median_task2 < 0.50:
        severity = "moderate"
    else:
        severity = "small"

    report = {
        "comparison": "WDR91 full challenge training library versus provided challenge test compounds",
        "fingerprint": "RDKit Morgan radius 2, 2048 bits; similarity is Tanimoto",
        "counts": {
            "train_valid": len(train_fps),
            "train_invalid": invalid_train,
            "train_positive": int(labels.sum()),
            "test_valid": len(test_fps),
            "test_invalid": invalid_test,
        },
        "overall_shift_assessment": severity,
        "domain_classifier": classifier,
        "exact_structure_overlap": {
            "test_count": int(exact_matches.sum()),
            "test_fraction": float(exact_matches.mean()),
        },
        "bemis_murcko_scaffold_overlap": {
            "train_unique_nonempty_scaffolds": len(train_scaffolds),
            "test_unique_nonempty_scaffolds": len(set(filter(None, test_scaffolds))),
            "test_count_with_train_scaffold": int(scaffold_matches.sum()),
            "test_fraction_with_train_scaffold": float(scaffold_matches.mean()),
            "test_acyclic_count": int(sum(not value for value in test_scaffolds)),
        },
        "nearest_neighbor_similarity": {
            "to_full_training_library": similarity_summary(nearest_all),
            "to_any_of_138_training_positives": similarity_summary(nearest_positive),
            "to_task2_finetuning_train_split": similarity_summary(nearest_task2),
            "to_task2_finetuning_positives": similarity_summary(nearest_task2_positive),
        },
        "transfer_from_internal_holdout": {
            "source_classifier_internal_holdout_vs_challenge_test": internal_external_classifier,
            "descriptor_shift_internal_holdout_to_challenge_test": descriptor_shift_rows(
                internal_descriptors, test_descriptor_array
            ),
            "nearest_to_task2_finetuning_train": {
                "internal_holdout": similarity_summary(internal_nearest),
                "challenge_test": similarity_summary(nearest_task2),
                "challenge_fraction_below_internal_10th_percentile": float(
                    np.mean(nearest_task2 < np.quantile(internal_nearest, 0.10))
                ),
            },
            "nearest_to_task2_positive_train": {
                "internal_holdout": similarity_summary(internal_nearest_positive),
                "challenge_test": similarity_summary(nearest_task2_positive),
            },
            "internal_finetuned_performance_by_domain": {
                "all": performance_subset(
                    internal_labels, internal_scores, np.ones(len(internal_labels), dtype=bool)
                ),
                "similarity_below_0_4": performance_subset(
                    internal_labels, internal_scores, internal_nearest < 0.4
                ),
                "similarity_at_least_0_4": performance_subset(
                    internal_labels, internal_scores, internal_nearest >= 0.4
                ),
                "scaffold_seen_in_finetuning_train": performance_subset(
                    internal_labels, internal_scores, internal_scaffold_seen
                ),
                "scaffold_unseen_in_finetuning_train": performance_subset(
                    internal_labels, internal_scores, ~internal_scaffold_seen
                ),
            },
        },
        "descriptor_shift": shift_rows,
        "interpretation": [
            "A source-classifier AUROC near 0.5 means little detectable shift; values above 0.8 indicate large shift and above 0.9 severe shift.",
            "Low test-to-fine-tuning Tanimoto similarity indicates extrapolation beyond compounds that supplied supervised signal.",
            "No test labels are available, so this analysis diagnoses transfer risk but cannot measure test performance.",
            "The random 80/20 internal split shares the same source library and is expected to be optimistic when the external test set is shifted.",
        ],
        "elapsed_seconds": time.perf_counter() - started,
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
