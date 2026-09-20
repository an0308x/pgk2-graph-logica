#!/usr/bin/env python3
"""Create UPGMA/MCS-informed PGK2 top-50 portfolios from a ranked candidate pool.

This is a *Kruger-inspired* implementation of the approach in Kruger, Fechner
and Stiefl (J. Chem. Inf. Model. 2020, DOI 10.1021/acs.jcim.0c00204): UPGMA
clustering over molecular similarity followed by maximum-common-substructure
(MCS) definitions for audit.  The authors' exact supporting-code parameters
are not bundled with the challenge, so this tool does not claim to reproduce
the organizer's hidden cluster labels.

It creates score-ranked, series-capped portfolios to measure the retrieval /
chemical-series trade-off on the blind validation queue.  It never uses test
labels or validation labels.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, rdFMCS
from scipy.cluster.hierarchy import fcluster, linkage

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ranked-csv",
        type=Path,
        default=ROOT / "artifacts" / "pgk2_matched_sar_validation_scores" / "merged" / "validation_ranked_scores.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts" / "pgk2_series_portfolios")
    parser.add_argument("--pool-size", type=int, default=1000, help="Top-ranked molecules to cluster.")
    parser.add_argument("--minimum-similarity", type=float, default=0.50, help="UPGMA cut at 1 - this value.")
    parser.add_argument("--mcs-timeout-seconds", type=int, default=5)
    parser.add_argument("--caps", type=int, nargs="+", default=[1, 3, 5], help="Maximum selected molecules per inferred series.")
    return parser.parse_args()


def fingerprint(smiles: str):
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"Invalid SMILES: {smiles!r}")
    return AllChem.GetMorganGenerator(radius=2, fpSize=2048).GetFingerprint(molecule)


def condensed_tanimoto_distances(fingerprints: list) -> np.ndarray:
    size = len(fingerprints)
    values = np.empty(size * (size - 1) // 2, dtype=np.float64)
    offset = 0
    for index, fp in enumerate(fingerprints[:-1]):
        similarities = DataStructs.BulkTanimotoSimilarity(fp, fingerprints[index + 1 :])
        count = len(similarities)
        values[offset : offset + count] = 1.0 - np.asarray(similarities, dtype=np.float64)
        offset += count
    return values


def mcs_smarts(rows: list[dict[str, str]], timeout: int) -> str:
    if len(rows) == 1:
        return Chem.MolToSmarts(Chem.MolFromSmiles(rows[0]["SMILES"]))
    molecules = [Chem.MolFromSmiles(row["SMILES"]) for row in rows]
    result = rdFMCS.FindMCS(
        molecules,
        timeout=timeout,
        ringMatchesRingOnly=True,
        completeRingsOnly=True,
        atomCompare=rdFMCS.AtomCompare.CompareElements,
        bondCompare=rdFMCS.BondCompare.CompareOrderExact,
    )
    return result.smartsString if result.smartsString else ""


def select_portfolio(clusters: dict[int, list[dict[str, str]]], cap: int) -> list[dict[str, str]]:
    """Score-first greedy selection, with an explicit maximum per series."""

    if cap <= 0:
        raise ValueError("All series caps must be positive")
    ranked = sorted(
        (row for members in clusters.values() for row in members),
        key=lambda row: -float(row["logica_score"]),
    )
    selected: list[dict[str, str]] = []
    counts: dict[int, int] = defaultdict(int)
    for row in ranked:
        cluster = int(row["series_id"])
        if counts[cluster] >= cap:
            continue
        selected.append(row)
        counts[cluster] += 1
        if len(selected) == 50:
            break
    if len(selected) != 50:
        raise ValueError(f"Only {len(selected)} candidates selected with series cap {cap}; increase pool size")
    return selected


def write_submission(path: Path, rows: list[dict[str, str]]) -> None:
    path.write_text("\r\n".join(row["CatalogID"] for row in rows) + "\r\n", newline="")


def main() -> int:
    args = parse_args()
    if not args.ranked_csv.exists():
        raise FileNotFoundError(args.ranked_csv)
    if args.pool_size < 50 or not 0 < args.minimum_similarity <= 1 or args.mcs_timeout_seconds <= 0:
        raise ValueError("invalid pool size, similarity, or MCS timeout")
    started = time.perf_counter()
    with args.ranked_csv.open(newline="") as handle:
        rows = list(csv.DictReader(handle))[: args.pool_size]
    required = {"CatalogID", "SMILES", "logica_score"}
    if len(rows) != args.pool_size or not required.issubset(rows[0]):
        raise ValueError("Ranked input is shorter than pool-size or missing required columns")
    fingerprints = [fingerprint(row["SMILES"]) for row in rows]
    hierarchy = linkage(condensed_tanimoto_distances(fingerprints), method="average", optimal_ordering=True)
    labels = fcluster(hierarchy, t=1.0 - args.minimum_similarity, criterion="distance")
    clusters: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row, label in zip(rows, labels, strict=True):
        augmented = dict(row)
        augmented["series_id"] = str(int(label))
        clusters[int(label)].append(augmented)
    for members in clusters.values():
        members.sort(key=lambda row: -float(row["logica_score"]))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    series_rows: list[dict[str, object]] = []
    for identifier, members in sorted(clusters.items()):
        series_rows.append(
            {
                "series_id": identifier,
                "pool_members": len(members),
                "best_catalog_id": members[0]["CatalogID"],
                "best_score": float(members[0]["logica_score"]),
                "mcs_smarts": mcs_smarts(members, args.mcs_timeout_seconds),
            }
        )
    with (args.output_dir / "series_definitions.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(series_rows[0]))
        writer.writeheader()
        writer.writerows(series_rows)
    with (args.output_dir / "clustered_pool.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["CatalogID", "SMILES", "logica_score", "series_id"])
        writer.writeheader()
        for members in clusters.values():
            writer.writerows(members)
    portfolios: dict[str, dict[str, object]] = {}
    for cap in sorted(set(args.caps)):
        selected = select_portfolio(clusters, cap)
        name = f"validation_series_cap{cap}"
        with (args.output_dir / f"{name}_ranked.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["CatalogID", "SMILES", "logica_score", "series_id"])
            writer.writeheader()
            writer.writerows(selected)
        write_submission(args.output_dir / f"{name}_submission.txt", selected)
        portfolios[name] = {
            "series_cap": cap,
            "selected": len(selected),
            "distinct_inferred_series": len({row["series_id"] for row in selected}),
            "top_catalog_id": selected[0]["CatalogID"],
        }
    report = {
        "method": "Kruger-inspired UPGMA (average linkage) over Morgan radius-2/2048-bit Tanimoto distance; MCS audit definitions",
        "not_exact_organizer_reimplementation": True,
        "pool_size": args.pool_size,
        "minimum_similarity": args.minimum_similarity,
        "inferred_series_in_pool": len(clusters),
        "portfolios": portfolios,
        "elapsed_seconds": time.perf_counter() - started,
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
