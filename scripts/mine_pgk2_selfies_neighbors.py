#!/usr/bin/env python3
"""Create control-confirmed PGK2 ranking pairs from SELFormer neighborhoods."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem


ROOT = Path(__file__).resolve().parents[1]


def dominates(positive: dict[str, object], negative: dict[str, object]) -> bool:
    """Use only directional, observed DEL counter-evidence for a preference."""
    return (
        int(positive["count_PGK2"]) > int(negative["count_PGK2"])
        and int(positive["count_PGK2_with_inhibitor"]) <= int(negative["count_PGK2_with_inhibitor"])
        and int(positive["count_NTC_evidence"]) <= int(negative["count_NTC_evidence"])
        and (
            int(negative["count_PGK2_with_inhibitor"]) > 0
            or int(negative["count_NTC_evidence"]) > 0
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--embeddings", type=Path,
        default=ROOT / "artifacts/pgk2_selfies_embeddings/embeddings.npy",
    )
    parser.add_argument(
        "--metadata", type=Path,
        default=ROOT / "artifacts/pgk2_selfies_embeddings/metadata.parquet",
    )
    parser.add_argument("--minimum-cosine", type=float, default=0.90)
    parser.add_argument(
        "--minimum-morgan",
        type=float,
        default=0.60,
        help="Independent local-chemistry gate after SELFormer neighbor retrieval.",
    )
    parser.add_argument("--neighbors-per-anchor", type=int, default=32)
    parser.add_argument("--pairs-per-anchor", type=int, default=2)
    parser.add_argument("--allow-cross-library", action="store_true")
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "artifacts/pgk2_selfies_neighbor_pairs"
    )
    args = parser.parse_args()
    if (
        not 0 <= args.minimum_cosine <= 1
        or not 0 <= args.minimum_morgan <= 1
        or args.neighbors_per_anchor < 1
        or args.pairs_per_anchor < 1
    ):
        raise ValueError("invalid similarity or neighbor count")
    metadata = pl.read_parquet(args.metadata)
    embeddings = np.load(args.embeddings, mmap_mode="r")
    if metadata.height != embeddings.shape[0]:
        raise ValueError("metadata and embedding counts differ")
    if not np.allclose(np.linalg.norm(embeddings[: min(1000, len(embeddings))], axis=1), 1, atol=1e-4):
        raise ValueError("embeddings must be L2-normalized")
    rows = metadata.to_dicts()
    fingerprint_generator = AllChem.GetMorganGenerator(radius=2, fpSize=2048)
    fingerprints: dict[str, object] = {}

    def fingerprint(smiles: str):
        if smiles not in fingerprints:
            molecule = Chem.MolFromSmiles(smiles)
            if molecule is None:
                raise ValueError(f"Invalid SMILES: {smiles}")
            fingerprints[smiles] = fingerprint_generator.GetFingerprint(molecule)
        return fingerprints[smiles]
    anchors = np.asarray([index for index, row in enumerate(rows) if row["pool_role"] == "anchor"])
    comparators = np.asarray(
        [index for index, row in enumerate(rows) if row["pool_role"] == "counter_evidence_comparator"]
    )
    pairs: list[dict[str, object]] = []
    anchors_with_pair = 0
    groups = [None] if args.allow_cross_library else sorted({str(rows[index]["library"]) for index in anchors})
    for group in groups:
        anchor_ids = anchors if group is None else np.asarray([index for index in anchors if rows[index]["library"] == group])
        comparator_ids = comparators if group is None else np.asarray([index for index in comparators if rows[index]["library"] == group])
        if not len(anchor_ids) or not len(comparator_ids):
            continue
        for offset in range(0, len(anchor_ids), 128):
            query_ids = anchor_ids[offset : offset + 128]
            similarities = np.asarray(embeddings[query_ids] @ embeddings[comparator_ids].T)
            k = min(args.neighbors_per_anchor, len(comparator_ids))
            nearest = np.argpartition(similarities, -k, axis=1)[:, -k:]
            for local_index, positive_index in enumerate(query_ids):
                positive = rows[int(positive_index)]
                ranked = sorted(
                    ((float(similarities[local_index, location]), int(comparator_ids[location])) for location in nearest[local_index]),
                    reverse=True,
                )
                selected = 0
                for cosine, negative_index in ranked:
                    negative = rows[negative_index]
                    morgan = float(
                        DataStructs.TanimotoSimilarity(
                            fingerprint(str(positive["SMILES"])), fingerprint(str(negative["SMILES"]))
                        )
                    )
                    if (
                        cosine < args.minimum_cosine
                        or morgan < args.minimum_morgan
                        or not dominates(positive, negative)
                    ):
                        continue
                    pairs.append(
                        {
                            "positive_smiles": positive["SMILES"],
                            "negative_smiles": negative["SMILES"],
                            "pair_weight": 1.0 + cosine,
                            "pair_type": "selfies_selformer_local_control_confirmed",
                            "same_library": positive["library"] == negative["library"],
                            "selformer_cosine_similarity": cosine,
                            "morgan_tanimoto_similarity": morgan,
                            "positive_count_PGK2": positive["count_PGK2"],
                            "negative_count_PGK2": negative["count_PGK2"],
                            "positive_count_inhibitor": positive["count_PGK2_with_inhibitor"],
                            "negative_count_inhibitor": negative["count_PGK2_with_inhibitor"],
                            "positive_count_NTC_evidence": positive["count_NTC_evidence"],
                            "negative_count_NTC_evidence": negative["count_NTC_evidence"],
                        }
                    )
                    selected += 1
                    if selected == 1:
                        anchors_with_pair += 1
                    if selected == args.pairs_per_anchor:
                        break
    if not pairs:
        raise RuntimeError("No control-confirmed pairs passed the requested SELFormer similarity threshold")
    frame = pl.DataFrame(pairs)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "pgk2_selfies_neighbor_pairs.parquet"
    frame.write_parquet(output, compression="zstd")
    report = {
        "policy": {
            "embedding": "frozen SELFormer mean-pooled SELFIES token embedding",
            "minimum_cosine": args.minimum_cosine,
            "minimum_morgan": args.minimum_morgan,
            "same_library_required": not args.allow_cross_library,
            "negative_evidence": "observed inhibitor or NTC signal; no control zero is treated as positive evidence",
        },
        "anchors": len(anchors),
        "comparators": len(comparators),
        "anchors_with_pair": anchors_with_pair,
        "pairs": frame.height,
        "cosine": {
            "q10": float(frame["selformer_cosine_similarity"].quantile(0.1)),
            "median": float(frame["selformer_cosine_similarity"].median()),
            "q90": float(frame["selformer_cosine_similarity"].quantile(0.9)),
        },
        "morgan": {
            "q10": float(frame["morgan_tanimoto_similarity"].quantile(0.1)),
            "median": float(frame["morgan_tanimoto_similarity"].median()),
            "q90": float(frame["morgan_tanimoto_similarity"].quantile(0.9)),
        },
        "output": str(output),
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
