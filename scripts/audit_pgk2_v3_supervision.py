#!/usr/bin/env python3
"""Audit train/dev coverage and exclusions without changing labels or touching holdout metrics."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
import polars as pl
from run_pgk2_full_logica import verify_manifest
from prepare_pgk2_full_logica_manifest import sha256


def audit_counts(frame):
    n = np.maximum(frame["source_rows"].to_numpy(), 1)
    t = frame["count_PGK2"].to_numpy() / n
    i = frame["count_PGK2_with_inhibitor"].to_numpy() / n
    sn = frame["count_NTC_selection"].to_numpy() / n
    nn = frame["count_NTC_supplement"].to_numpy() / np.maximum(frame["ntc_supplement_rows"].to_numpy(), 1)
    h = frame["historic_hits"].to_numpy()
    included = frame["confidence_0"].to_numpy() > 0
    any_channel = np.any(frame.select([f"confidence_{k}" for k in range(3)]).to_numpy() > 0, axis=1)
    result = {"molecules": len(t), "competition_supervised": int(included.sum()),
        "any_supervised_channel": int(any_channel.sum()), "zero_competitor": int((i == 0).sum()),
        "target_mean_at_most_one": int((t <= 1).sum()), "duplicate_source_molecules": int((n > 1).sum()),
        "zero_competitor_receiving_competition_weight": int(((i == 0) & included).sum())}
    for threshold in (2, 5, 10, 20, 50, 100):
        support = t >= threshold
        result[f"target_ge_{threshold}"] = int(support.sum())
        result[f"target_ge_{threshold}_zero_competitor"] = int((support & (i == 0)).sum())
        result[f"target_ge_{threshold}_zero_competitor_zero_ntc_no_history"] = int(
            (support & (i == 0) & (sn == 0) & (nn == 0) & (h == 0)).sum())
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    manifest = verify_manifest(args.manifest)
    totals = {split: defaultdict(int) for split in ("train", "dev")}
    examples = []
    for entry in manifest["shards"]:
        path = args.manifest.parent / entry["metadata"]
        if sha256(path) != entry["metadata_sha256"]:
            raise ValueError("Changed metadata")
        frame = pl.read_parquet(path)
        for split in totals:
            rows = frame.filter(pl.col("split") == split)
            for key, value in audit_counts(rows).items():
                totals[split][key] += value
            if split == "train":
                # Descriptive examples, not new positive labels or candidates.
                examples.extend(rows.filter((pl.col("count_PGK2_with_inhibitor") == 0)
                    & (pl.col("count_PGK2") / pl.col("source_rows") >= 20))
                    .with_columns((pl.col("count_PGK2") / pl.col("source_rows")).alias("mean_target"))
                    .sort(["mean_target", "molecule_id"], descending=[True, False]).head(100).to_dicts())
    top = sorted(examples, key=lambda row: (-row["mean_target"], row["molecule_id"]))[:100]
    if top:
        pl.DataFrame(top).write_parquet(args.output_dir / "excluded_high_target_train_examples.parquet")
    for split, values in totals.items():
        if values["molecules"] != manifest["splits"][split]["molecules"]:
            raise ValueError("Audit coverage differs from the manifest")
    report = {"manifest_sha256": manifest["manifest_sha256"], "splits": {s:dict(v) for s,v in totals.items()},
        "holdout_evaluated": False, "labels_changed": False,
        "interpretation": [
            "Numeric zero competitor counts are excluded from v3 competition loss by design, not established non-binders.",
            "Target-enriched zero-competitor rows may warrant a count/depth-aware sensitivity analysis; they are not proven inhibitors.",
            "Zero NTC counts are absence of observed reads, not proof of no background binding.",
            "Threshold counts use mean reads per source row; duplicate records are not independent replicates."]}
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
