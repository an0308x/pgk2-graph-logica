#!/usr/bin/env python3
"""Summarize a complete, matched six-run Graph LogiCA repair diagnostic."""
import argparse
import json
from pathlib import Path
import statistics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    reports = {}
    for mode in ("on", "off"):
        for seed in (2026, 2027, 2028):
            path = args.run_dir / f"graph_{mode}_seed{seed}" / "report.json"
            report = json.loads(path.read_text())
            if report["graph_conditioning"] != mode or report["ligand_mode"] != "2d":
                raise ValueError(f"Unexpected model condition: {path}")
            if not report["graphs"]["base_attention_interface_restricted"]:
                raise ValueError(f"Unrestricted attention: {path}")
            if report["holdout"]["pairs"] != report["pairs"]["heldout"]:
                raise ValueError(f"Inconsistent holdout size: {path}")
            history = report["training"]["history"]
            # The trainer retains the first epoch in a tie.
            best = max(history, key=lambda epoch: epoch["dev"]["weighted_win_rate"])
            if report["selected_epoch"] != best["epoch"]:
                raise ValueError(f"Checkpoint was not selected on development: {path}")
            reports[(mode, seed)] = report
    values = list(reports.values())
    if len({r["split_manifest_sha256"] for r in values}) != 1:
        raise ValueError("Runs do not share a split")
    if len({r["protocol"] for r in values}) != 1:
        raise ValueError("Runs do not share a validation protocol")
    if any(r["smoke_only"] or r["run_purpose"] != "diagnostic" for r in values):
        raise ValueError("Unexpected run scope")
    if len({json.dumps(r["pairs"], sort_keys=True) for r in values}) != 1:
        raise ValueError("Runs evaluated different data")
    result = {"protocol": values[0]["protocol"],
              "split_manifest_sha256": values[0]["split_manifest_sha256"],
              "pairs": values[0]["pairs"], "conditions": {},
              "limitation": "Seeds share a holdout; SD is seed variation, not a sampling confidence interval. DEL pair ranking is not kinase validation."}
    for mode in ("on", "off"):
        group = [reports[(mode, seed)] for seed in (2026, 2027, 2028)]
        result["conditions"][mode] = {
            metric: {"mean": statistics.mean(r["holdout"][metric] for r in group),
                     "seed_sd": statistics.stdev(r["holdout"][metric] for r in group),
                     "values": [r["holdout"][metric] for r in group]}
            for metric in ("win_rate", "weighted_win_rate")}
        result["conditions"][mode]["selected_epochs"] = [r["selected_epoch"] for r in group]
    result["graph_minus_control_weighted_win_rate"] = [
        reports[("on", seed)]["holdout"]["weighted_win_rate"] - reports[("off", seed)]["holdout"]["weighted_win_rate"]
        for seed in (2026, 2027, 2028)]
    path = args.run_dir / "comparison_report.json"
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
