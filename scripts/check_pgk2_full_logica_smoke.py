#!/usr/bin/env python3
"""Check execution readiness, not predictive performance, before a full run."""
import argparse
import json
import math
from pathlib import Path

from run_pgk2_full_logica import verify_manifest
from logica_binding.streaming_graph_logica import FULL_PROTOCOL


def check(report, manifest):
    if report["protocol"] != FULL_PROTOCOL or not report["smoke_only"]:
        raise ValueError("Expected a v3 smoke report")
    if report["identity"]["manifest"] != manifest["manifest_sha256"]:
        raise ValueError("Smoke used different data")
    config = report["identity"]["config"]
    if config["device"] != "cuda" or config["batch_size"] != 64 or config["rank_batch_size"] != 32 or config["rank_every"] != 4:
        raise ValueError("Smoke did not test the intended GPU configuration")
    if report["counts"]["optimizer_steps"] < 64 or report["counts"]["train_molecule_visits"] < 4096:
        raise ValueError("Insufficient bounded execution test")
    for phase in (report["training_ranking"], report["history"][-1]["dev"], report["holdout"]):
        competition = phase["channels"]["competition"]
        if not competition["pairs"] or competition["weight_sum"] <= 0:
            raise ValueError("No supported competition comparisons")
        if phase["selection_loss"] is None or not math.isfinite(phase["selection_loss"]):
            raise ValueError("Invalid competition evaluation")
    speed = report["train_molecules_per_second"]
    if not math.isfinite(speed) or speed <= 0 or report["peak_gpu_memory_bytes"] <= 0:
        raise ValueError("Missing valid GPU throughput/memory measurements")
    estimated = manifest["splits"]["train"]["molecules"] / speed
    wall_minutes = math.ceil((estimated * 1.5 + 3600) / 3600) * 60
    return {"execution_ready": True, "predictive_performance_validated": False,
        "train_molecules_per_second": speed,
        "peak_gpu_memory_gib": report["peak_gpu_memory_bytes"] / 2**30,
        "estimated_training_hours": estimated / 3600,
        "recommended_walltime_minutes": wall_minutes,
        "note": "Estimate extrapolates a short run; allowance includes shard I/O and evaluation. No hit-rate claim."}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--smoke-report", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    args = p.parse_args()
    print(json.dumps(check(json.loads(args.smoke_report.read_text()), verify_manifest(args.manifest)), indent=2))


if __name__ == "__main__":
    main()
