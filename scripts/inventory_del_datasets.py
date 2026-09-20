#!/usr/bin/env python3
"""Inventory local DEL Parquet datasets without modifying source files."""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.compute as pc
import pyarrow.parquet as pq
from rdkit import Chem, RDLogger

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts" / "data_inventory"
RDLogger.DisableLog("rdApp.*")


def dataset_class(path: Path) -> str:
    name = path.name
    if "DREAM_Challenge_1_TrainSet" in name:
        return "dream_del_training"
    if name.startswith("R2_"):
        return "round2_curated"
    if name.startswith("MLReadyPlusFPs_"):
        return "mlready_assay"
    return "assay_export"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def numeric_summary(table, name: str) -> dict[str, float | int | None]:
    array = table[name]
    return {
        "nulls": array.null_count,
        "min": pc.min(array).as_py(),
        "max": pc.max(array).as_py(),
        "mean": pc.mean(array).as_py(),
    }


def inspect_file(path: Path) -> dict[str, object]:
    parquet = pq.ParquetFile(path)
    schema = parquet.schema_arrow
    names = schema.names
    label_column = "LABEL" if "LABEL" in names else "BINARY_LABEL" if "BINARY_LABEL" in names else None
    enrichment_column = (
        "EASMS_ENRICHMENT"
        if "EASMS_ENRICHMENT" in names
        else "ENRICHMENT"
        if "ENRICHMENT" in names
        else None
    )
    wanted = [
        name
        for name in ("SMILES", "TARGET_ID", label_column, enrichment_column, "PVALUE")
        if name and name in names
    ]
    table = parquet.read(columns=wanted)
    smiles = table["SMILES"].to_pylist() if "SMILES" in wanted else []
    target_counts = Counter(table["TARGET_ID"].to_pylist()) if "TARGET_ID" in wanted else Counter()
    label_counts = Counter(table[label_column].to_pylist()) if label_column else Counter()
    sample = [value for value in smiles if value is not None][:100]
    sample_valid = sum(
        all(Chem.MolFromSmiles(part) is not None for part in str(value).split(";")) for value in sample
    )
    result: dict[str, object] = {
        "path": str(path.relative_to(ROOT)),
        "filename": path.name,
        "dataset_class": dataset_class(path),
        "size_bytes": path.stat().st_size,
        "rows": parquet.metadata.num_rows,
        "row_groups": parquet.metadata.num_row_groups,
        "columns": names,
        "schema": [(field.name, str(field.type)) for field in schema],
        "schema_signature": hashlib.sha256(
            json.dumps([(field.name, str(field.type)) for field in schema]).encode()
        ).hexdigest()[:12],
        "label_column": label_column,
        "label_counts": {str(key): value for key, value in sorted(label_counts.items(), key=lambda item: str(item[0]))},
        "target_counts": dict(target_counts),
        "smiles_nulls": table["SMILES"].null_count if "SMILES" in wanted else None,
        "unique_smiles": len(set(smiles)) if smiles else None,
        "sample_smiles_checked": len(sample),
        "sample_smiles_valid": sample_valid,
        "fingerprint_columns": [
            name
            for name in ("ECFP4", "ECFP6", "FCFP4", "FCFP6", "MACCS", "RDK", "AVALON", "TOPTOR", "ATOMPAIR")
            if name in names
        ],
        "fingerprint_storage": str(schema.field("ECFP4").type) if "ECFP4" in names else None,
    }
    if enrichment_column:
        result["enrichment_column"] = enrichment_column
        result["enrichment_summary"] = numeric_summary(table, enrichment_column)
    if "PVALUE" in wanted:
        result["pvalue_summary"] = numeric_summary(table, "PVALUE")
    return result


def write_summary(report: dict[str, object]) -> None:
    totals = report["totals"]
    class_counts = report["dataset_classes"]
    schema_groups = report["schema_groups"]
    duplicate_groups = report["exact_duplicate_groups"]
    wdr91 = report["wdr91"]
    lines = [
        "# DEL dataset inventory",
        "",
        "## Inventory result",
        "",
        f"- Valid Parquet files: **{totals['valid_parquet_files']} / {totals['parquet_files']}**",
        f"- Total rows across files: **{totals['rows']:,}**",
        f"- Total storage: **{totals['size_gib']:.2f} GiB**",
        f"- Distinct schema signatures: **{len(schema_groups)}**",
        f"- Exact duplicate groups: **{len(duplicate_groups)}**",
        "",
        "## Dataset families",
        "",
        "| Family | Files | Rows |",
        "|---|---:|---:|",
    ]
    for name, values in sorted(class_counts.items()):
        lines.append(f"| {name} | {values['files']} | {values['rows']:,} |")
    lines += [
        "",
        "## WDR91 datasets",
        "",
        "| Dataset | Rows | Positives | Role |",
        "|---|---:|---:|---|",
    ]
    for item in wdr91["datasets"]:
        lines.append(
            f"| `{item['filename']}` | {item['rows']:,} | {item['positives']:,} | {item['role']} |"
        )
    lines += [
        "",
        f"The ML-ready WDR91 assay contains **{wdr91['mlready_challenge_test_overlap']}** exact challenge-test structures; "
        f"all **{wdr91['mlready_positive_challenge_test_overlap']}** overlapping labeled structures are positives.",
        "",
        "## Initial modeling interpretation",
        "",
        "- The original DREAM DEL file remains the large target-specific training source.",
        "- The ML-ready WDR91 and Round 2 WDR91 files provide assay-style external outcomes and should be reserved for validation before being considered for training.",
        "- The other targets are suitable candidates for multi-target or sequential LogiCA adaptation after protein sequences and assay provenance are resolved.",
        "- Fingerprints are stored as strings in the assay exports but as integer lists in the DREAM file, so direct concatenation requires normalization.",
        "- Exact duplicate files should be removed from any aggregate training manifest to avoid double weighting.",
        "",
    ]
    (OUTPUT / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    started = time.perf_counter()
    paths = sorted(ROOT.glob("drive-download-*/*.parquet"))
    dream = ROOT / "DREAM_challenge" / "DREAM_Challenge_1_TrainSet.parquet"
    if dream.exists():
        paths.append(dream)

    records, errors = [], []
    for index, path in enumerate(paths, 1):
        try:
            records.append(inspect_file(path))
        except Exception as exc:  # retain failures in the audit report
            errors.append({"path": str(path.relative_to(ROOT)), "error": repr(exc)})
        if index % 20 == 0:
            print(f"inspected {index}/{len(paths)}")

    size_groups: dict[int, list[Path]] = defaultdict(list)
    for path in paths:
        size_groups[path.stat().st_size].append(path)
    hash_groups: dict[str, list[str]] = defaultdict(list)
    for candidates in size_groups.values():
        if len(candidates) < 2:
            continue
        for path in candidates:
            hash_groups[sha256(path)].append(str(path.relative_to(ROOT)))
    duplicate_groups = [values for values in hash_groups.values() if len(values) > 1]

    class_counts: dict[str, dict[str, int]] = defaultdict(lambda: {"files": 0, "rows": 0})
    schema_groups: dict[str, dict[str, object]] = {}
    for record in records:
        family = record["dataset_class"]
        class_counts[family]["files"] += 1
        class_counts[family]["rows"] += int(record["rows"])
        signature = record["schema_signature"]
        group = schema_groups.setdefault(
            signature,
            {"files": 0, "rows": 0, "columns": record["columns"], "example": record["path"]},
        )
        group["files"] += 1
        group["rows"] += int(record["rows"])

    by_filename = {record["filename"]: record for record in records}
    wdr91_specs = [
        ("DREAM_Challenge_1_TrainSet.parquet", "DEL training labels"),
        ("MLReadyPlusFPs_WDR91_A4D1P6_392_747_AsmBatchNumber2.parquet", "independent assay validation"),
        ("R2_WDR91_curated.parquet", "curated Round 2 validation"),
    ]
    wdr91_datasets = []
    for filename, role in wdr91_specs:
        record = by_filename[filename]
        positives = int(record["label_counts"].get("1", 0))
        wdr91_datasets.append(
            {"filename": filename, "rows": record["rows"], "positives": positives, "role": role}
        )

    mlready_path = next(path for path in paths if path.name == wdr91_specs[1][0])
    mlready = pq.read_table(mlready_path, columns=["SMILES", "LABEL"]).to_pydict()
    with (ROOT / "DREAM_challenge" / "DREAM_Target2035_Challenge_test_data.csv").open(
        encoding="utf-8"
    ) as handle:
        import csv

        challenge_rows = list(csv.DictReader(handle))
    challenge_smiles = {row["SMILES"] for row in challenge_rows}
    mlready_smiles = set(mlready["SMILES"])
    mlready_positive = {
        smiles for smiles, label in zip(mlready["SMILES"], mlready["LABEL"], strict=True) if label == 1
    }

    report = {
        "totals": {
            "parquet_files": len(paths),
            "valid_parquet_files": len(records),
            "invalid_parquet_files": len(errors),
            "rows": sum(int(record["rows"]) for record in records),
            "size_bytes": sum(path.stat().st_size for path in paths),
            "size_gib": sum(path.stat().st_size for path in paths) / 2**30,
        },
        "dataset_classes": dict(class_counts),
        "schema_groups": schema_groups,
        "exact_duplicate_groups": duplicate_groups,
        "wdr91": {
            "datasets": wdr91_datasets,
            "mlready_challenge_test_overlap": len(mlready_smiles & challenge_smiles),
            "mlready_positive_challenge_test_overlap": len(mlready_positive & challenge_smiles),
        },
        "files": records,
        "errors": errors,
        "elapsed_seconds": time.perf_counter() - started,
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "inventory.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    write_summary(report)
    print(json.dumps({key: report[key] for key in ("totals", "dataset_classes", "exact_duplicate_groups", "wdr91", "elapsed_seconds")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
