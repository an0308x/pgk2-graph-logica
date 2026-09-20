#!/usr/bin/env python3
"""Convert a ranked PGK2 validation CSV to the required 50-ID text file."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ranked-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.ranked_csv.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or "CatalogID" not in rows[0]:
        raise ValueError("Input must be a CSV with a CatalogID column")
    selected = [str(row["CatalogID"]).strip() for row in rows[:50]]
    if len(selected) != 50 or any(not value for value in selected) or len(set(selected)) != 50:
        raise ValueError("Input must provide 50 unique, non-empty CatalogIDs")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\r\n".join(selected) + "\r\n", newline="")
    print({"input": str(args.ranked_csv), "output": str(args.output), "catalog_ids": len(selected)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
