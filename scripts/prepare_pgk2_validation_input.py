#!/usr/bin/env python3
"""Read the official validation CSV without changing it; emit an audited JSON input."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import zipfile


def read_validation(path):
    member = "Val-Test-set/PGK2_Validation_split.csv"
    with zipfile.ZipFile(path) as z:
        data = z.read(member)  # Never read the test split.
    reader = csv.DictReader(io.StringIO(data.decode("utf-8-sig")))
    if reader.fieldnames != ["CatalogID", "SMILES"]:
        raise ValueError("Unexpected validation columns")
    rows = list(reader)
    ids = [r["CatalogID"] for r in rows]
    if len(rows) != 244328 or len(set(ids)) != len(rows):
        raise ValueError("Incomplete or duplicate validation IDs")
    if any(not r["SMILES"].strip() or not r["CatalogID"].strip() or r["CatalogID"] != r["CatalogID"].strip() for r in rows):
        raise ValueError("Blank or malformed validation row")
    return {"panel":"validation", "member":member, "rows":len(rows),
        "zip_sha256":hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        "member_sha256":hashlib.sha256(data).hexdigest(), "candidates":rows}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--zip", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    payload = read_validation(a.zip)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open("x") as f:
        json.dump(payload,f)
    print(json.dumps({k:v for k,v in payload.items() if k!="candidates"}))
