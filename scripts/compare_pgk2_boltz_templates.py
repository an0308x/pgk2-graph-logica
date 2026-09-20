#!/usr/bin/env python3
"""Compare PGK2 Boltz poses generated from two protein templates.

This is an auditable *reranking diagnostic*, not a calibrated binding or
inhibition predictor.  A candidate is rewarded only when its predicted
active-site contacts agree between the hPGK2–compound-47 and 2PAA templates.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def integer_set(value: str) -> set[int]:
    return {int(item) for item in value.split(";") if item}


def jaccard(first: set[int], second: set[int]) -> float:
    union = first | second
    return len(first & second) / len(union) if union else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--compound47-results",
        type=Path,
        default=ROOT / "artifacts/pgk2_structural_pilot/boltz_pilot_results.csv",
    )
    parser.add_argument(
        "--2paa-results",
        dest="two_paa_results",
        type=Path,
        default=ROOT / "artifacts/pgk2_structural_pilot/boltz_2paa_results.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts/pgk2_structural_pilot/template_consensus.csv",
    )
    args = parser.parse_args()

    with args.compound47_results.open(newline="") as handle:
        first = {row["CatalogID"]: row for row in csv.DictReader(handle)}
    with args.two_paa_results.open(newline="") as handle:
        second = {row["CatalogID"]: row for row in csv.DictReader(handle)}
    shared = sorted(set(first) & set(second), key=lambda catalog: int(first[catalog]["pilot_rank"]))
    output_rows: list[dict[str, object]] = []
    for catalog in shared:
        cmp47, two_paa = first[catalog], second[catalog]
        if cmp47["status"] != "succeeded" or two_paa["status"] != "succeeded":
            continue
        cmp47_active = integer_set(cmp47["active_site_contacts_5A"])
        two_paa_active = integer_set(two_paa["active_site_contacts_5A"])
        cmp47_atp = integer_set(cmp47["atp_adp_contacts_5A"])
        two_paa_atp = integer_set(two_paa["atp_adp_contacts_5A"])
        confidence_47 = float(cmp47["binding_confidence"])
        confidence_2paa = float(two_paa["binding_confidence"])
        active_jaccard = jaccard(cmp47_active, two_paa_active)
        atp_jaccard = jaccard(cmp47_atp, two_paa_atp)
        min_confidence = min(confidence_47, confidence_2paa)
        output_rows.append(
            {
                "pilot_rank": cmp47["pilot_rank"],
                "CatalogID": catalog,
                "SMILES": cmp47["SMILES"],
                "series_id": cmp47["series_id"],
                "logica_score": cmp47["logica_score"],
                "compound47_binding_confidence": confidence_47,
                "2paa_binding_confidence": confidence_2paa,
                "mean_binding_confidence": (confidence_47 + confidence_2paa) / 2,
                "minimum_binding_confidence": min_confidence,
                "active_site_contact_jaccard_5A": active_jaccard,
                "atp_adp_contact_jaccard_5A": atp_jaccard,
                "compound47_pg3_contacts_5A": cmp47["pg3_contact_count_5A"],
                "2paa_pg3_contacts_5A": two_paa["pg3_contact_count_5A"],
                "template_consensus_score": min_confidence * active_jaccard,
                "interpretation": "Structural reranking diagnostic only; not a probability of kinase inhibition.",
            }
        )
    if not output_rows:
        raise RuntimeError("No jointly completed template pairs")
    output_rows.sort(key=lambda row: (-float(row["template_consensus_score"]), int(row["pilot_rank"])))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
