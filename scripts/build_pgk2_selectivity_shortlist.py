#!/usr/bin/env python3
"""Build an auditable PGK2 structural-screening shortlist by late rank fusion.

No challenge labels or portal feedback are consumed.  Missing optional sources
are disabled globally; missing compounds within a supplied sparse source receive
zero rank contribution from that source.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import zipfile
from pathlib import Path

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold

ROOT = Path(__file__).resolve().parents[1]
QUINAZOLINE = Chem.MolFromSmarts("c1ccc2ncncc2c1")
FP_GENERATOR = AllChem.GetMorganGenerator(radius=2, fpSize=2048)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", choices=("validation", "test"), default="validation")
    parser.add_argument(
        "--candidate-panels",
        type=Path,
        default=ROOT / "DREAM_challenge_2026" / "Val-Test-set.zip",
    )
    parser.add_argument("--dual-logica-scores", type=Path, default=None)
    parser.add_argument(
        "--competition-scores",
        type=Path,
        default=None,
        help="Optional PGK2 active-site competition-refined scores with CatalogID and logica_score.",
    )
    parser.add_argument(
        "--matched-sar-scores",
        type=Path,
        default=ROOT / "artifacts" / "pgk2_series_portfolios" / "clustered_pool.csv",
    )
    parser.add_argument(
        "--fingerprint-scores",
        type=Path,
        default=ROOT / "artifacts" / "pgk2_fingerprint_baseline" / "validation_ranked_scores.csv",
    )
    parser.add_argument(
        "--reference-ligands",
        type=Path,
        default=None,
        help="Verified CSV with name,SMILES,reference_weight; screenshot transcriptions are not accepted.",
    )
    parser.add_argument(
        "--structural-scores",
        type=Path,
        default=None,
        help="Optional CSV with CatalogID and structural_selectivity_score.",
    )
    parser.add_argument("--matched-sar-weight", type=float, default=1.0)
    parser.add_argument("--competition-weight", type=float, default=1.0)
    parser.add_argument("--pgk2-logica-weight", type=float, default=0.25)
    parser.add_argument("--logica-delta-weight", type=float, default=1.0)
    parser.add_argument("--fingerprint-weight", type=float, default=0.0)
    parser.add_argument("--reference-ligand-weight", type=float, default=1.0)
    parser.add_argument("--structural-weight", type=float, default=1.0)
    parser.add_argument("--shortlist-size", type=int, default=1000)
    parser.add_argument("--max-per-scaffold", type=int, default=5)
    parser.add_argument("--max-heavy-atoms", type=int, default=49)
    parser.add_argument("--allow-quinazolines", action="store_true")
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "artifacts" / "pgk2_selectivity_shortlist"
    )
    return parser.parse_args()


def read_panel(path: Path, panel: str) -> list[dict[str, str]]:
    member = f"Val-Test-set/PGK2_{panel.capitalize()}_split.csv"
    with zipfile.ZipFile(path) as archive, archive.open(member) as handle:
        rows = list(csv.DictReader(line.decode("utf-8") for line in handle))
    if not rows or set(rows[0]) != {"CatalogID", "SMILES"}:
        raise ValueError(f"Unexpected columns in {member}")
    if len({row["CatalogID"] for row in rows}) != len(rows):
        raise ValueError(f"Duplicate CatalogIDs in {member}")
    return rows


def read_score_map(path: Path | None, score_column: str) -> dict[str, float]:
    if path is None:
        return {}
    if not path.exists():
        return {}
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or not {"CatalogID", score_column}.issubset(rows[0]):
        raise ValueError(f"{path} must contain CatalogID and {score_column}")
    result: dict[str, float] = {}
    for row in rows:
        catalog_id = row["CatalogID"].strip()
        if catalog_id in result:
            raise ValueError(f"Duplicate CatalogID {catalog_id} in {path}")
        value = float(row[score_column])
        if not math.isfinite(value):
            raise ValueError(f"Non-finite {score_column} for {catalog_id}")
        result[catalog_id] = value
    return result


def percentile_map(scores: dict[str, float]) -> dict[str, float]:
    """Map higher scores to (0, 1], preserving ties and reserving zero for missing."""
    if not scores:
        return {}
    unique = sorted(set(scores.values()))
    rank = {value: (index + 1) / len(unique) for index, value in enumerate(unique)}
    return {key: rank[value] for key, value in scores.items()}


def fingerprint(molecule: Chem.Mol):
    return FP_GENERATOR.GetFingerprint(molecule)


def load_references(path: Path | None) -> list[tuple[str, object, float]]:
    if path is None:
        return []
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"name", "SMILES", "reference_weight"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"{path} must contain {sorted(required)}")
    references: list[tuple[str, object, float]] = []
    for row in rows:
        molecule = Chem.MolFromSmiles(row["SMILES"])
        weight = float(row["reference_weight"])
        if molecule is None or not math.isfinite(weight) or weight < 0:
            raise ValueError(f"Invalid reference ligand row: {row}")
        if QUINAZOLINE is not None and molecule.HasSubstructMatch(QUINAZOLINE):
            raise ValueError(f"Reference {row['name']} contains excluded quinazoline core")
        references.append((row["name"], fingerprint(molecule), weight))
    return references


def reference_score(molecule: Chem.Mol, references: list[tuple[str, object, float]]) -> tuple[float, str]:
    if not references:
        return 0.0, ""
    query = fingerprint(molecule)
    scored = [
        (float(DataStructs.TanimotoSimilarity(query, fp)) * weight, name)
        for name, fp, weight in references
    ]
    return max(scored)


def scaffold_smiles(molecule: Chem.Mol) -> str:
    scaffold = MurckoScaffold.GetScaffoldForMol(molecule)
    return Chem.MolToSmiles(scaffold, canonical=True) if scaffold.GetNumAtoms() else "<acyclic>"


def select_diverse(rows: list[dict[str, object]], size: int, max_per_scaffold: int) -> list[dict[str, object]]:
    selected: list[dict[str, object]] = []
    counts: dict[str, int] = {}
    for row in rows:
        scaffold = str(row["murcko_scaffold"])
        if counts.get(scaffold, 0) >= max_per_scaffold:
            continue
        selected.append(row)
        counts[scaffold] = counts.get(scaffold, 0) + 1
        if len(selected) == size:
            break
    return selected


def main() -> int:
    args = parse_args()
    if args.shortlist_size <= 0 or args.max_per_scaffold <= 0 or args.max_heavy_atoms <= 0:
        raise ValueError("Shortlist size, scaffold cap, and heavy-atom limit must be positive")
    weight_args = {
        "matched_sar_rank": args.matched_sar_weight,
        "competition_rank": args.competition_weight,
        "pgk2_logica_rank": args.pgk2_logica_weight,
        "logica_selectivity_rank": args.logica_delta_weight,
        "fingerprint_rank": args.fingerprint_weight,
        "reference_ligand_rank": args.reference_ligand_weight,
        "structural_selectivity_rank": args.structural_weight,
    }
    if any(not math.isfinite(value) or value < 0 for value in weight_args.values()):
        raise ValueError("Fusion weights must be finite and non-negative")

    panel = read_panel(args.candidate_panels, args.panel)
    panel_ids = {row["CatalogID"] for row in panel}
    raw_sources = {
        "matched_sar_rank": read_score_map(args.matched_sar_scores, "logica_score"),
        "competition_rank": read_score_map(args.competition_scores, "logica_score"),
        "pgk2_logica_rank": read_score_map(args.dual_logica_scores, "pgk2_logica_score"),
        "logica_selectivity_rank": read_score_map(
            args.dual_logica_scores, "logica_selectivity_delta"
        ),
        "fingerprint_rank": read_score_map(args.fingerprint_scores, "fingerprint_score"),
        "structural_selectivity_rank": read_score_map(
            args.structural_scores, "structural_selectivity_score"
        ),
    }
    for name, source in raw_sources.items():
        extra = set(source) - panel_ids
        if extra:
            raise ValueError(f"{name} contains {len(extra)} CatalogIDs outside the {args.panel} panel")
    references = load_references(args.reference_ligands)

    reference_raw: dict[str, float] = {}
    reference_names: dict[str, str] = {}
    prepared_rows: list[dict[str, object]] = []
    invalid_smiles = 0
    for row in panel:
        molecule = Chem.MolFromSmiles(row["SMILES"])
        if molecule is None:
            invalid_smiles += 1
            continue
        ref_score, ref_name = reference_score(molecule, references)
        if references:
            reference_raw[row["CatalogID"]] = ref_score
            reference_names[row["CatalogID"]] = ref_name
        quinazoline = bool(
            QUINAZOLINE is not None and molecule.HasSubstructMatch(QUINAZOLINE)
        )
        prepared_rows.append(
            {
                **row,
                "contains_quinazoline": quinazoline,
                "heavy_atoms": molecule.GetNumHeavyAtoms(),
                "murcko_scaffold": scaffold_smiles(molecule),
                "nearest_reference": ref_name,
                "reference_ligand_score": ref_score,
            }
        )
    raw_sources["reference_ligand_rank"] = reference_raw
    ranks = {name: percentile_map(source) for name, source in raw_sources.items()}
    enabled = {
        name: weight_args[name]
        for name, source in raw_sources.items()
        if source and weight_args[name] > 0
    }
    if not enabled:
        raise ValueError("No non-empty score source has positive fusion weight")
    total_weight = sum(enabled.values())

    for row in prepared_rows:
        catalog_id = str(row["CatalogID"])
        contribution = 0.0
        available = 0
        for name, weight in enabled.items():
            value = ranks[name].get(catalog_id, 0.0)
            row[name] = value
            contribution += weight * value
            available += int(catalog_id in ranks[name])
        row["source_available_count"] = available
        row["fusion_score"] = contribution / total_weight
        row["structural_screen_eligible"] = (
            (args.allow_quinazolines or not bool(row["contains_quinazoline"]))
            and int(row["heavy_atoms"]) <= args.max_heavy_atoms
        )
    ranked = sorted(
        prepared_rows,
        key=lambda row: (
            not bool(row["structural_screen_eligible"]),
            -float(row["fusion_score"]),
            str(row["CatalogID"]),
        ),
    )
    for index, row in enumerate(ranked, start=1):
        row["fusion_rank"] = index
    ranked_fields = list(ranked[0])
    eligible = [row for row in ranked if bool(row["structural_screen_eligible"])]
    shortlist = select_diverse(eligible, min(args.shortlist_size, len(eligible)), args.max_per_scaffold)
    for index, row in enumerate(shortlist, start=1):
        row["shortlist_rank"] = index

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ranked_path = args.output_dir / f"{args.panel}_fused_ranked.csv"
    shortlist_path = args.output_dir / f"{args.panel}_structural_shortlist.csv"
    with ranked_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ranked_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(ranked)
    shortlist_fields = ["shortlist_rank", *ranked_fields]
    with shortlist_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=shortlist_fields)
        writer.writeheader()
        writer.writerows(shortlist)

    report = {
        "panel": args.panel,
        "panel_rows": len(panel),
        "valid_smiles": len(prepared_rows),
        "invalid_smiles": invalid_smiles,
        "enabled_sources": enabled,
        "source_coverage": {
            name: len(source) for name, source in raw_sources.items() if source
        },
        "reference_ligands": len(references),
        "quinazolines_excluded_from_shortlist": sum(
            bool(row["contains_quinazoline"]) for row in prepared_rows
        ) if not args.allow_quinazolines else 0,
        "heavy_atom_limit": args.max_heavy_atoms,
        "shortlist_rows": len(shortlist),
        "distinct_shortlist_scaffolds": len({row["murcko_scaffold"] for row in shortlist}),
        "outputs": {"ranked": str(ranked_path), "shortlist": str(shortlist_path)},
        "limitations": [
            "Rank fusion is a prespecified heuristic and is not calibrated to inhibition probability.",
            "Sparse source files contribute zero to compounds absent from that source.",
            "No challenge validation or test labels are consumed.",
        ],
    }
    (args.output_dir / f"{args.panel}_report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
