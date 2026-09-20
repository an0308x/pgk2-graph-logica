#!/usr/bin/env python3
"""Freeze preparation provenance only; this script cannot launch model training."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/"artifacts/pgk2_substrate_protocol_v1"


def main():
    output=BASE/"preparation_manifest.json"
    if output.exists(): raise FileExistsError(output)
    files=[ROOT/"PGK2_SUBSTRATE_COMPARISON_PROTOCOL_2026-09-20.md",
        ROOT/"scripts/derive_pgk2_substrate_pocket.py",ROOT/"scripts/audit_pgk2_evidence_coverage.py",
        ROOT/"scripts/freeze_pgk2_substrate_protocol.py",ROOT/"tests/test_pgk2_substrate_preparation.py",
        *sorted((BASE/"pocket").glob("*.json")),BASE/"evidence/report.json",BASE/"evidence/evidence_panel.parquet"]
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    evidence=json.loads((BASE/"evidence/report.json").read_text())
    if hashes[str((BASE/"evidence/evidence_panel.parquet").relative_to(ROOT))]!=evidence["evidence_panel_sha256"]:
        raise ValueError("Evidence checksum mismatch")
    payload={"protocol":"pgk2_substrate_comparison_v1","date":"2026-09-20",
        "status":"preparation_complete_execution_not_authorized","artifact_sha256":hashes,
        "source_training_manifest_sha256":evidence["manifest_sha256"],"audit_job":10945609,
        "tests_passed":78,"training_launched":False,"challenge_submission_created":False,
        "arms":["substrate_mask_common_context","inhibitor_mask_common_context","ligand_only"],
        "seeds":[2026,2027,2028],"reuse_cached_graphs":True,
        "readiness":{"structural_masks":True,"source_evidence_coverage":True,
            "independent_kinase_activity_evaluation":False,"adequate_transfer_stress_evaluation":False},
        "decision_required":"Authorize explicitly exploratory DEL-only comparison, or revise evaluation using independent permissible evidence before training."}
    output.write_text(json.dumps(payload,indent=2)+"\n")
    print(json.dumps({"path":str(output),"status":payload["status"],"hashed_files":len(hashes)},indent=2))


if __name__=="__main__": main()
