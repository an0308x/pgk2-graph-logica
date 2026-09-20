#!/usr/bin/env python3
"""Audit complete validation scores and emit the raw top-50 CatalogID text file."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import polars as pl
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checked_ranking(frame,candidates):
    if frame.height!=len(candidates) or frame["CatalogID"].n_unique()!=len(candidates):
        raise ValueError("Missing or duplicate candidate scores")
    ordered=frame.sort("row_index")
    if ordered["row_index"].to_list()!=list(range(len(candidates))):
        raise ValueError("Row coverage mismatch")
    if ordered["CatalogID"].to_list()!=[r["CatalogID"] for r in candidates]:
        raise ValueError("CatalogID/source order mismatch")
    if not np.isfinite(ordered["score"].to_numpy()).all():
        raise ValueError("Nonfinite score")
    return ordered.sort(["score","CatalogID"],descending=[True,False])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input",type=Path,required=True)
    p.add_argument("--chunks",type=Path,required=True)
    p.add_argument("--verification",type=Path,required=True)
    p.add_argument("--output-dir",type=Path,required=True)
    a=p.parse_args()
    inp=json.loads(a.input.read_text()); verified=json.loads(a.verification.read_text())
    if inp["panel"]!="validation" or inp["rows"]!=244328 or not verified["verified"]:
        raise ValueError("Not a verified validation run")
    if digest(a.input)!=verified["identity"]["input_sha256"]:
        raise ValueError("Candidate input changed")
    frames=[]; reports=[]
    for i in range(8):
        path=a.chunks/f"shard-{i:03d}.parquet"
        r=json.loads(path.with_suffix(".json").read_text())
        if not r["completed"] or r["identity"]!=verified["identity"] or r["num_shards"]!=8:
            raise ValueError("Inconsistent shard identity")
        if r["score_sha256"]!=digest(path) or r["range"]!=[244328*i//8,244328*(i+1)//8]:
            raise ValueError("Shard checksum or bounds mismatch")
        f=pl.read_parquet(path)
        if f.height!=r["rows"] or f["row_index"].to_list()!=list(range(*r["range"])):
            raise ValueError("Shard coverage mismatch")
        frames.append(f); reports.append(r)
    ranking=checked_ranking(pl.concat(frames),inp["candidates"])
    top=ranking.head(50)
    ids=top["CatalogID"].to_list()
    if len(set(ids))!=50 or any(not i or "\n" in i or "\r" in i for i in ids):
        raise ValueError("Invalid top-50 IDs")
    a.output_dir.mkdir(parents=True,exist_ok=False)
    ranking.write_parquet(a.output_dir/"validation_ranked.parquet")
    top.write_parquet(a.output_dir/"top50.parquet")
    output=a.output_dir/"graph_logica_v4_cap5_validation_submission.txt"
    output.write_bytes(("\r\n".join(ids)+"\r\n").encode("utf-8"))
    assert output.read_text().splitlines()==ids
    scaffolds=[Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(Chem.MolFromSmiles(s))) for s in top["canonical_smiles"]]
    result={"identity":verified["identity"],"panel":"validation","rows":ranking.height,"selected":50,
        "selection":"Raw descending model score; CatalogID ascending breaks exact ties; no reranking",
        "unique_top50_structures":top["canonical_smiles"].n_unique(),"unique_top50_murcko_scaffolds":len(set(scaffolds)),
        "truncated_visits":sum(r["truncated_visits"] for r in reports),"submission_sha256":digest(output),
        "submission_file":output.name,"score_quantiles":np.quantile(ranking["score"].to_numpy(),[0,.01,.5,.99,1]).tolist(),
        "challenge_submitted":False,"blind_test_scored":False}
    (a.output_dir/"report.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))


if __name__=="__main__": main()
