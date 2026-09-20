#!/usr/bin/env python3
"""Prepare fixed inference probes and diagnose DEL/validation chemical shift."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import time
import numpy as np
import polars as pl
from rdkit import Chem,DataStructs
from rdkit.Chem import Descriptors,Crippen,rdMolDescriptors,rdFingerprintGenerator

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"src"),str(ROOT/"scripts")]
from logica_binding.development_protocol import evidence_intervals
from logica_binding.failure_audit import spearman,pair_diagnostics
from run_pgk2_full_logica import verify_manifest
from prepare_pgk2_full_logica_manifest import sha256

PROPERTIES=("heavy_atoms","mol_weight","logp","tpsa","rotatable_bonds","aromatic_rings")


def properties(smi):
    m=Chem.MolFromSmiles(smi)
    if m is None: raise ValueError("Invalid audited molecule")
    return dict(zip(PROPERTIES,[m.GetNumHeavyAtoms(),Descriptors.MolWt(m),Crippen.MolLogP(m),
        rdMolDescriptors.CalcTPSA(m),rdMolDescriptors.CalcNumRotatableBonds(m),rdMolDescriptors.CalcNumAromaticRings(m)]))


def summary(frame):
    return {name:dict(zip(("min","q10","median","q90","max"),
        np.quantile(frame[name].to_numpy(),[0,.1,.5,.9,1]).tolist())) for name in PROPERTIES}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest",type=Path,required=True)
    p.add_argument("--merged",type=Path,required=True)
    p.add_argument("--submission",type=Path,required=True)
    p.add_argument("--output-dir",type=Path,required=True)
    a=p.parse_args(); started=time.monotonic()
    a.output_dir.mkdir(parents=True,exist_ok=False)
    manifest=verify_manifest(a.manifest)
    report=json.loads((a.merged/"report.json").read_text())
    ranking=pl.read_parquet(a.merged/"validation_ranked.parquet")
    submitted=a.submission.read_text().splitlines()
    if (report["submission_sha256"]!=sha256(a.submission) or ranking.height!=244328
        or ranking.head(50)["CatalogID"].to_list()!=submitted
        or report["identity"]["training_manifest_sha256"]!=manifest["manifest_sha256"]):
        raise ValueError("Submission/ranking identity mismatch")
    validation=ranking.with_row_index("submitted_model_rank",offset=1)
    rng=np.random.default_rng(9780912)
    random_ids=set(rng.choice(validation["CatalogID"].to_numpy(),4096,replace=False).tolist())
    probes=validation.filter((pl.col("submitted_model_rank")<=1000)|pl.col("CatalogID").is_in(list(random_ids)))
    probes=probes.with_columns(pl.col("CatalogID").is_in(list(random_ids)).alias("random_probe"))
    probes.write_parquet(a.output_dir/"validation_probe.parquet")
    training=[]; background=[]; dev=[]; original_batch=0
    pair_totals=defaultdict(lambda:{"pairs":0,"weight_sum":0.,"weighted_wins":defaultdict(float)})
    for shard_id,entry in enumerate(manifest["shards"]):
        mp=a.manifest.parent/entry["metadata"]
        if sha256(mp)!=entry["metadata_sha256"]: raise ValueError("Metadata changed")
        f=pl.read_parquet(mp); lo,hi,w=evidence_intervals(f,5)
        train=f["split"].to_numpy()=="train"
        eligible=train & (w>0).any(axis=1)
        rows=f[np.flatnonzero(eligible)].with_columns(pl.Series("positive_competition_proxy",(lo[eligible,0]>0)&(w[eligible,0]>0)))
        training.append(rows)
        pool=np.flatnonzero(train)
        bg=np.random.default_rng(9780912+shard_id).choice(pool,min(32,len(pool)),replace=False)
        background.append(f[bg])
        for channel in range(3):
            group=np.flatnonzero((f["split"].to_numpy()=="dev")&(f[f"confidence_{channel}"].to_numpy()>0))
            for offset in range(0,len(group),32):
                pos=group[offset:offset+32]
                if len(pos)<2: continue
                batch=f[pos].with_columns(pl.lit(original_batch).alias("batch_id"),pl.lit(channel).alias("eval_channel"))
                dev.append(batch); original_batch+=1
                if channel==0:
                    for name,row in pair_diagnostics(batch,{}).items():
                        pair_totals[name]["pairs"]+=row["pairs"]; pair_totals[name]["weight_sum"]+=row["weight_sum"]
                        for pred,value in row["weighted_wins"].items(): pair_totals[name]["weighted_wins"][pred]+=value
    reference=pl.concat(training); dev=pl.concat(dev); background=pl.concat(background)
    reference.write_parquet(a.output_dir/"training_reference.parquet")
    dev.write_parquet(a.output_dir/"development_probe.parquet")
    for value in pair_totals.values():
        value["weighted_accuracy"]={k:v/value["weight_sum"] if value["weight_sum"] else None for k,v in value["weighted_wins"].items()}
    print(json.dumps({"phase":"metadata","training_reference":reference.height,"dev_visits":dev.height,"validation_probe":probes.height}),flush=True)
    props=pl.DataFrame([properties(s) for s in validation["canonical_smiles"]])
    validation=validation.hstack(props); validation.write_parquet(a.output_dir/"validation_properties.parquet")
    reference=reference.hstack(pl.DataFrame([properties(s) for s in reference["canonical_smiles"]]))
    background=background.hstack(pl.DataFrame([properties(s) for s in background["canonical_smiles"]]))
    strata={"validation_all":validation,"validation_top50":validation.head(50),
        "validation_random4096":validation.filter(pl.col("CatalogID").is_in(list(random_ids))),
        "all_rank_supervised_training":reference,"training_background_32_per_shard":background}
    correlation={k:spearman(validation[k],validation["score"]) for k in PROPERTIES}
    print(json.dumps({"phase":"properties","score_correlations":correlation}),flush=True)
    # Exact NN to all ranking-supervised training molecules, NOT all 6M molecules.
    generator=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048)
    reference_fp=[generator.GetFingerprint(Chem.MolFromSmiles(s)) for s in reference["canonical_smiles"]]
    positive=np.flatnonzero(reference["positive_competition_proxy"].to_numpy())
    nn=[]
    # Second seeded random draw avoids CatalogID/library-order selection bias.
    random_nn=np.random.default_rng(9780913).choice(sorted(random_ids),512,replace=False).tolist()
    query=validation.filter((pl.col("submitted_model_rank")<=50)|pl.col("CatalogID").is_in(random_nn))
    dev_unique=dev.unique(subset="molecule_id",maintain_order=True)
    dev_sample=dev_unique.sample(n=min(512,dev_unique.height),seed=9780912)
    for kind,frame in [("validation",query),("development",dev_sample)]:
        for row in frame.iter_rows(named=True):
            fp=generator.GetFingerprint(Chem.MolFromSmiles(row["canonical_smiles"]))
            sim=np.asarray(DataStructs.BulkTanimotoSimilarity(fp,reference_fp))
            best=int(sim.argmax())
            nn.append({"kind":kind,"id":str(row["CatalogID"] if kind=="validation" else row["molecule_id"]),
                "submitted_top50":kind=="validation" and row["submitted_model_rank"]<=50,
                "max_tanimoto":float(sim[best]),"nearest_training_molecule_id":int(reference["molecule_id"][best]),
                "max_positive_proxy_tanimoto":float(sim[positive].max()) if len(positive) else None})
    nn=pl.DataFrame(nn); nn.write_parquet(a.output_dir/"nearest_supervised_training.parquet")
    groups={"submitted_top50":nn.filter(pl.col("submitted_top50")),
        "random_validation_non_top50":nn.filter((pl.col("kind")=="validation")&~pl.col("submitted_top50")),
        "development_sample":nn.filter(pl.col("kind")=="development")}
    nn_summary={label:{"n":g.height,"max_tanimoto_quantiles":np.quantile(g["max_tanimoto"],[0,.1,.5,.9,1]).tolist(),
        "fraction_ge_05":float((g["max_tanimoto"]>=.5).mean())} for label,g in groups.items()}
    result={"submission_id":9780912,"user_reported_hits":0,"submission_sha256":sha256(a.submission),
        "checkpoint_sha256":report["identity"]["checkpoint_sha256"],"training_manifest_sha256":manifest["manifest_sha256"],
        "training_reference_molecules":reference.height,"positive_proxy_reference_molecules":len(positive),
        "validation_probe_molecules":probes.height,"development_probe_visits":dev.height,
        "count_oracle_pair_diagnostics":dict(pair_totals),"properties":{k:summary(f) for k,f in strata.items()},
        "full_validation_score_property_correlations":correlation,"nearest_neighbor_summary":nn_summary,
        "elapsed_seconds":time.monotonic()-started,"labels_changed":False,"holdout_evaluated":False,
        "limitations":["Count oracles use assay counts and are not deployable compound predictors.",
            "Exact fingerprint NN reference is all ranking-supervised train molecules, not the full molecular library.",
            "Chemical similarity and property associations do not prove activity or the cause of zero hits.",
            "Aggregate zero-hit feedback is recorded only, never used to create training labels."]}
    (a.output_dir/"preparation_report.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2),flush=True)


if __name__=="__main__": main()
