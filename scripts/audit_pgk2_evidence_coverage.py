#!/usr/bin/env python3
"""Count evidence availability without changing labels, training, or scoring hits."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import polars as pl
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, rdFingerprintGenerator

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"src"),str(ROOT/"scripts")]
from logica_binding.development_protocol import evidence_intervals
from run_pgk2_full_logica import verify_manifest
from prepare_pgk2_full_logica_manifest import sha256


def evidence_columns(frame):
    def read(name):
        v=frame[name].to_numpy().astype(float)
        if not np.isfinite(v).all() or (v<0).any():
            raise ValueError(f"Invalid {name}")
        return v
    n=read("source_rows")
    if (n<1).any():
        raise ValueError("Source-row denominator must be positive")
    t=read("count_PGK2")/n; i=read("count_PGK2_with_inhibitor")/n
    ns=read("count_NTC_selection")/n
    nu=read("count_NTC_supplement")/np.maximum(read("ntc_supplement_rows"),1)
    h=read("historic_hits")
    return t,i,ns,nu,h


def coverage(frame):
    t,i,ns,nu,h=evidence_columns(frame)
    ntc=(ns>0)|(nu>0)
    result={"molecules":len(frame),"duplicate_source_molecules":int((frame["source_rows"].to_numpy()>1).sum()),
        "inhibitor_nonzero":int((i>0).sum()),"inhibitor_zero_uncertain":int((i==0).sum()),
        "ntc_selection_nonzero":int((ns>0).sum()),"ntc_supplement_nonzero":int((nu>0).sum()),
        "any_ntc_nonzero":int(ntc.sum()),"both_controls_nonzero":int(((i>0)&ntc).sum()),
        "historic_nonzero":int((h>0).sum())}
    for name,mask in [("target_le1",t<=1),("target_gt1_lt5",(t>1)&(t<5)),
                      ("target_5_lt10",(t>=5)&(t<10)),("target_10_lt20",(t>=10)&(t<20)),("target_ge20",t>=20)]:
        result[name]=int(mask.sum())
    for cutoff in (5,10,20):
        enough=t>=cutoff
        # Cross-arm ratios below are uncalibrated audit flags, not hit labels.
        clean=(ns<=.1*t)&(nu<=.1*t)&(h==0)
        comp=(i>0)&(i<=.1*t)
        for name,mask in {
            "target_support":enough,"observed_competition":enough&(i>0),
            "observed_ntc":enough&ntc,"tenfold_competition_nonzero":enough&comp,
            "zero_inhibitor":enough&(i==0),"background_or_history_flag":enough&~clean,
            "joint_nonzero_inhibitor":enough&comp&clean,
            "joint_zero_inhibitor_uncertain":enough&(i==0)&clean,
            "joint_nonzero_inhibitor_and_observed_ntc":enough&comp&clean&ntc,
            "competition_retained_observed":enough&(i>=t),
        }.items():
            result[f"T{cutoff}_{name}"]=int(mask.sum())
    _,_,w=evidence_intervals(frame,5)
    result["legacy_cap5_any_confidence"]=int((w>0).any(axis=1).sum())
    for c in range(3): result[f"legacy_cap5_channel_{c}"]=int((w[:,c]>0).sum())
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest",type=Path,required=True)
    p.add_argument("--output-dir",type=Path,required=True)
    a=p.parse_args(); started=time.monotonic()
    if a.output_dir.exists(): raise FileExistsError(a.output_dir)
    manifest=verify_manifest(a.manifest)
    total=defaultdict(Counter); panel=[]; holdout_inventory=0
    for idx,entry in enumerate(manifest["shards"]):
        path=a.manifest.parent/entry["metadata"]
        if sha256(path)!=entry["metadata_sha256"]: raise ValueError("Metadata checksum changed")
        f=pl.read_parquet(path)
        if f["molecule_id"].to_list()!=list(range(*entry["range"])): raise ValueError("Row mapping changed")
        holdout_inventory+=f.filter(pl.col("split")=="holdout").height
        for split in ("train","dev"):
            g=f.filter(pl.col("split")==split)
            total[split].update(coverage(g))
            t,i,ns,nu,h=evidence_columns(g)
            _,_,w=evidence_intervals(g,5)
            eligible=(w>0).any(axis=1)
            take=eligible|(t>=5)
            panel.append(g[np.flatnonzero(take)].with_columns(
                pl.Series("mean_target",t[take]),pl.Series("mean_inhibitor",i[take]),
                pl.Series("mean_ntc_selection",ns[take]),pl.Series("mean_ntc_supplement",nu[take]),
                pl.Series("legacy_cap5_any_confidence",eligible[take])))
        if idx%16==0: print(json.dumps({"phase":"coverage","shard":idx}),flush=True)
    for split in total:
        if total[split]["molecules"]!=manifest["splits"][split]["molecules"]: raise ValueError("Split count mismatch")
    frame=pl.concat(panel)
    generator=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048)
    props=[]; fps=[]
    for smi in frame["canonical_smiles"]:
        m=Chem.MolFromSmiles(smi)
        if m is None: raise ValueError("Invalid molecule")
        props.append((m.GetNumHeavyAtoms(),Descriptors.MolWt(m)))
        fps.append(generator.GetFingerprint(m))
    frame=frame.with_columns(pl.Series("heavy_atoms",[x[0] for x in props]),pl.Series("mol_weight",[x[1] for x in props]))
    reference_indices=np.flatnonzero((frame["split"].to_numpy()=="train")&frame["legacy_cap5_any_confidence"].to_numpy())
    ref=[fps[x] for x in reference_indices]
    similarities=[None]*len(frame)
    for j in np.flatnonzero(frame["split"].to_numpy()=="dev"):
        similarities[j]=max(DataStructs.BulkTanimotoSimilarity(fps[j],ref))
    frame=frame.with_columns(pl.Series("nearest_legacy_supervised_train_tanimoto",similarities,dtype=pl.Float64))
    # This is only a prespecified diagnostic subgroup of existing development,
    # not a newly independent split and not a claim of ASMS distribution equality.
    frame=frame.with_columns(((pl.col("split")=="dev")&(pl.col("heavy_atoms")<=37)&(pl.col("mol_weight")<=500)
        &(pl.col("nearest_legacy_supervised_train_tanimoto")<=.4)).fill_null(False).alias("transfer_stress"))
    def flags(g):
        t=pl.col("mean_target"); i=pl.col("mean_inhibitor")
        clean=(pl.col("mean_ntc_selection")<=.1*t)&(pl.col("mean_ntc_supplement")<=.1*t)&(pl.col("historic_hits")==0)
        groups={"all":g,"nonzero_inhibitor_joint_proxy":g.filter((t>=5)&(i>0)&(i<=.1*t)&clean),
            "zero_inhibitor_joint_uncertain":g.filter((t>=5)&(i==0)&clean),
            "observed_competition_retained":g.filter((t>=5)&(i>=t)),
            "target_ge5":g.filter(t>=5)}
        return {k:{"molecules":v.height,"scaffolds":v["scaffold_key"].n_unique()} for k,v in groups.items()}
    scope={"train_evidence_panel":flags(frame.filter(pl.col("split")=="train")),
        "development_evidence_panel":flags(frame.filter(pl.col("split")=="dev")),
        "development_transfer_stress":flags(frame.filter(pl.col("transfer_stress")))}
    a.output_dir.mkdir(parents=True)
    frame.write_parquet(a.output_dir/"evidence_panel.parquet")
    result={"protocol":"pgk2_substrate_preparation_v1","manifest_sha256":manifest["manifest_sha256"],
        "coverage":dict(total),"holdout_inventory_only":holdout_inventory,
        "panel_coverage":scope,"nearest_neighbor_reference_count":len(ref),
        "evidence_panel_sha256":sha256(a.output_dir/"evidence_panel.parquet"),
        "source_sha256":sha256(Path(__file__)),"elapsed_seconds":time.monotonic()-started,
        "training_launched":False,"labels_changed":False,"holdout_labels_evaluated":False,
        "limitations":["Mean counts reuse historical aggregation; duplicate source rows are not independent replicates.",
            "Cross-arm tenfold ratios are descriptive heuristics, not depth-calibrated enrichment or inhibition labels.",
            "Inhibitor or NTC zero is uncertain; absence of a historical-hit flag does not establish specificity.",
            "Missing inhibitor-arm-only rows prevent reconstruction of complete inhibitor depth.",
            "The evidence panel is an audit subset, not a replacement training universe.",
            "Stress similarity is against all legacy cap5 ranking-eligible training molecules, not the entire 6M pool.",
            "Existing development was previously used for model selection; the stress subgroup is not an untouched holdout."]}
    (a.output_dir/"report.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2),flush=True)


if __name__=="__main__": main()
