# PGK2 substrate-pocket comparison: preparation and frozen design v1

Status: preparation complete; **no training or submission authorized or launched
by this protocol**. Structural readiness passes. Inhibition/ASMS-transfer evaluation
readiness does not pass. The three-arm design below can support an exploratory DEL
comparison, not a claim of validated kinase-hit prediction. Any change requires a
new protocol version before examining new model results.

## 1. What changed, and what did not

The new direct-interaction mask comes exclusively from substrate heavy-atom
neighborhoods in [RCSB 2PAA](https://www.rcsb.org/structure/2PAA): ATP chain A,
author residue 500 (31 heavy atoms), and 3-PG chain B, residue 501 (11 heavy atoms).
Distances are measured against protein atoms of the ligand's own chain. No chain
superposition or union of Cartesian coordinate frames is used to define contacts.
The union is taken only after mapping residues to human PGK2 P07205.

Direct radius is 6 Å and native context radius 8 Å. This is a geometric prior,
not a physical contact-energy cutoff or an experimentally validated inhibitor site.
The source is mouse PGK2, not a human substrate-bound crystal. Each chain's 416-aa
SEQRES maps injectively onto the 417-aa human sequence (359 identical positions).
Author numbering is checked against SEQRES; missing coordinates are recorded.
The only mouse/human substitution in the selected 8 Å neighborhoods is mouse
Ile339 / human Leu340; the generated JSON records every mapping and distance.

- New direct mask: **43 residues** (32 ATP; 11 3-PG).
- Native substrate context: **78 residues**.
- Old inhibitor mask: **26 residues**, of which 23 overlap the substrate mask.
- New direct residues relative to old, human 1-based: 24, 26, 63, 65, 123, 129,
  167, 168, 170, 171, 173, 219, 220, 337, 372, 373, 374, 376, 396, 400.
- Old-only direct residues: 255, 256, 345. Thus this is not simply a superset.

For the **controlled comparison only**, both graph arms share the union of their
native context shells: 90 residues. Both use the same human 21/47 protein-coordinate
ensemble. Every selected position exists in both templates with the expected human
amino acid. Inhibitor ligand atoms are not model inputs and do not select the new
direct mask. Reusing these human coordinates isolates mask choice from species,
conformation and template-count changes. It does **not** make the whole structural
pipeline independent of inhibitor-bound structures. A template-free or mouse-geometry
comparison would be a different experiment, not silently mixed into this one.

Full human sequence ESM remains the representation source; pocket-only sequences
must never be encoded. Both bipartite graph connections and base cross-attention
must enforce the selected direct mask. Context-only nodes supply protein graph
messages, never direct ligand attention. Candidate ligand graphs remain 2D;
there are no candidate complex poses or intermolecular distance features.

Historical files, selected checkpoints and production source files are unchanged.

## 2. Full-library evidence inventory

CPU audit `10945609` completed over all 64 checksum-verified metadata shards.
Training/development counts below are molecular counts, not independent replicates;
duplicate source rows are averaged as in the historical pipeline. Holdout was
counted for inventory only; no holdout evidence breakdown or model evaluation ran.

| Evidence availability | Train | Development |
|---|---:|---:|
| Total molecules | 6,026,213 | 732,325 |
| Mean target count <=1 | 5,928,254 | 720,807 |
| Nonzero inhibitor counts | 20,208 | 2,433 |
| Any nonzero NTC signal | 5,932 | 726 |
| Both inhibitor and NTC nonzero | 1,695 | 192 |
| Mean target count >=5 | 3,640 | 359 |
| Target >=5 and inhibitor nonzero | 754 | 74 |
| Target >=5 and 0 < inhibitor <=0.1 × target | 97 | 2 |
| Same tenfold flag plus NTC <=0.1 × target and history=0 | 7 | 0 |
| Target >=5, inhibitor=0, same NTC/history filter | 441 | 52 |
| Legacy cap5 ranking-eligible molecules | 27,327 | 3,252 |

These are **coverage flags, not new binary labels**. In particular, history=0 is
an intentionally strict audit slice, not a proposed hard training exclusion;
nonzero history is not proof of promiscuity. Zero inhibitor/NTC reads are uncertain,
not confirmed competition/clean background. Threshold sensitivity at target counts
5, 10 and 20 is preserved in the raw report. The 7/0 result must not be interpreted
as evidence that only seven real inhibitors exist in training.

The organizer confirms the released inhibitor column is censored to PGK2-positive
compounds and no comparable inhibitor-arm-only supplement is available:
https://www.synapse.org/Synapse:syn75349604/discussion/threadId=15177
Released column totals therefore must not be treated as complete sequencing depths.
The ratio flags above are not depth-calibrated enrichment or inhibition probabilities.

## 3. Transfer-stress feasibility

Predeclared diagnostic condition: existing development molecule with heavy atoms
<=37, molecular weight <=500 Da, and maximum radius-2/2,048-bit Morgan Tanimoto
<=0.4 to **all 27,327 legacy cap5-supervised training molecules**. This reference
is not all six million training molecules; maximum similarity to the full pool
could be higher. The chemistry bounds reflect the available validation library,
not per-compound activity feedback.

Among all 3,252 evidence-bearing development molecules, only **nine** meet this
condition. One has target counts >=5. None meets either the strict observed
competition-positive flag or the observed competition-retained flag with target
>=5. The subgroup is too sparse for a credible positive-vs-control retrieval test.
Existing development already informed earlier model selection; its subgroup is
not an untouched validation set. Do not relax thresholds after seeing model scores
or describe this subgroup as an ASMS-equivalent labeled test set.

## 4. Locked three-arm exploratory design (not launched)

| Arm | Ligand representation | Protein inputs / interaction mask |
|---|---|---|
| S | Existing 2D atom/bond graph + frozen ligand encoder | Full-sequence ESM; 43 substrate-defined direct residues; common 90-residue context |
| I | Identical to S | Same ESM, coordinates and context; 26 historical inhibitor-defined direct residues |
| L | Same drug encoder and ligand graph branch | No protein inputs, protein graph messages, cross-attention or protein likelihood term |

I is a **matched-context inhibitor-mask control**, not a reproduction of the
historical 49-context-residue model. Record that historical result separately.
An inference-time graph-off switch does not replace retraining arm L.

### Shared data and initialization

- Reuse all 7,487,567 cached 2D graphs; 6,026,213 are train, 732,325 development,
  729,029 historical holdout. Preserve exclusion of challenge candidate identities.
- No manually selected 308/241 subset. All train rows remain available to molecular
  representation learning; rank supervision is limited by evidence, not an arbitrary
  molecule budget. Development/holdout may not enter auxiliary training.
- Start from the same official base checkpoint, not the submission-selected cap5
  adapter. One shared ligand-only masked-reconstruction stage will cover all training
  molecules once (seed 2026, batch64, lr3e-5); freeze its resulting checkpoint identity
  before any arm-specific training. No protein inputs or assay labels in that stage.
- Initialize each arm's shared ligand components from that same artifact. Rank-stage
  seeds 2026/2027/2028; matched dropout, data order, optimizer (AdamW, lr3e-5,
  weight_decay0.01), and at most two complete eligible-pool passes. No competing
  auxiliary loss during the rank stage. Report actual molecule and pair coverage.
- Free ligand conformers are out of scope for this experiment. Token truncation
  must be reported by split and evidence stratum; unchanged ligand graphs retain all
  atoms. No unseen full-complex coordinates may be fabricated.

### Evidence-aware DEL objective and what it cannot establish

The objective remains a **DEL surrogate**, not kinase activity. Preserve separate
competition, selection-NTC and supplementary-NTC terms in logs (weights 2:1:1).
No per-compound ground-truth kinase labels are available in this preparation.

Primary observed-control recipe: per-channel proxy log(1+T)-log(1+C); both molecules
must have C>0. Eligible pair separation >=log(2), and the preferred molecule must
have mean T>=5. Confidence retains T/(T+20) × C/(C+5) / (1+historic_hits), pair weight
the smaller confidence; logistic pair loss temperature0.1, denominator the number
of eligible pairs. History is a soft weight, not a hard exclusion. NTC arms remain
separate because their aggregation/provenance differ. Use all eligible train rows,
not a sampled positive set; generate deterministic global channel batches of32,
merging singleton tails, and log rows that find no eligible pair. The full pool
is not equivalent to claiming every row supplies inhibition supervision.

Do not make (T=1,I=0) a positive. Zero-control rows may contribute to the ligand
representation stage but do not enter the primary observed-control rank loss.
Their **exclusion from primary ranking is a conservative assumption, not proof
that they carry no evidence**. A separately versioned sensitivity experiment may
use the old cap1/cap5 assumptions; they are not statistical confidence intervals
and must not silently replace the primary objective after seeing results.

Graph arms use the existing ensemble token-likelihood score; L uses drug-only
graph-conditioned token likelihood, without any PGK2 tensor or protein term.
Only parameters on that ligand-only path are optimized in L. Parameter counts
and training compute must be reported; the control is data/budget matched, not
claimed to have identical capacity. No new inference-time rank blend or filter.

### Evaluation and checkpoint selection

- Lock deterministic development comparisons before training, using the same
  channel recipe as training but never cross-split pairs. Build global channel
  batches to avoid arbitrary small per-shard tails; retain comparison identities.
- Evaluate epoch zero, then after each half-pass, for at most two passes. Select
  minimum combined confidence-scaled development loss; retain earlier checkpoint
  on ties. Report per-channel ordering, ties, pair count, total evidence weight,
  source-molecule/scaffold counts, and the historical 4,086-pair diagnostic separately.
- Report stronger-target strata >=5/10/20 separately, without tuning thresholds.
  Raw count oracles are diagnostic references, never deployable predictors.
- Compare S-I and S-L using all three seeds, report each seed and mean differences;
  uncertainty resampling unit is scaffold, not the many correlated molecular pairs.
- No source-derived retrieval metric is to be named a kinase hit rate. A strict
  joint-positive retrieval endpoint is **not estimable** with zero qualifying
  development positives. The transfer-stress subgroup is descriptive only.
- Historical holdout has been evaluated previously and is not pristine. Any
  further use must be separately approved and clearly labeled as reused.

## 5. Decision gate

Preparation passes: substrate provenance, residue mapping, identical comparison
geometry/context, cached graph reuse, and evidence coverage reporting.

**The originally hoped-for inhibition/transfer model-selection gate cannot be
implemented credibly from this audit's strict proxy groups:** zero strict observed
joint-positive development molecules and only nine stress-subgroup molecules.
Do not manufacture positives, shift cutoffs to obtain a desired model result,
or relabel the 50 submitted compounds from aggregate leaderboard feedback.

Before GPU training, choose and record one of:

1. Authorize this as an explicitly exploratory DEL-only comparison, accepting
   that it cannot establish which model will yield kinase hits; or
2. Obtain/identify independent, permissible activity evidence or assay metadata
   and revise the evaluation protocol with Aurelien before training.

Neither path is automatically authorized by completion of this preparation.
No challenge file will be generated merely because an internal score improves.

## Artifacts and verification

- `artifacts/pgk2_substrate_protocol_v1/pocket/substrate_pocket.json`: independent
  substrate-derived direct mask and native context with per-residue provenance.
- `substrate_controlled_context.json`, `inhibitor_controlled_context.json` in the
  same directory: matched-context graph-arm definitions.
- `artifacts/pgk2_substrate_protocol_v1/evidence/report.json`: full train/dev
  counts, cutoff sensitivities and limitations.
- `evidence_panel.parquet` in the same directory: audit-eligible molecules and
  transfer flags; **not a replacement full training manifest**.
- `scripts/derive_pgk2_substrate_pocket.py`,
  `scripts/audit_pgk2_evidence_coverage.py`: reproducible preparation.
- `tests/test_pgk2_substrate_preparation.py`: substrate identity, numbering,
  missing-coordinate handling, common-geometry and uncertain-zero tests.

All **78 tests pass**, including the existing real-checkpoint smoke tests. The
downloaded evidence-panel checksum matches its remote report. Audit job `10945609`
completed with exit code zero in 70 seconds scheduler wall time (29.76 seconds
measured computation). The protocol and artifacts are frozen by
`artifacts/pgk2_substrate_protocol_v1/preparation_manifest.json`.

Public provenance was checked through the RCSB skill on 2026-09-20 UTC. Successful
entry container identifiers resolve to https://www.rcsb.org/structure/2PAA.
The separate `struct` title request is retained as a metadata-only checked source,
not independent experimental evidence. Source coordinate SHA-256 and human FASTA
SHA-256 are recorded in the generated pocket definition. No raw API dump was saved.
