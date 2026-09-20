# PGK2 DREAM × CACHE — current handoff

Updated: 2026-09-20

## Current request: substrate pocket and comparison preparation — complete

User approved the preparation-only plan after asking whether to abandon the
exclusive 21/47 pocket. See `PGK2_SUBSTRATE_COMPARISON_PROTOCOL_2026-09-20.md`.
No GPU training, full conformer generation, candidate scoring, or submission is
authorized by that preparation. All 78 tests pass.

New substrate-only direct mask: ATP/3PG in 2PAA, separately measured in chains
A/B, sequence-mapped to human PGK2; 43 direct residues and 78 native-context
residues. Both controlled graph arms use a common 90-residue context and the same
human 21/47 coordinate ensemble, with complete sequence/coordinate coverage. This
removes inhibitor dependence from direct-mask derivation, not from human geometry.
Historical pocket files and production model sources remain unchanged.

CPU coverage job `10945609` completed successfully (70s wall time), auditing all
64 metadata shards. Training: 6,026,213 molecules; only 754 have target mean>=5
and nonzero inhibitor counts. Development: 732,325 molecules; 74 meet that same
condition, and only two meet a raw tenfold competition heuristic. With additional
strict NTC/history filtering, the observed joint-positive slice has 7 train and
0 development molecules; these are NOT true inhibitor labels or a recommended
hard training filter. Of 3,252 evidence-bearing development molecules, only nine
pass the small/distant chemistry stress condition; one has target mean>=5.

Protocol v1 freezes the exploratory three-arm design and explicitly fails the
inhibition/ASMS-transfer evaluation readiness gate. Next user decision: authorize
DEL-only exploratory comparison with these limits, or revise evaluation with
independent permissible evidence/Aurelien before training. Do not silently launch.
Artifacts and hashed provenance: `artifacts/pgk2_substrate_protocol_v1/`.

Current authenticated SSH socket (if still live):
`/tmp/pgk2-audit-ssh.FeiTF8/control-substrate-plan`; session `92975`.

## Completed diagnosis of zero-hit submission 9780912

User reports submission `9780912` scored `N_hits: 0` after receiving the v4 cap5
ranking-only top-50 file. This establishes failure of this submitted ranking,
not the cause of failure or general failure of the architecture. User approved
a diagnostic audit, not further training or challenge submission.

See `PGK2_FAILURE_AUDIT_9780912.md`. CPU preparation `10943595` and GPU rerun
`10943727` completed successfully. Original GPU `10943597` stopped on a tiny
saved-score numerical discrepancy; rerun added tighter same-batch production
parity, verified unchanged top 50, and reproduced development metrics. All 73
local tests pass. Production sources and selected checkpoint are unchanged.

Key results: graph-off/base-cross-attention-retained competition accuracy 80.09%
versus full 80.20%; no-cross-attention 50.29%; inverse-inhibitor-count oracle 88.04%
(diagnostic, not deployable). Of 4,086 development pairs, 1,946 prefer a molecule
with mean target count <=1. Median nearest supervised-training Morgan similarity
is 0.6955 in a development sample versus 0.3085 in submitted 50. Major concerns:
weak-count surrogate evaluation and chemical transfer. These do not prove a
unique cause of zero hits or blame compounds 21/47. No training or submission
was performed. Recommended evidence/evaluation redesign is not yet authorized
or implemented. Do not resume Synapse upload.

## Completed v4 validation file delivery (historical)

All nine development tasks `10937301_0`–`_8` completed successfully. Leading
selected configuration: cap5 graph ranking only, step 2198, competition ordering
80.1968% and combined loss 0.0013149744. User explicitly requested validation
submission without the additional seed/holdout experiments.

See `PGK2_V4_VALIDATION_SUBMISSION_2026-09-19.md`. Verification `10939858_0`
completed successfully, reproducing all selected development metrics and fresh
2D graph scores. Full validation scoring array `10939870` submitted for all
244,328 validation candidates. No blind test scoring or submission authorized.
Original adapter flags/checkpoints unchanged; new inference has an explicit
user-authorization gate. All eight scoring tasks and merge completed. User
requested file delivery only and uploaded personally; the result above
supersedes the previous pending-login status. All 68 inference-era tests passed.

## Earlier bounded development experiments (completed)

User approved supervision-handling and checkpoint-selection experiments.
See `PGK2_DEV_V4_2026-09-19.md`: nine prespecified arms, unchanged observed-control
development pairs, initial/intermediate checkpoint selection, at most two
eligible-pool passes with early stopping. No new full-library training,
holdout evaluation, candidate inference or challenge submission. All 65 local
tests pass, including the new real-checkpoint smoke tests. The new development
runner implements checkpoint selection; historical v3 outputs remain unchanged.
Submitted array `10937301` (tasks 0–8; at most two concurrent generic GPUs).
Initial 12:47 scheduler check: pending for Priority; runtime/results not yet
verified. Task mapping and output paths are in the v4 experiment document.

## Latest post-training evaluation

Full run `10927166` completed with exit code zero in 6h11m44s and verified
coverage of all 6,026,213 training molecules. Its holdout competition weighted
pair accuracy was 60.513% (3,857 comparisons), not a kinase hit rate. Selected
`best.pt` and resumable `last.pt` remain on McCleary; `report.json` is also local.

On user approval, development-only baseline job `10936708` and supervision
audit `10936707` were launched. They compare the completed model to its starting
initialization and a simple ligand-only ranker, reproduce the exact development
evaluation, and count target-enriched zero-competitor rows excluded by v3.
No labels are changed, no new holdout evaluation is run, and no challenge
submission is made. See `PGK2_FULL_LOGICA_V3.md` for the exact comparison recipe.
Both jobs completed successfully. Trained Graph LogiCA achieved 61.25% weighted
development competition accuracy versus 60.93% at initialization and 73.34% for
the simple Morgan baseline. Combined development pair loss was worse after graph
training; Morgan also has worse loss despite better ordering. Supervision audit
found 248 high-target (mean reads >=20), zero-competitor train molecules excluded
from competition loss, all with some NTC/history signal. No relabeling occurred.
See `PGK2_V3_BASELINE_AUDIT_2026-09-19.md` for the completed analysis. All 56 local
tests pass. ASMS transfer checks and any new challenge submission remain pending.

## Full-library Graph LogiCA v3 implementation

See `PGK2_FULL_LOGICA_V3.md` for the new streamed model, exact objective,
coverage, and execution record. The full 7,487,567-molecule manifest is complete:
6,026,213 train, 732,325 development, and 729,029 holdout molecules under the
corrected shared scaffold protocol. Graphs are reused, not rebuilt.

All train molecules contribute ligand-only molecular self-supervision; the
24,445 train molecules with observed controls support separate confidence-scaled
DEL ranking. These are not millions of kinase-inhibition labels. The scorer is
actual graph-conditioned LogiCA, not the previous nine-head regression GNN.

Manifest job `10924498` completed. Original GPU smoke `10924529` failed before
training due to an unnecessary SciPy import from the older evaluation script.
Graph I/O was separated into `src/logica_binding/packed_graph_data.py`; replacement
smoke `10927119` passed on an A5000. It measured 267.47 molecules/second and
1.70 GiB peak GPU memory across 4,096 auxiliary and 512 ranking molecule visits.
Local verification passes 53 tests, including actual-checkpoint CPU training,
exact restart/evaluation agreement, and full-launch readiness checks.

**Full training job `10927166` completed on `r102u33n01`**: one A5000,
four CPUs, 40 GB host RAM, one complete epoch over 6,026,213
training molecules in 6h11m44s. Output:
`artifacts/pgk2_full_logica_v3_train`; logs under
`artifacts/pgk2_full_logica_v3_logs/train-10927166.{out,err}`. No challenge
submission is made. About 32.6% of smoke auxiliary visits hit the 128-token
SELFIES cap; full atom/bond graphs are retained and this limitation is documented.
The full run completed without runtime errors; the post-training audit above
supersedes its initial launch status.

## Correctness repair after submission 9780853 (2026-09-17)

This section supersedes earlier statements that the zero-hit submissions
"falsified" the Graph LogiCA architecture. They establish failure of those
rankings, not of a correctly implemented and independently evaluated method.
9780853 used a Morgan MLP; 9780845 used the earlier graph-conditioned LogiCA.
The later nine-output GNN ablation is not the same architecture as Graph LogiCA.

Audit reproduced three concrete problems: the original LogiCA attention bypassed
the graph's pocket restriction; the old pair splitter leaked molecules between
training and holdout (17 in refinement, plus 18 refinement-holdout molecules
seen during pretraining); and custom ranking metrics mishandled ties.

Repairs now restrict both directions of every base cross-attention layer to
the direct-pocket tokens, while full-sequence ESM and the 49-residue structural
context remain available. SciPy/sklearn now provide tie-aware rank metrics.
New regression tests verify score invariance and zero gradients for embeddings
outside the context after full-sequence encoding. All 40 tests pass, and a
local 12-pair CPU smoke run verified actual checkpoint training/evaluation.
The 12-pair smoke is only an execution check and is blocked from panel scoring.

`scripts/prepare_pgk2_corrected_pairs.py` builds one deterministic canonical
molecule/scaffold assignment shared across stages and seeds, excluding all
428,960 challenge candidates. Pairs crossing splits are dropped. The audited
manifest SHA-256 is
`ea9be4496f01f51193f2a06d080c18bc2b66195cf346b3166f52cc1b0e8d4661`.
Cross-stage train/dev/holdout molecule overlap is zero.

The repaired broad pair pool has 7,989 train, 171 development, and 158 holdout
pairs. The repaired refinement pool has 171/6/2; it is too small for a useful
held-out comparison and is not run in the repair diagnostic. This diagnostic
does NOT fulfill full-library Graph LogiCA training. The full-data objective
remains pending; do not substitute these diagnostic checkpoints for it.

Counts now downweight uncertain pair preferences using
`original_weight * positive_target/(positive_target+20) * comparator_control/(comparator_control+5)`.
Only observed comparator inhibitor/NTC reads supply control support. This is
an explicitly heuristic reliability weight, not a calibrated probability or a
correction for unknown sequencing depths. No normalized z-score aggregation is
used in the corrected pair objective. Historical `sum(z)/sqrt(n)` aggregation
in the full index remains unsuitable for interpreting those columns as
statistical Z-scores and must be revisited before a new full-data fit.

Array `10905406` runs the repaired broad-pool diagnostic with the graph branch
on/off and seeds 2026--2028, three epochs each, topology-only ligand graphs.
Both conditions use the same masked protein interface and weighted objective;
the off condition is LogiCA without graph conditioning, not a Morgan baseline.
Development weighted pair win rate selects the checkpoint; holdout is evaluated
only afterward. Diagnostic adapters and legacy unversioned adapters are blocked
from challenge candidate scoring. No challenge submission is made by this run.

September 18 execution update: all three graph-on tasks and the seed-2027
graph-off control completed successfully. The seed-2026 and seed-2028 controls
failed before training because CUDA was unavailable on node `r109u04n01`.
Replacement array `10923943`, tasks 3 and 5 only, requests A5000 GPUs and
excludes that node. No successful run was resubmitted. The complete local test
suite was rerun: 40 passed.

Task 3 completed successfully in 3m32s. For queued task 5, the GPU restriction
was broadened and the wall-time limit reduced to 20 minutes for scheduling;
it then started on `r104u12n01`, the same A5000 node that ran task 3. No model,
data, seed, batch size, or epoch setting was changed. Added summary-script
regression tests bring the local suite to 45 passing tests. The summary refuses
mixed splits/conditions and verifies development-based checkpoint selection.

The retained diagnostic pairs cover 7,145/237/212 unique molecules and
4,051/136/138 scaffolds in train/dev/holdout respectively. The 158 holdout pairs
are therefore not 158 independent biological validation measurements; they
share compounds and carry DEL-derived preferences. Their weight-only effective
sample size is approximately 115.8, before accounting for pair dependence.

### Completed repair diagnostic (2026-09-18)

Both replacement controls completed with exit code zero: task 3 in 3m32s and
task 5 in 3m18s. All six reports are available locally under
`artifacts/pgk2_graph_logica_v2_diagnostic/`; the verified summary is
`comparison_report.json` in that directory. Checkpoints remain on McCleary.

| Condition | Holdout pair win rate, mean across 3 seeds | Evidence-weighted win rate, mean across 3 seeds |
| --- | ---: | ---: |
| Repaired Graph LogiCA | 60.13% | 63.28% |
| Repaired pocket-masked LogiCA, graph conditioning off | 59.28% | 62.42% |

Weighted graph-minus-control differences by seed were +0.80, -1.47, and
+3.24 percentage points; their mean is +0.86 points. This is a small, mixed
diagnostic result, not established statistical or biological improvement.
The three seeds reuse the same holdout, so seed variation is not a confidence
interval over independent test sets. Checkpoint epochs were selected only on
development data: graph-on 2/1/3 and graph-off 3/2/2.

The repaired path now passes the tested implementation invariants, but these
results do not verify kinase inhibition, explain every prior zero-hit result,
or establish that compounds 21/47 are responsible. No challenge predictions
or submissions were generated. No jobs from these diagnostic arrays remain
running or pending.

Before production: extend the actual repaired Graph LogiCA path to a streamed
full-library objective, preserve the canonical scaffold exclusions across all
stages, replace the historical normalized-z aggregation, and avoid treating
zero reads as confirmed negative labels. Low-count and censored observations must remain
uncertain. The existing 7,487,567 RDKit 2D graphs can be reused; this diagnostic
does not replace full-library training. Repeat a matched 2D-versus-ETKDG
comparison under the corrected full-data protocol before considering full
3D conformer generation. Do not relabel the nine-head regression GNN as
Graph LogiCA, or promote these small-pool diagnostic adapters to production.

Relevant files: `src/logica_binding/validation_protocol.py`,
`scripts/prepare_pgk2_corrected_pairs.py`,
`scripts/submit_pgk2_graph_logica_v2_diagnostic.slurm`, and
`tests/test_pgk2_audit_repairs.py`. Source files preceding the repair are backed
up remotely in `artifacts/pgk2_graph_logica_v2_diagnostic/pre_repair_source.tar.gz`.

## Full-data RDKit graph rebuild (2026-09-17)

The small-pair Graph LogiCA result (`9780845`, zero validation hits) is no
longer the training direction.  The new path uses the complete released DEL
table rather than the 308-molecule competition set or the 11,953 matched-SAR
pairs.

A 100k RDKit ETKDG pilot completed as McCleary job `10891645` in 1h17m59s.
It produced 99,657 unique graphs at 22.0 graphs/s with no job errors, but only
3 force-field optimizations converged; 99,641 hit the 50-iteration limit.
This established that full-library MMFF/UFF optimization is wasteful.  The
packed-shard integrity audit passed for metadata, node/edge offsets,
coordinates, and fingerprints.

The approved full-data design is therefore:

- canonicalize all 7,703,070 selection rows and aggregate duplicate canonical
  SMILES;
- exclude all validation/test structures by canonical SMILES;
- retain target, active-site-inhibitor competition, selection NTC,
  supplemental NTC, z-score, and historic-hit observations without converting
  low-count rows into hard negatives;
- construct one topology-only RDKit atom/bond graph for every unique training
  molecule;
- retain the fixed two-conformation human PGK2 pocket ensemble and contextual
  ESM representation of the complete 417-residue sequence;
- compare topology-only graphs with unoptimized ETKDG graphs on the identical
  deterministic evidence-enriched, scaffold-split ablation subset;
- generate ETKDG conformers for the full library only if the held-out ablation
  demonstrates benefit.

New scripts:

- `scripts/build_pgk2_canonical_index.py`: restartable process-parallel
  canonicalization, canonical leakage exclusion, duplicate aggregation,
  separate supplemental-NTC fields, deterministic scaffold splits, and the
  fixed ablation-subset definition (`informative OR sample_bucket < 14`).
- `scripts/precompute_pgk2_graph_shard.py`: array-safe packed graph generation
  with explicit `2d` and `etkdg` modes.  In `2d` mode no coordinates or
  radius edges are stored; ETKDG coordinates are explicitly ligand-internal,
  never protein-aligned poses.
- `scripts/submit_pgk2_canonical_index.slurm`,
  `scripts/submit_pgk2_full_2d_graphs.slurm`, and
  `scripts/submit_pgk2_etkdg_ablation_graphs.slurm`.

All 32 local project tests pass, and local and McCleary runtime smoke tests
passed.  Remote source/data SHA-256 hashes were verified after upload.

Active McCleary dependency chain:

- `10901703`: canonical full-data index, running on 10 CPUs;
- `10901708`: 64-shard full topology-only graph array, held `afterok:10901703`
  with at most 16 concurrent tasks;
- `10901709`: 8-shard unoptimized-ETKDG ablation array, held
  `afterok:10901703`.

Completion update: `10901703` finished successfully in 10m16s.  After removing
28,214 blank selection rows and the two canonical structures overlapping the
challenge panels, the index contains 7,487,567 unique training molecules:
5,951,851 train, 752,921 dev, and 782,795 scaffold holdout.  There are 326,716
informative molecules and 426,912 molecules in the fixed ablation subset.

The full 2D array `10901708` also completed successfully.  All 64 reports are
present and cover one contiguous range of exactly 7,487,567 molecules with
7,487,567 `ok` statuses and no nonempty error logs.  The packed output contains
323,985,919 atom nodes and 711,936,672 directed covalent-bond edges, stores no
coordinates, and occupies approximately 2.5 GB (2,122,468,991 graph bytes plus
493,037,003 metadata bytes).  The eight `10901709` ETKDG ablation tasks were
still running normally with all ten requested CPU cores active and no error
logs at the latest check.

The first attempt (`10901352`, with dependent jobs `10901358` and `10901359`)
was deliberately cancelled before writing canonical parts because live CPU
telemetry showed that Python threads used only one core.  Canonicalization and
graph construction now use process workers.  The replacement job wrote its
first three 100k-row canonical parts without errors and used roughly 7--8 CPU
cores on the first live check.

ETKDG completion update: all eight tasks of `10901709` completed successfully
in 16m52s--18m40s with no nonempty error logs.  The eight contiguous shards
cover all 426,912 ablation molecules with 426,912 `ok` statuses.  Geometry
routes were 422,352 normal unoptimized ETKDG, 4,516 random-coordinate ETKDG,
and 44 deterministic 2D fallbacks.  The packed ensemble contains 18,144,433
atom nodes and 204,476,080 directed ligand edges, occupies approximately
1.1 GB, and achieved 49.03--54.41 graphs/s per ten-CPU shard.  The input data
for the controlled 2D-versus-ETKDG model comparison are therefore complete;
no full-library ETKDG run has been authorized or launched.

Full-data ablation training update: `src/logica_binding/full_graph_model.py`
and `scripts/run_pgk2_full_graph_ablation.py` implement a packed, vectorized
multitask trainer.  The complete 417-residue sequence is encoded once with
frozen ESM-2; its contextual residue embeddings are injected into both fixed
compound-21/47 pocket graphs.  Ligand atoms can cross-attend only to the 26
direct-pocket nodes, and no protein--ligand distances are used.  Nine
continuous DEL targets retain PGK2, inhibitor-competition, both NTC sources,
historic-hit, and z-score information.  Models are selected on the development
split; scaffold holdout is evaluated only after selection.

The four preregistered conditions are Morgan fingerprint, ligand-only 2D GNN,
pocket-conditioned 2D GNN, and pocket-conditioned ETKDG GNN, each with seeds
2026--2028.  All 34 local tests pass.  A bounded real-data A5000 smoke job
`10903088` completed in 31 seconds, trained 3,840 rows, ran development and
holdout evaluation, used 3.3 GB peak host memory, and produced a checkpoint and
report without errors.  The earlier smoke `10903075` failed before model load
because it referenced artifact-copy pocket paths absent remotely; SHA-256
checks proved the canonical `data/structures/cache7` copies are byte-identical,
and the corrected job passed.

The complete 4-condition x 3-seed matrix was submitted as McCleary array
`10903103`, limited to four concurrent A5000 tasks on the regular `gpu`
partition with a six-hour ceiling.  It explicitly requested A5000, not A100.
All 12 tasks completed successfully, all 12 reports are present, and no
non-empty error logs were produced.

Three-seed scaffold-holdout results (mean +/- sample SD) were:

- Morgan fingerprint: loss 0.140042 +/- 0.002817, active-score Spearman
  0.523260 +/- 0.008376, competition ROC AUC 0.971985 +/- 0.006485.
- Ligand-only 2D GNN: loss 0.146947 +/- 0.001249, active-score Spearman
  0.450980 +/- 0.025931, competition ROC AUC 0.958518 +/- 0.008682.
- Pocket-conditioned 2D GNN: loss 0.145918 +/- 0.000674, active-score
  Spearman 0.465614 +/- 0.012512, competition ROC AUC 0.959653 +/- 0.006551.
- Pocket-conditioned ETKDG GNN: loss 0.145895 +/- 0.000719, active-score
  Spearman 0.453138 +/- 0.002733, competition ROC AUC 0.958846 +/- 0.001686.

The pocket adds a small improvement over the ligand-only 2D GNN in continuous
active-score ranking, but neither graph model beats the Morgan baseline.  ETKDG
does not improve over the topology-only pocket model, so these results do not
justify generating 3D conformers for the full 7.49-million-molecule library.
Competition average precision is not a useful discriminator here because the
holdout classification slice contains 1,219 competition-sensitive compounds
and only 20 competition-retained compounds; its approximately 0.999 values are
therefore dominated by class imbalance.

Production Morgan refit update: the ablation winner was not submitted directly.
McCleary array `10905113` refit the fingerprint multitask architecture with
seeds 2026--2028 on every valid molecule in the leakage-safe training split:
5,951,851 molecules per epoch, with 752,921 development molecules used for
checkpoint selection and 782,795 scaffold-holdout molecules evaluated only
after selection.  All three A5000 tasks completed without error in 4--5
minutes.  Mean holdout active-score Spearman was 0.510378 +/- 0.005660 and
competition ROC AUC was 0.896992 +/- 0.012797.  This full-panel result is not
directly comparable to the evidence-enriched ablation holdout because the row
populations differ.

The three-seed production ensemble scored all 244,328 organizer-provided
validation compounds.  The audit found exact panel coverage, 50 unique output
IDs, correct descending scores, CRLF formatting, 46 Murcko scaffolds in the top
50, and no overlap with any earlier submitted top 50.  It overlaps the
unsubmitted ablation-ensemble diagnostic in 4/50 positions.  The blind test was
not scored.  Submission-ready output:

- `outputs/pgk2_validation_submission/full_signal_morgan_production_validation_submission.txt`
- SHA-256 `9d54cef887ab5cfa2f67a128b3ece31324a4dbb29a538502cb3beb11b8b9efc3`
- full ranking and report under
  `artifacts/pgk2_full_signal_fingerprint_production_validation/`

The formatted file was submitted to Blind validation as submission `9780853`.
The organizer reported `N_hits: 0`.  This falsifies the full-data Morgan
multitask transfer strategy despite training on the complete leakage-safe DEL
training split and despite its internal scaffold-holdout performance.  Do not
blend, diversify, or lightly rerank this output into another submission.  The
result strengthens the conclusion that DEL-derived internal ranking metrics are
not presently predictive of the downstream ASMS kinase-inhibition endpoint.

## Bottom line

We are participating in the DREAM × CACHE Target 2035 challenge for human PGK2. The endpoint is catalytic active-site kinase inhibition, not generic PGK2 binding.

Current DEL-derived models have not transferred reliably to blinded ASMS/kinase-assay validation. Do not submit the blind test. The competition-refined model explicitly contrasts active-site-sensitive and competition-retained PGK2 binders, but its validation submission returned zero hits. A clean implementation of the official LogiCA DTI score produced a substantially different retrieval head and also returned zero hits.

The primary current direction is the active-site-competition rank plus a human PGK2 compound-21/47 inhibitor-pocket ensemble. PGK2-versus-PGK1 selectivity is ancillary because the DREAM endpoint is kinase inhibition, not selectivity alone.

## Data and strict boundaries

- PGK2_selection.parquet: 7,703,070 released DEL rows. Fields include target, target-plus-inhibitor, NTC reads, z-scores, and historic-hit count.
- PGK2_NTC_supplement.parquet: supplemental NTC observations.
- DREAM_challenge_2026/Val-Test-set.zip: 244,328 unlabeled ASMS validation compounds and 184,632 unlabeled ASMS test compounds.
- data/proteins/PGK2_P07205.fasta: human PGK2 sequence, UniProt P07205.

All 428,960 validation/test SMILES were excluded before all DEL pair builds. Portal feedback is aggregate only and must never become compound labels. The blind test is untouched.

The original screen is described as roughly 890M DEL compounds, but the usable selection release is 7.7M rows. It is extremely imbalanced: most rows have one target read. DEL and ASMS/kinase assay are different assays.

Observed inhibitor/NTC reads provide control evidence against specificity. A released zero is not proof of specificity; the NTC supplement was used to recheck positives.

## Original DEL labels and models

scripts/prepare_pgk2_strict_hits.py exact-SMILES-deduplicates DEL rows, sums counts, Stouffer-combines z-scores, takes maximum historic-hit count, and applies:

    count_PGK2 > 3
    inhibitor == 0 OR inhibitor < 0.1 * target
    NTC == 0 OR NTC < 0.1 * target
    historic_hits < 5

This produced 13,184 strict positives; NTC-supplement rechecking left 13,176 eligible anchors. Artifact: artifacts/pgk2_training/strict_positive_hits.parquet.

The original matched-SAR builder, scripts/build_pgk2_matched_sar_pairs.py, produced 11,953 pairs. A comparator had to be in the same library, share two of three building blocks, have Morgan Tanimoto >= 0.60, show observed inhibitor or NTC counter-signal, and be directionally worse in target/control evidence.

LogiCA's base encoders remained frozen; adapter/cross-attention layers were fine-tuned for five epochs. This matched-SAR model is the only variant with a nonzero validation result, but that result is borderline.

## Blind-validation history

All entries below used Blind validation. None used Blind test.

| Submission | Method | Result |
|---:|---|---|
| 9779037 | Broad-pair LogiCA | 0 hits |
| 9779038 | Matched-SAR LogiCA | 1 hit, 1 cluster, p = 0.05201186181800366 |
| 9779039 | Matched-SAR fingerprint/logistic baseline | 0 hits |
| 9779051 | Matched-SAR one-per-inferred-series portfolio | 1 hit, 1 cluster, same p |
| 9780724 | SELFormer + Morgan dual-gated LogiCA | 0 hits |
| 9780732 | Active-site competition-refined LogiCA | 0 hits |
| 9780734 | Official-score LogiCA baseline | 0 hits |
| 9780853 | Full-data, three-seed full-signal Morgan multitask ensemble | 0 hits |

Validation uploads contain exactly 50 unique CatalogIDs, one per line, no header, no scores, no SMILES.

The original matched-SAR top 50 was already diverse. A Kruger-inspired UPGMA/Morgan/MCS audit found 613 inferred series in its top 1,000 and 43 inferred series in the raw top 50. A one-per-series cap did not improve validation. The implementation is not the organizer's exact cluster algorithm.

## Structural active-site pilot

The catalytic cleft is the union of ATP/ADP and 3-PG subsites. scripts/derive_pgk2_active_site.py produced artifacts/pgk2_structural_pilot/active_site_union.json with 36 zero-based human residues:

    [23,25,62,64,122,128,166,167,169,170,172,212,213,214,215,236,237,238,
     239,241,254,255,256,291,311,312,313,314,338,339,340,341,342,343,344,374]

ATP/ADP contacts came from the CACHE high-resolution human PGK2–compound-47 co-crystal. 3-PG contacts came from mouse PDB 2PAA and were mapped to human PGK2.

Ten diverse, non-quinazoline candidates from the original matched-SAR ranking were evaluated with Boltz-2.1 in two approved conditions ($0.50 each):

1. Human PGK2–compound-47 template.
2. Mouse 2PAA ATP + 3-PG template, using the same human PGK2 sequence and the same combined-cleft constraint.

All 20 predictions completed. At a 5 Å contact cutoff, every pilot pose contacted ATP/ADP-side residues and none contacted a 3-PG-pocket residue in either template. This supports ATP-side placement consistency for this small pilot only; it neither proves inhibition nor excludes 3-PG-site inhibitors.

Relevant artifacts:

- artifacts/pgk2_structural_pilot/boltz_pilot_results.csv
- artifacts/pgk2_structural_pilot/boltz_2paa_results.csv
- artifacts/pgk2_structural_pilot/template_consensus.csv
- scripts/prepare_pgk2_2paa_boltz_inputs.py
- scripts/summarize_pgk2_boltz_pilot.py
- scripts/compare_pgk2_boltz_templates.py

The highest structural-consensus diagnostics among the ten were Z2902574654, Z319092024, Z1325235222, Z2286322095, and Z1139198969. They are not confirmed hits and were never submitted as a structural-only portfolio.

## Active-site competition refinement

The corrected DEL objective uses inhibitor competition as the label-defining contrast. It does not treat all PGK2-enriched molecules as positives.

`scripts/build_pgk2_competition_pairs.py` aggregates exact SMILES, uses the NTC supplement as additional background evidence, excludes every validation/test SMILES, and defines:

- competition-sensitive anchors: PGK2 count >= 4, inhibitor/PGK2 < 0.1, NTC/PGK2 < 0.1, historic hits <= 4;
- competition-retained binders: the same target/background criteria but inhibitor/PGK2 >= 0.5;
- primary matched contrasts: same library, two shared building blocks, Morgan >= 0.60, target counts within twofold;
- bounded fallback contrasts: same library, Morgan >= 0.50, target counts within twofold.

This produced 279 pairs covering 173 competition-retained binders: 213 two-building-block pairs and 66 same-library fingerprint fallbacks. Pair Tanimoto q10/median/q90 was 0.548/0.667/0.796, and target-count fold difference was 1.00/1.25/1.50. Artifact: `artifacts/pgk2_competition_pairs/pgk2_competition_pairs.parquet`.

McCleary job `10868073` initialized from the original matched-SAR adapter and refined it for five epochs on these 279 contrasts. It completed successfully on an RTX A5000. The scaffold-held-out DEL pair win rate was 0.7632 (29/38); this is a DEL diagnostic, not kinase-assay performance.

McCleary array `10868104` then scored all 244,328 validation compounds in 25 chunks. Both workers completed successfully, and `scripts/merge_pgk2_panel_scores.py` verified exact, unique panel coverage. The primary competition-refined ranking is:

- `artifacts/pgk2_competition_validation_scores/validation_ranked_scores.csv`
- `artifacts/pgk2_active_site_shortlist/validation_structural_shortlist.csv`

Compared with the original matched-SAR ranking, Spearman correlation is 0.9045, but top-50 and top-100 overlap are only 42%. This shows that the competition correction materially reorders the retrieval head. Because the refined adapter already starts from the matched-SAR checkpoint, this competition-refined rank was treated as primary; a separate 1.0 competition + 0.5 matched-SAR blend is retained only as a sensitivity analysis under `artifacts/pgk2_active_site_blended_shortlist/`.

The primary top 50 was submitted to Blind validation as submission `9780732`. The organizer reported `N_hits: 0`. This falsifies that particular competition-refined retrieval head; it does not establish that inhibitor competition is irrelevant, but it shows that this ligand-only transfer formulation is insufficient.

`scripts/derive_pgk2_inhibitor_pocket.py` defines the primary structural region from the union of residues within 6 A of the crystallographic compound-21 and compound-47 ligands. It contains 26 direct-edge residues. A broader 8 A protein context shell contains 49 residues, and the mapped 3-PG subsite contributes 11 auxiliary residues. The exact audited definition is `artifacts/pgk2_inhibitor_pocket/inhibitor_pocket.json`.

`scripts/prepare_pgk2_inhibitor_ensemble_boltz_inputs.py` prepared 200 inputs for the primary top 100: each compound is evaluated against both human PGK2 inhibitor-bound conformations, using only the 26-residue direct pocket. These inputs are stored in `artifacts/pgk2_inhibitor_ensemble_inputs/`. They have not been submitted because Boltz prediction is a separately costed step.

## Official LogiCA code audit and Graph LogiCA status

The official LogiCA source archive is now available as `logica-main.zip` (archive commit `b6d6c2efcf21ac7ea854e4f78f37d3b2047e06f7`, SHA-256 `df80bd3e90c40eaa01ac3d895fced87f71ae079a47a313705be47b9e64456804`). Its 20 unit tests pass locally.

The official implementation is sequence-only ESM-2 + SELFormer with bidirectional cross-attention; it contains no GNN, protein contact graph, ligand atom graph, coordinates, or ligand-protein spatial edges. Therefore Aurelien's Graph LogiCA proposal remains untested.

Our reconstructed scorer is checkpoint-compatible at the pretrained adapter layers, but it is not the official downstream scoring recipe. Important differences are:

- official DTI scoring uses absolute bidirectional conditional token log-likelihood across non-special tokens and learns the protein/drug mixing weight;
- our scorer uses a fixed 0.5 mixing weight and deterministic masked-token context gain relative to unconditional logits;
- official DTI fine-tuning uses encoder LoRA by default, 40 epochs in the BindingDB recipe, and Bradley-Terry negative sampling;
- our PGK2 runs froze both language encoders and trained only the adapter for five epochs on hand-built DEL pairs.

On a bounded 30-compound diagnostic sample, pretrained official and reconstructed scores had Spearman correlation 0.823. This is only a smoke comparison, but it confirms that the ranking functions are not interchangeable. Past submissions remain valid results for the reconstructed models, but they should not be described as exact executions of the official downstream LogiCA recipe.

The official-score compatibility layer and the first Graph LogiCA implementation are now present locally:

- `src/logica_binding/model.py` supports the official unmasked conditional-token likelihood score, learned protein/drug mixing weight, and official partial-identity projection initialization while retaining the historical masked-gain mode for reproducibility.
- `scripts/run_pgk2_pairwise_finetune.py` can train the interaction adapter with that official score and an explicit pocket readout. It still freezes the two language-model encoders, so this is an official-score baseline, not yet the official downstream LoRA recipe.
- `src/logica_binding/graph_model.py` implements node-preserving residue and ligand graph encoders, residue-to-ESM injection, atom-to-SELFIES-token attention, and bidirectional pose-contact messages. Protein graph injection can only touch mapped pocket tokens. Ligand-protein edges can only reach the 26 direct inhibitor-envelope residues.
- A bare SMILES is explicitly rejected for contact-edge construction. Candidate contacts require a pose in the same coordinate frame as its protein structure. Boltz CIF parsing verifies the complete ordered ligand element sequence against the supplied SMILES before attaching coordinates.
- `scripts/build_pgk2_pocket_graphs.py` built the compound-21/47 protein-template ensemble under `artifacts/graph_logica/pocket_graphs/`: 49 context nodes, 26 direct-contact nodes, and 676 directed protein edges per template.
- `scripts/build_pgk2_pilot_complex_graphs.py` converted the 20 already-completed pilot poses into aligned protein, ligand, and bipartite-contact graphs under `artifacts/graph_logica/pilot_complex_graphs/`. The compound-47 condition has 48--94 contact edges over 9--17 pocket residues per candidate; the ATP/3-PG-template condition has 57--88 edges over 13--16 residues. These ten candidates have no kinase-assay labels, so this is an input-integrity/architecture pilot rather than Graph LogiCA validation.
- The full project test suite now contains 23 passing tests, including coordinate/atom-order checks and the invariant that only direct-pocket residues receive ligand contacts.

Graph LogiCA is therefore implemented at the architecture and data-interface level but remains untrained. A scientifically meaningful training run still requires pose-derived graphs for labeled or defensibly pseudo-labeled training pairs; the existing ten-pose pilot is too small and unlabeled. The official archive itself supplies no graph components.

Update after clarifying the intended WALLE-style construction with Aurelien: candidate complex poses are not required. The primary Graph LogiCA path now runs ESM-2 over the complete 417-residue PGK2 sequence, injects those contextual embeddings into the 49 structure-derived pocket nodes, builds deterministic ligand-internal RDKit ETKDG graphs, and permits learned ligand edges only to the 26 direct inhibitor-interface residues. Cross edges carry no protein--ligand distance because the coordinate frames are independent. The compound-21 and compound-47 PGK2 conformations form a two-template ensemble.

`scripts/run_pgk2_graph_logica_finetune.py` implements a two-stage-compatible pairwise trainer, including optional initialization from a broader Graph LogiCA adapter. `scripts/run_pgk2_graph_logica.py` scores any validation/test slice directly from SMILES using the same RDKit graph construction; no docking or Boltz prediction is required. `scripts/submit_pgk2_graph_logica_train.slurm` specifies three epochs of matched-SAR pretraining on 11,953 pairs followed by 15 epochs of active-site competition refinement on 279 pairs. All 26 local tests pass, including an explicit check that pocket graph nodes change when their full-sequence ESM embeddings change, and bounded end-to-end training and inference smoke runs pass.

The full two-stage GPU fit completed successfully as McCleary job `10882169` on an RTX A5000 in 19m36s. Stage 1 used 11,953 matched-SAR pairs over 9,820 unique molecules; its weighted pairwise loss fell from 0.4286 to 0.2296 over three epochs and its scaffold-held-out pair win rate was 0.8545 (2,237/2,618). Stage 2 initialized from that checkpoint and used 279 active-site competition pairs over 308 molecules; its loss fell from 0.7326 to 0.0042 over 15 epochs. The scaffold-held-out competition-pair win rate was 0.8421 (32/38). These are internal DEL-pair diagnostics, not kinase-assay validation evidence, and the very low stage-2 training loss indicates that overfitting remains a material risk. The verified local artifacts are:

- `artifacts/pgk2_graph_logica_pretrain/graph_logica_adapter.pt`
- `artifacts/pgk2_graph_logica_pretrain/report.json`
- `artifacts/pgk2_graph_logica_competition_finetune/graph_logica_adapter.pt`
- `artifacts/pgk2_graph_logica_competition_finetune/report.json`

Full validation scoring with this trained graph model completed on McCleary in 31 bounded chunks. The strict merge audit verified exactly 244,328 unique validation CatalogIDs with no missing or extra compounds. Rare ETKDG failures use a tested deterministic 2-D coordinate fallback rather than aborting a panel; all ordinary molecules retain the training-time ETKDGv3 plus MMFF/UFF construction. Verified local outputs are:

- `artifacts/pgk2_graph_logica_validation/merged/validation_ranked_scores.csv`
- `artifacts/pgk2_graph_logica_validation/merged/validation_top50.csv`
- `artifacts/pgk2_graph_logica_validation/merged/validation_merge_report.json`
- `outputs/pgk2_validation_submission/graph_logica_validation_submission.txt`

The Graph LogiCA top 50 contains 45 Murcko scaffolds, with no scaffold represented more than three times. Its top-50 overlap is 1 compound with the clean official-score baseline, 1 with matched-SAR LogiCA, 1 with the competition LogiCA ranking, 0 with the fingerprint baseline, and 4 with SELFIES dual-gated LogiCA. The formatted ranking was submitted to validation as submission `9780845`; the organizer reported `N_hits: 0`. This falsifies the present WALLE-style, fixed-pocket Graph LogiCA transfer strategy despite its strong internal DEL pairwise diagnostics and chemically distinct top 50. Do not blend or perturb this ranking into another validation entry merely because it is diverse. The blind test remains untouched.

### Official-score baseline run

The clean adapter-only official-score baseline has now completed. McCleary job `10874875` started from the released LogiCA pretraining checkpoint, trained for 20 epochs on the 279 active-site competition pairs, used unmasked own-token conditional likelihood across all 417 protein residues and all valid drug tokens, and learned the protein/drug mixing weight. It did not initialize from any prior PGK2 adapter. The job completed on an RTX A5000 in 40 seconds. The held-out scaffold pair win rate was 0.7105 (27/38), and pair alpha remained 0.49995. This is a DEL training diagnostic, not kinase-assay performance.

McCleary array `10874974` then scored the full validation panel in 25 chunks. Both workers completed successfully. The merge audit verified exactly 244,328 unique validation CatalogIDs with no missing or extra compounds. Local outputs are:

- `artifacts/pgk2_official_score_finetune/finetuned_adapter.pt`
- `artifacts/pgk2_official_score_finetune/report.json`
- `artifacts/pgk2_official_score_validation/merged/validation_ranked_scores.csv`
- `artifacts/pgk2_official_score_validation/merged/validation_top50.csv`
- `artifacts/pgk2_official_score_validation/comparison_report.json`
- `outputs/pgk2_validation_submission/official_score_logica_validation_submission.txt`

This retrieval head is materially independent of the earlier submissions: versus matched-SAR it has Spearman 0.1786 and zero top-50 overlap; versus competition-refined masked gain it has Spearman 0.2423 and zero top-50 overlap. The official-score top 50 contain 48 Murcko scaffolds. A pretrained-versus-fine-tuned control on 1,000 compounds had Spearman 0.4154 and only 5/50 top overlap, so the result is not merely the unchanged frozen-model ranking. A property probe found no gross molecular-weight, heavy-atom, or logP rank bias.

The formatted file was uploaded to Blind validation as submission `9780734`. The organizer reported `N_hits: 0`. This falsifies the clean official-score ligand-only retrieval head despite its low correlation and zero top-50 overlap with the earlier failed rankings. It must not be blended into another candidate list merely because it is diverse. It is still not Graph LogiCA.

## SELFormer/SELFIES dual-gated experiment

This implemented the suggestion to use chemically close SELFIES-space neighbors with different DEL behavior.

1. scripts/prepare_pgk2_selfies_neighbor_pool.py made a DEL-only pool of 43,523 molecules: 13,176 rechecked strict-positive anchors and 30,347 comparators with observed inhibitor/NTC counter-signal, across 10 libraries.
2. scripts/embed_pgk2_selfies_pool.py used frozen HUBioDataLab/SELFormer: SMILES -> SELFIES -> mean-pool non-special tokens -> L2-normalized 768D embeddings.
3. scripts/mine_pgk2_selfies_neighbors.py retrieved same-library nearest neighbors with directional, control-confirmed DEL preferences.

The raw cosine-only version was too broad: 26,332 pairs, with Morgan q10/median/q90 = 0.286/0.465/0.696. Only 22.1% had Morgan >=0.60, so it was not fine-tuned.

The dual gate added Morgan >=0.60. It made 10,222 pairs from 6,497 anchors. SELFormer cosine q10/median/q90 was 0.974/0.986/0.995; Morgan was 0.609/0.667/0.778.

Local artifacts:

- artifacts/pgk2_selfies_neighbor_pool/selfies_neighbor_pool.parquet
- artifacts/pgk2_selfies_embeddings/embeddings.npy
- artifacts/pgk2_selfies_embeddings/metadata.parquet
- artifacts/pgk2_selfies_neighbor_pairs_dualgated/pgk2_selfies_neighbor_pairs.parquet

McCleary job 10841628 fine-tuned LogiCA for five epochs on the 10,222 dual-gated pairs. It completed in 2m52s on an RTX A5000. Loss fell from 0.4782 to 0.1241 and the scaffold-held-out DEL pair win rate was 0.8774. These are DEL diagnostics, not ASMS metrics.

Remote checkpoint:

    ~/pgk2_logica/artifacts/pgk2_selfies_dualgated_finetune/finetuned_adapter.pt

McCleary array 10842015 scored all 244,328 validation compounds using two A5000 workers and 25 bounded 10k chunks. It completed in under eight minutes. Its top 50 had only 3 molecules in common with the original matched-SAR top 50 and had 45 Murcko scaffolds, with the largest family containing 3 molecules.

The formatted submission was outputs/pgk2_validation_submission/selfies_dualgated_logica_validation_submission.txt. Submission 9780724 scored 0 hits. Conclusion: the local SELFormer + Morgan DEL pseudo-label strategy did not transfer to ASMS/kinase retrieval.

## McCleary notes

- SSH alias: ssh mccleary.
- The SSH key is accepted, then Yale requests keyboard-interactive Duo. Use a real terminal with ssh -tt and select 1 for Duo Push. Noninteractive SSH cannot answer the prompt.
- Remote project root: ~/pgk2_logica.
- gpu_devel provided immediate RTX A5000 jobs. Observed QOS limit: two submitted jobs per user and 10 CPU / 2 GPU TRES per user.
- Job 10840725 was an old queued A100 request and was intentionally cancelled after switching to immediate A5000 job 10841096.

Useful Slurm scripts:

- scripts/submit_pgk2_selfies_embeddings.slurm
- scripts/submit_pgk2_selfies_dualgated_finetune.slurm
- scripts/submit_pgk2_selfies_dualgated_validation_scoring.slurm

## Next recommendation

1. Do not submit Blind test.
2. Stop submitting ligand-only DEL pseudo-label variants. Broad-pair, fingerprint, SELFIES/Morgan, and active-site competition refinement have all returned zero hits; the two matched-SAR submissions recovered the same single borderline cluster.
3. Do not submit the untested 1.0 competition + 0.5 matched-SAR blend. It reuses the same failed signals and is not an independent hypothesis.
4. If continuing, require genuinely independent structural or experimental evidence before another validation attempt: trained Graph LogiCA with pose-derived examples, experimentally characterized public PGK2 inhibitors, a prespecified compound-21/47 two-template structural consensus, or a calibrated kinase-inhibition training set. Do not use PGK1 selectivity as the primary filter.
5. Preserve the Blind test until a new method produces nonzero validation evidence. If no independent signal can be built, stop rather than spend additional submissions on rank perturbations.

## Selectivity-prior production run

This section records a completed historical experiment. It is not the primary objective after clarifying that DREAM rewards PGK2 kinase inhibition rather than PGK2-over-PGK1 selectivity.

The first structure-led pipeline stage is implemented and has been run over
the full blind-validation candidate panel:

- `data/proteins/PGK1_P00558.fasta` is the canonical human PGK1 sequence.
- `scripts/run_pgk2_pgk1_logica.py` scores PGK2 and PGK1 with identical
  deterministic masked residue positions and reports the PGK2-minus-PGK1
  context-gain delta.
- `scripts/merge_pgk2_selectivity_scores.py` audits complete chunk coverage.
- `scripts/build_pgk2_selectivity_shortlist.py` performs label-free late rank
  fusion, excludes quinazolines, enforces a heavy-atom bound, and applies a
  scaffold cap before costly structural screening.
- `scripts/submit_pgk2_selectivity_validation_scoring.slurm` provides the
  two-worker McCleary validation array.
- The official CACHE PGK2 compound-21/47 and PGK1 compound-45 coordinate files
  are stored under `data/structures/cache7/`; their SHA-256 digests were
  recorded at acquisition time. `scripts/derive_pgk_selectivity_priors.py`
  extracts selectivity-site distances, active-site contacts, and bounded
  crystallographic-water bridges as geometry priors.
- `scripts/prepare_pgk_selectivity_boltz_inputs.py` creates matched PGK2/PGK1
  inputs from a verified shortlist but deliberately does not submit paid jobs.

The sequence delta is a prioritization feature, not biochemical selectivity.
Verified non-quinazoline references and PGK2-versus-PGK1 pose features can be
added without retraining LogiCA or using portal feedback.

McCleary array `10860435` scored all 244,328 validation molecules against
PGK2 and PGK1 with the same matched-SAR LogiCA adapter, protein-mask key, and
chunk boundaries. Both array workers completed successfully (25 chunks total),
and the audited merge retained exactly 244,328 unique rows. The raw
PGK2-minus-PGK1 context-gain delta ranged from -0.9389369488 to 1.14375639.

McCleary job `10860885` then applied the prespecified label-free rank fusion:

    matched-SAR rank          weight 1.00
    PGK2 LogiCA rank          weight 0.25
    PGK2-minus-PGK1 rank      weight 1.00

All three sources covered every validation molecule. The structural shortlist
excluded 3,571 quinazolines, used a heavy-atom limit of 49, and retained 1,000
molecules across 925 Murcko scaffolds. Relative to the previous matched-SAR
ranking, the fused top 50 has 0 molecules in common with the previous top 50,
0 in common across the respective top 100s, and 82 in common across the
respective top 1,000s. Its top 50 contain 50 distinct Murcko scaffolds. These
statistics establish novelty and diversity only; they do not establish
inhibition or PGK2 selectivity.

Downloaded production artifacts:

- `artifacts/pgk2_selectivity_logica/merged/validation_ranked_scores.csv`
- `artifacts/pgk2_selectivity_shortlist/validation_fused_ranked.csv`
- `artifacts/pgk2_selectivity_shortlist/validation_structural_shortlist.csv`
- `artifacts/pgk2_selectivity_shortlist/validation_report.json`

Paired PGK2/PGK1 Boltz inputs for the top ten shortlist molecules were run as
20 predictions. All 20 jobs completed and downloaded successfully. They used
the PGK2 compound-47 and PGK1 compound-45 templates, one sample per complex,
and the same 36-residue pocket definition.

`Z2385710715` was the only compound with positive PGK2-minus-PGK1 deltas for
all three reported diagnostics: binding confidence (+0.0295), optimization
score (+0.0169), and ligand ipTM (+0.0570). Its ligand was also closer to PGK2
Y242 (4.07 A) than PGK1 F242 (7.48 A). However, its absolute PGK2 binding
confidence was only 0.0975. `Z1754551061` had the highest absolute PGK2 binding
confidence (0.3025), but PGK1 was nearly as high (0.2854), so it did not show a
strong selectivity diagnostic. The other pairs had mixed or PGK1-favoring
signals.

Results and extracted structures:

- `artifacts/pgk_selectivity_boltz_summary/paired_summary.csv`
- `artifacts/pgk_selectivity_boltz_summary/per_run_metrics.csv`
- `artifacts/pgk_selectivity_boltz_summary/structures/`
- `scripts/summarize_pgk_selectivity_boltz.py`

These are constrained single-sample model diagnostics, not measured affinity,
inhibition, or selectivity. The evidence is too sparse to claim that the full
fused top 50 is structurally supported. This portfolio has not been submitted
to the challenge validation or test queues.

## Guardrails

- Never include validation/test molecules in DEL-pair training.
- Never turn portal aggregate results into individual labels.
- Never describe DEL pair wins, pair loss, Boltz confidence, or pocket contacts as ASMS/kinase-assay performance.
- Never claim the Kruger-inspired clustering exactly matches the evaluator.
