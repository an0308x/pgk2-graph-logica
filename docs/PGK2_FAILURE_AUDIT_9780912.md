# Diagnostic audit after user-reported zero hits: 9780912

Date: 2026-09-19. Scope: read-only scientific diagnostics and separate diagnostic
artifacts; no retraining, relabeling, candidate reranking for submission, or
Synapse upload. User approved proceeding after reporting `N_hits: 0`.

## Inputs and provenance

- Selected adapter: `artifacts/pgk2_dev_v4_cap5_graph_aux0/best.pt`, step 2198.
- Adapter SHA-256: `409b88118952b3ea327a3244fa279b9f4400ea16064a0bc0c91768d282e7e91d`.
- Delivered top-50 TXT SHA-256: `c07f3f22690c187f01db11b08bcb5badca8a9301b8b466ec18b5232d1d65029c`.
- Complete 244,328-row validation ranking and existing 64-shard training manifest.
- Production model sources and checkpoints are not modified.
- Uploaded submission bytes have not been fetched independently from Synapse.

## Prespecified diagnostic probes

Preparation seed: 9780912. Validation probe is the union of the submitted ranking's
top 1,000 and 4,096 uniformly sampled validation molecules (5,078 unique rows).
All 50 submitted molecules are included. Random and selected groups are reported
separately. Development uses the original channel-specific evaluation batches
(238 batches, 3,606 molecule visits), preserving original comparisons.

Chemical properties are computed for all 244,328 validation molecules, every
training molecule supplying any cap5 ranking confidence (27,327), and a seeded
32-per-shard training background. Nearest-neighbor Morgan/Tanimoto searches use
all ranking-supervised training molecules, not all six million training rows.
Query groups are the submitted 50, 512 sampled validation probes, and 512 sampled
unique development molecules. The positive-competition-proxy reference is a
DEL-derived subset, not experimentally verified kinase inhibitors.

## Inference-only controls

Reproduce original full scores and selected development metrics before interpreting
controls. Decompose protein and ligand log-likelihood terms; score ligand-only LM
and ligand-only graph branches; remove graph contacts; remove base cross-attention;
turn off graph conditioning while retaining base cross-attention; zero protein
context; and permute pocket-edge features using a fixed seed.

Removing graph contacts still leaves base cross-attention, and vice versa.
Zero/shuffled context is out of distribution. Sensitivity is not proof of biological
specificity; fixed-target data cannot establish cross-target specificity. Rank
overlap is measured within each probe, not a new full-library top-50 ranking.
Unmasked molecular token likelihood is not a calibrated affinity or inhibition
probability. No challenge feedback is used as a training label.

Compare development competition ordering with target-count-only and inverse
inhibitor-count-only oracles, including equal-target and low-count strata. These
oracles directly use evaluation counts and are diagnostics, not deployable models.

## Organizer clarification

The organizer states that the released inhibitor column is censored to compounds
positive in the PGK2 arm and that a comparable inhibitor-arm supplement is not
available; the previously considered additional data were a different replicate.
This limits interpretation of released count contrasts and prevents treating
released column totals as complete sequencing depths. It does not by itself
establish why this ranking returned zero hits.

Source: https://www.synapse.org/Synapse:syn75349604/discussion/threadId=15177

## Execution

- New implementation: `src/logica_binding/failure_audit.py` and
  `scripts/prepare_pgk2_failure_audit.py`, `scripts/run_pgk2_failure_audit.py`.
- All 73 local tests pass, including real-checkpoint smoke tests and exact normal
  score parity between the diagnostic decomposition and production scorer.
- CPU preparation `10943595`: 4 CPUs, 16 GB, one-hour limit; initially running.
- Dependent GPU diagnostic `10943597`: one generic GPU, 4 CPUs, 32 GB, one-hour
  limit; initially waiting on successful preparation.
- Remote outputs: `artifacts/pgk2_failure_audit_9780912`.
- Logs: `artifacts/pgk2_full_logica_v3_logs/failure-prep-10943595.{out,err}` and
  `failure-audit-10943597.{out,err}`.

Initial GPU job `10943597` scored all probes but failed the saved-score absolute
tolerance: maximum difference 1.14440918e-5 exceeded 1e-5 for one of 5,078 rows.
Median difference was 2.38418579e-7; rank correlation 0.9999999992 and all 50
selected IDs unchanged. Its score artifacts were retained locally. Diagnostic
rerun `10943727` additionally compares the decomposed scorer directly against
the unchanged production scorer on every identical batch (tolerance 1e-6), and
requires unchanged top-50 membership plus saved-score error <=2e-5. This is a
documented diagnostic tolerance adjustment, not a production scoring change.

## Completed preparation results

CPU preparation completed in 182.24 seconds. The GPU diagnostic then started.
Local raw report: `artifacts/pgk2_failure_audit_9780912/preparation_report.json`.

### Supervision and evaluation

Across the exact 4,086 original development competition comparisons:

- Target-count-only oracle: 66.8212% weighted ordering accuracy.
- Inverse-inhibitor-count-only oracle: 88.0392% (selected model: 80.1968%).
- 1,946 pairs (47.63%) prefer a molecule with mean target count at most one.
- 1,737 pairs (42.51%) compare two nonpositive competition proxies.
- Only seven comparisons have both molecules' mean target counts at least five;
  639 have at least one such molecule.

The oracles use the measurements defining the labels: these are not fair predictive
baselines or evidence that the model directly reads assay counts. They expose what
the metric rewards. Good ordering can reflect distinctions among weak evidence
without demonstrating ability to recover strong active-site inhibitors.

### Chemical transfer

| Quantity | Ranking-supervised DEL train | Challenge validation | Submitted 50 |
|---|---:|---:|---:|
| Median heavy atoms | 43 | 23 | 23.5 |
| Median molecular weight (Da) | 591.74 | 332.40 | 335.01 |
| Median calculated logP | 4.04 | 2.44 | 2.21 |

Using radius-2, 2,048-bit Morgan fingerprints and all 27,327 ranking-supervised
training molecules as reference, median nearest-training Tanimoto is 0.6955 for
512 sampled development molecules, 0.2938 for 512 random validation molecules,
and 0.3085 for the submitted 50. Similarity >=0.5 occurs for 96.09% of the sampled
development molecules, 0.39% of random validation molecules, and none of the 50.
This quantifies a large transfer gap that the internal scaffold split does not
reproduce. Similarity is representation-dependent and does not itself prove activity.

Full-validation score/property Spearman correlations are small: molecular weight
-0.0535, logP +0.0305, heavy atoms -0.0272; largest absolute tested association is
rotatable bonds -0.1409. A simple monotonic size or lipophilicity preference is not
supported as the main explanation by these diagnostics; nonlinear biases remain
possible.

## Completed GPU results

Rerun `10943727` completed successfully (90 seconds scheduler wall time; 66.56
seconds measured analysis time). The same-batch decomposition matched production
within 4.76837158e-7; saved-score maximum difference remained 1.14440918e-5 with
unchanged selected 50. All three original development-channel metrics reproduced.
The initial strict-check failure remains documented above. All 73 tests pass.
Local raw report: `artifacts/pgk2_failure_audit_9780912/report.json`.

| Inference condition | DEL competition weighted ordering accuracy |
|---|---:|
| Submitted full model | 80.1968% |
| Graph conditioning off; base cross-attention retained | 80.0944% |
| Graph atom-pocket contacts removed; base cross-attention retained | 79.4232% |
| Pocket edge features shuffled | 80.1493% |
| Base cross-attention removed; graph contacts retained | 50.2938% |
| Zero protein hidden context and pocket node/edge features | 57.3341% |
| Ligand-only raw LM | 42.4973% |
| Ligand-only graph LM | 42.1443% |

On the random 4,096-validation-molecule probe, graph-off/full rank correlation is
0.9442, retaining 38/50 top candidates *within this probe*. Shuffled pocket-edge
features have correlation 0.99999775 and retain 50/50. Zero protein context has
correlation 0.4082 and retains 4/50. Raw ligand-only likelihood has correlation
0.0359 and retains 0/50. These controls do not support a simple explanation that
the final score is just the unconditional ligand language-model likelihood.
They do show that the added graph branch has little marginal effect on the
internal aggregate ordering metric in this trained checkpoint. Graph-off rankings
are not identical, and a single OOD perturbation cannot prove structure is unused.

Both protein and ligand terms contribute score variation (random-probe weighted
standard deviations 0.0542 and 0.0497; alpha 0.50076). Protein input matters
computationally; true target specificity and physical binding interactions remain
unvalidated. There are no candidate complex poses or intermolecular distances.

The 639 comparisons with at least one mean-target-count >=5 molecule achieve
83.26% full-model accuracy versus 63.76% inverse-inhibitor oracle accuracy.
This counterexample matters: the model is not uniformly worse than that oracle,
and weak-count structure does not by itself explain all model performance.
Only seven comparisons contain two such molecules, so strong-evidence ranking
is poorly characterized by this evaluation design. Counts >=5 are an audit
stratum, not a proposed universal biochemical hit threshold.

## Interpretation and recommended next step (not executed)

The strongest supported concerns are an inhibition-surrogate evaluation that
includes many weak-count comparisons and a large DEL-to-validation chemical
transfer gap. These are measured vulnerabilities, not a proven unique cause of
zero hits. No evidence here attributes failure to choosing compounds 21/47.
The audit verifies inference relative to the trained objective; it does not
validate that objective as a kinase-inhibition predictor or authenticate the
user-uploaded file against Synapse.

Before further training/submission:

1. Redesign the evidence/evaluation protocol to distinguish target-vs-background
   enrichment from inhibitor competition, preserve count uncertainty, and report
   strong-evidence retrieval separately from broad weak-evidence pair ordering.
   Do not interpret low-count zeros as confirmed inhibition or normalize by
   censored column totals as if they were complete sequencing depths.
2. Add a chemical-transfer stress test (low training similarity and smaller
   molecules, where available), audit overlap and evidence coverage, and compare
   graph and nongraph models under matched splits/seeds. If released DEL data
   cannot furnish a meaningful ASMS-like labeled subset, report that limitation
   rather than presenting an internal metric as external validation.
3. Keep all usable molecules available for representation learning; this does
   not manufacture millions of inhibition labels. Retain the pocket prior as an
   ablation until it demonstrates added predictive value. Do not respond to this
   result by automatically generating millions of conformers or more top-50 files.

No new training or challenge submission was performed during this audit.
