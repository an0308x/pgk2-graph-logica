# Full-library Graph LogiCA v3

Status: **full training job `10927166` completed successfully in 6h11m44s**,
covering one epoch over all 6,026,213 training molecules. Development-only
baseline and supervision-audit jobs `10936708` and `10936707` completed.
No challenge scoring or submission has been performed.

Next authorized work uses the separate development runner documented in
`PGK2_DEV_V4_2026-09-19.md`, including epoch-zero/intermediate checkpoint selection
and bounded supervision/objective sensitivity tests. Do not relaunch the
historical full trainer as a substitute for those gates.

Completed audit: trained graph competition ordering 61.25%, starting graph
60.93%, starting no-graph 59.88%, and linear Morgan 73.34% on identical
development comparisons. The graph's development loss worsened after training;
Morgan orders better but has worse margin-sensitive loss. See
`PGK2_V3_BASELINE_AUDIT_2026-09-19.md` for limitations and next gates.

## Post-training baseline and supervision audit (2026-09-19)

The completed full run produced development competition weighted pair accuracy
61.246% and holdout 60.513%. These are DEL preference metrics, not kinase hit
rates. The training-time competition metric was 92.619%, but it is accumulated
over repeated batches and evolving model weights, not a matched final training
evaluation. Do not interpret that difference as a precise generalization gap.

The current baseline comparison uses **development only** and replays the same
238 evaluation batches. Conditions: trained Graph LogiCA, its seed-2026 starting
graph initialization, starting pocket-restricted LogiCA without the random graph
branch, and a zero-initialized linear Morgan-2048 ranker. The latter replays the
exact supervised training batches (754,240 molecule visits), with fixed AdamW
learning rate 0.001 and no hyperparameter search. It does not receive graph
masked-token auxiliary learning, so this is a utility baseline, not a controlled
attribution of the graph branch's causal contribution. No existing full model
is retrained or changed. The evaluator must reproduce the original trained
development results before emitting a comparison report.

The supervision audit counts train/dev exclusions at target-mean-read thresholds
2/5/10/20/50/100. It reports high-target, zero-competitor examples as descriptive
observations only, not new positive labels. Zero NTC reads are not interpreted
as proof of no background binding. Evaluation weight-only effective sample size
and concentration by compound are also reported; these are not independent
sample-size estimates or confidence intervals.

Jobs: `10936707` (CPU supervision audit), `10936708` (GPU matched baselines).
Both jobs completed with exit code zero. Output directories:
`artifacts/pgk2_v3_supervision_audit` and `artifacts/pgk2_v3_baselines`.
Reports and development scores have been downloaded locally. All 56 local tests pass.

## What is and is not being trained

This is a custom PGK2 adaptation of the repaired Graph LogiCA model. It retains
the native protein/ligand token-likelihood binding score, the released LogiCA
initial checkpoint, full-sequence frozen ESM, two human PGK2 conformations,
49 context residues, and 26 direct interface residues. Packed execution has
been tested against the repaired row-wise score and its gradients.

The model is not the earlier nine-output regression GNN. The auxiliary recipe
below is also not claimed to be the original paper's training procedure.
The [LogiCA paper](https://arxiv.org/abs/2606.18703) describes logit-space
contrastive alignment; this implementation adds a separate ligand-only
self-supervised objective for the largely unlabeled DEL library.

## Data coverage

The manifest reuses all 64 existing RDKit topology graph shards. No ligand
conformers or complexes are generated. Challenge validation/test structures
are excluded again by canonical identity (428,960 unique candidate structures).
Stereoisomers of a Murcko scaffold share the corrected deterministic v2 split.

| Split | Molecules | Any observed control | Observed competitor reads |
| --- | ---: | ---: | ---: |
| Train | 6,026,213 | 24,445 | 20,208 |
| Development | 732,325 | 2,967 | 2,433 |
| Holdout | 729,029 | 3,060 | 2,512 |
| Total | 7,487,567 | 30,472 | 25,153 |

These counts describe measurements, not confirmed inhibitors or independent
biological replicates. Development and holdout molecules never train the model.
Metadata and graph checksums are verified when shards are loaded.

Manifest SHA-256:
`10eb32cf64b5a556609a781ab55f260834aa936fea28adf129df7948b0809c7a`.
The new split counts differ from the historical index because scaffold
stereochemistry and the split hash have been corrected. Graph row alignment is
unchanged. Old diagnostic adapters do not initialize this run.

## Objectives

1. Every training-split molecule contributes to **ligand-only masked-token
   reconstruction** from its atom/bond graph. This updates the ligand graph
   branch without seeing PGK2 or assigning a binding label. Frozen SELFormer
   encodes the masked SELFIES; 15% of eligible tokens are masked deterministically.
2. Separate protein-conditioned LogiCA batches learn **relative DEL rankings**
   from observed competition, selection NTC, and supplemental NTC measurements.
   Zero controls supply no ranking supervision. They are not hard negatives,
   strong displacement labels, or imputed assay measurements.
3. Counts are averaged across duplicate source rows rather than interpreting
   duplicate records as independent replicates. Supplemental NTC uses its own
   source-row denominator. No normalized-Z columns are used.
4. For each observed-control channel, the weak ordering proxy is
   `log1p(mean_target_count) - log1p(mean_control_count)`. Pairs require a proxy
   separation of at least `log(2)`. Per-molecule reliability is
   `T/(T+20) * C/(C+5) / (1+historic_hits)`; a pair receives the lower reliability.
   Competition/selection-NTC/supplement-NTC loss weights are 2/1/1. Pair loss is
   normalized by pair count, preserving absolute confidence downweighting.

These proxies and weights are prespecified heuristics, **not calibrated
enrichment, censoring correction, or kinase-inhibition probabilities**. Unknown
sequencing depth, target-conditioned release, and DEL-to-ASMS chemical shift
remain limitations. In particular, this conservative recipe forgoes supervised
competition loss on zero-competitor rows even when some could be true inhibitors.
It does not manufacture millions of reliable assay labels from sparse data.

Default plan: one complete training epoch; auxiliary batch 64; observed-control
ranking batch 32 every four auxiliary batches, weighted to retain the intended
average contribution; auxiliary coefficient 0.01; AdamW learning rate 3e-5.
Eligible control pools are sampled within hash-distributed graph shards.
All training molecules are visited once for auxiliary learning; ranking
measurements may be revisited. These are counted separately.

## Evaluation and safeguards

Development confidence-scaled ranking loss selects the checkpoint and requires
observed competition comparisons. Holdout is evaluated only after selection.
Evaluation uses all eligible observed-control pools, in fixed batches; it is
not an exhaustive all-pairs test. Reported pair wins remain DEL diagnostics,
not kinase assay accuracy. The 128-token SELFormer cap is retained and truncation
visits are counted; full atom/bond graphs remain available.

Checkpoints record optimizer state, random state, traversal position, input
identity, and coverage counters for explicit resume. Snapshot replacement is
atomic. Completed runs are not overwritten. All new adapters remain blocked
from challenge scoring pending review, even after full training.

## Execution record

### Successful smoke and full submission

Replacement smoke `10927119` completed successfully in 53 seconds on A5000 node
`r102u33n01`. It completed 64 optimizer updates / 4,096 auxiliary molecule visits
and 512 ranking molecule visits. Finite optimization, checkpoint selection, and
supported competition comparisons were verified. Measured training throughput
was 267.47 molecules/second; peak allocated GPU memory was 1.70 GiB. Projected
training time is 6.26 hours, with an 11-hour request allowing I/O/evaluation and
performance variation. This is an extrapolation, not a completion guarantee.

The smoke's bounded dev/holdout metrics are execution diagnostics and were not
used to tune hyperparameters or claim improved hit prediction. Of 4,096 auxiliary
visits, 1,337 (32.64%) exceeded the 128-token SELFIES cap; full atom/bond graphs
were retained. The full run preserves the tested configuration, and truncation
remains a stated limitation.

Full job `10927166` uses `scripts/submit_pgk2_full_logica_train.slurm`, initialized
fresh from the released checkpoint, not from the smoke adapter. It has no
training/evaluation batch caps. Its output directory is
`artifacts/pgk2_full_logica_v3_train`; logs are
`artifacts/pgk2_full_logica_v3_logs/train-10927166.{out,err}`. The launcher checks
smoke readiness again and archives the source before training. Explicit resume
is supported by passing `--resume` to the same launcher. No challenge scoring
or submission is scheduled. All 53 tests pass, including real-model CPU resume
and the full-launch guard.

Startup verified: first optimizer step completed with finite auxiliary loss
0.15816, ranking loss 0.004625, and pre-clipping gradient norm 0.73579. No runtime
errors were present; stderr contained only a PyTorch TypedStorage deprecation
warning. Training is in progress, not completed.

Latest update after restored VPN/Duo access: original smoke `10924529` failed
before training with `ModuleNotFoundError: scipy`. Its graph I/O functions had
been imported from the older evaluation script, unnecessarily importing that
script's metrics dependencies. Graph I/O now lives in
`src/logica_binding/packed_graph_data.py`, with no SciPy dependency. All 52 local
tests pass, including the real-model integration check. The fixed trainer and
previous local hardening changes were uploaded; its `--help` import check passed
in the actual McCleary runtime. Replacement smoke `10927119` requests one GPU
of any available type and excludes `r109u04n01`. The original failed logs are
preserved, and no existing fit/checkpoint was overwritten.

- CPU manifest job `10924498` completed and produced the manifest above.
- GPU smoke job `10924529` was submitted with an `afterok` dependency, 64 training
  batches and 12 evaluation batches per split. Latest verified scheduler state:
  pending for priority after the manifest completed.
- A subsequent GPU-request update was **not executed** because the tool approval
  service hit a usage limit. After the user resumed, approval worked but SSH
  failed because `mccleary.ycrc.yale.edu` did not resolve. Public `yale.edu` did
  resolve. Do not infer that the smoke failed or completed from this connection
  problem; retrieve its actual Slurm state and report when access returns.
- Full-run submission is gated on inspection of the GPU smoke result, measured
  throughput, peak memory, finite gradients, and nonzero supported ranking pairs.
- The GPU smoke still uses the A5000 request in its original launcher unless a
  later verified scheduler change is recorded.

Main implementation: `src/logica_binding/streaming_graph_logica.py`,
`scripts/prepare_pgk2_full_logica_manifest.py`, and
`scripts/run_pgk2_full_logica.py`.

Local regression suite: **52 passed**, including the opt-in offline integration
test against the actual released pretrained checkpoint. It verified two CPU
optimizer updates, development selection, checkpoint writing, holdout evaluation,
and restart from the last snapshot without an extra optimizer step; resumed
holdout results matched exactly. That test uses synthetic observations only to
verify execution and is not predictive validation.

The integration check prompted hardening after the original GPU job was
submitted: writable copies of NumPy evidence arrays, a synthetic-fixture guard,
and handling of a resume at the smoke-step limit. These changes were uploaded
before replacement smoke `10927119` and are included in full job `10927166`.
The core scientific objective and packed scoring are unchanged.
