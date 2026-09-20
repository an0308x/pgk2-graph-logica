# User-authorized v4 validation submission

On 2026-09-19 the user requested: “Let us just submit to validation.” This
authorizes scoring the validation library and submitting one top-50 prediction
file without the additional seed/holdout experiments previously recommended.
It does not authorize scoring or submitting the blind test.

## Frozen selection

Use `artifacts/pgk2_dev_v4_cap5_graph_aux0/best.pt` from completed task
`10937301_7`. It is native token-likelihood Graph LogiCA with ranking-only
training, the zero-competitor assumed 0–5 count interval, and seed 2026.
Selected step: 2,198. Development competition ordering: 80.1968%; combined
development pair loss: 0.0013149744. These are not kinase-assay hit rates.
All nine development runs completed successfully. No new holdout evaluation
is conducted for this submission.

The original checkpoint's `candidate_scoring_allowed: false` is retained as its
historical development status. The new inference command requires explicit
`--authorize-validation` and a successful, checkpoint-bound verification report.
Authorization is recorded in inference provenance, not by overwriting the model.

## Input and inference

The official `Val-Test-set/PGK2_Validation_split.csv` has 244,328 rows and unique,
nonempty CatalogIDs, with exactly the columns `CatalogID` and `SMILES`.
The CSV is read using bundled Python following the spreadsheet skill's source
preservation and identifier checks; the original file remains unchanged.
It is converted losslessly to `artifacts/pgk2_v4_validation/input.json` for the
scientific scoring pipeline. No spreadsheet workbook is created.

- Source ZIP SHA-256: `06168468f943df6ac420466f16ad398ce097608da9e23dbf9ea98d3d5dbf8ab8`.
- Validation CSV member SHA-256: `337040d538210a005075a431b4c42aa0ad96d3fa19c76e389ff521e17ff3cb72`.
- The ZIP checksum must match the candidate exclusion source in the training manifest.

SMILES are canonicalized with stereochemistry retained; 2D atom/bond graphs use
the exact training implementation. The model uses full-sequence ESM and the
unchanged compounds-21/47 pocket ensemble. No ETKDG or complex prediction.
Keep the existing 128-token limit and report truncation; do not change the
trained scoring function during inference.

Before full inference: reproduce selected development metrics; compare freshly
built molecular graphs and their scores against cached training-format graphs;
test inference on the first 128 validation molecules for runtime correctness.
This is not a candidate selection pass. All model, input and checkpoint
identities must match at scoring and merge time.

Full inference uses eight disjoint shards with at most two generic GPUs in
parallel. The merge must verify all 244,328 source row indices and CatalogIDs,
finite scores, shard bounds/checksums and identical provenance.

Select raw descending score, with CatalogID ascending for exact ties. No
blending, diversity reranking, feedback-based exclusions or post-hoc filters.
Emit exactly 50 unique CatalogIDs, one per line, without a header or scores.
Retain complete ranked Parquet and audit JSON locally/remotely as appropriate.

## Portal

The current PGK2 challenge is `syn75349604` (not the older WDR91 project).
Official submission instructions:
https://www.synapse.org/Synapse:syn75349604/wiki/641048
Upload the prediction file to the team's existing Synapse project, then submit
the FILE to **Blind validation**, as Yale CompBio. Never select Blind test.
The organizer confirms validation is the top-50 CatalogID text format:
https://www.synapse.org/Synapse:syn75349604/discussion/threadId=14446

The in-app browser was signed out. The user was asked asynchronously to sign
in to the existing account. Do not claim challenge submission without a
confirmed submission ID from the portal. No new account or terms acceptance is
authorized here.

## Execution record

- All 68 local tests pass, including exact graph construction, checkpoint
  identity, descending ordering, duplicate/missing-ID and finite-score checks.
- Verification job `10939858_0` completed successfully in 2m01s. Selected
  competition metric reproduced exactly (80.1968%); all channel losses match
  within tolerance; fresh graph tensor/score parity passed. The 128-row
  validation smoke completed in 0.507 seconds with no token truncations.
- Selected adapter SHA-256: `409b88118952b3ea327a3244fa279b9f4400ea16064a0bc0c91768d282e7e91d`.
- Full scoring array: `10939870`, eight tasks with at most two GPUs concurrently,
  dependent on successful verification. No new training is performed.
- Automatic audit/merge job: `10939895`, dependent on all scoring tasks succeeding.
  It writes `artifacts/pgk2_v4_validation/merged/graph_logica_v4_cap5_validation_submission.txt`
  on McCleary, plus ranked Parquet and audit JSON. It does not upload to Synapse.
- Portal submission remains pending full predictions and Synapse login.

## Completed file delivery

All eight scoring tasks `10939870_0`–`_7` and merge `10939895` completed with
exit code zero. Coverage: 244,328 validation candidates; no token truncations.
The selected 50 IDs represent 50 structures and 49 Murcko scaffolds.

The user subsequently requested file delivery only and will upload to Synapse
personally. Do not continue the earlier browser-upload workflow.
Downloaded file:
`outputs/pgk2_validation_submission/graph_logica_v4_cap5_validation_submission.txt`.
SHA-256: `c07f3f22690c187f01db11b08bcb5badca8a9301b8b466ec18b5232d1d65029c`.
Local checks confirm 50 unique, nonempty lines and matching checksum.
No challenge submission was made by the assistant.

## User-reported result and diagnostic follow-up

User reports submission `9780912` scored `N_hits: 0`. The uploaded bytes have not
been independently fetched from Synapse; the provenance above describes the
file delivered to the user. Internal DEL pair accuracy is not a kinase hit rate.

User approved a diagnostic audit. Preparation job `10943595` and dependent
inference-only job `10943597` examine target dependence, chemical transfer and
count supervision without retraining or creating a new submission. See
`PGK2_FAILURE_AUDIT_9780912.md`.

Audit completed: preparation `10943595`, verified GPU rerun `10943727`.
Production scoring reproduced with unchanged top 50; development metrics match.
The report identifies weak-count evaluation and chemical-transfer concerns, and
little aggregate development benefit from graph conditioning in this checkpoint.
No new training or challenge submission was performed.
