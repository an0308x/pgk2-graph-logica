# PGK2 DREAM × CACHE — project handoff

Last updated: 2026-09-16

> Historical snapshot. For the current active-site-competition model, inhibitor-pocket ensemble, completed McCleary jobs, and latest recommendation, use [`PGK2_HANDOFF_2026-09-16.md`](PGK2_HANDOFF_2026-09-16.md).

## Current status

The current reference is **matched-SAR LogiCA**. It retrieved 1 blinded ASMS
validation hit from 50 molecules (one chemical cluster; `p=0.05201186181800366`).
This is better than the initial model, but not nearly enough to justify a test
submission. Keep the blind test untouched while improving through validation.

## Data boundaries

| Data | Contents | Use |
|---|---:|---|
| `PGK2_selection.parquet` | 7,703,070 released DEL rows | Training signal |
| `PGK2_NTC_supplement.parquet` | Supplemental control observations | Control recheck |
| ASMS validation split | 244,328 unlabeled candidates | Rank and submit top 50 |
| ASMS test split | 184,632 unlabeled candidates | Final evaluation only |

The challenge describes the original screen as close to one billion DEL
compounds; the selection file is the practical release. ASMS per-compound
labels are not locally available. Portal feedback is aggregate and must not be
used as fine-tuning labels.

All validation and test SMILES were excluded before DEL pair construction
(428,960 candidate SMILES total).

## DEL labels and pair sets

`scripts/prepare_pgk2_strict_hits.py` deduplicates exact SMILES, sums counts,
Stouffer-combines z-scores, and applies:

```text
count_PGK2 > 3
inhibitor == 0 OR inhibitor < 0.1 × target
NTC == 0 OR NTC < 0.1 × target
historic_hits < 5
```

This initially produced 13,184 strict positives. Rechecking against the NTC
supplement leaves **13,176** eligible positives. A control zero is always
treated as missing evidence, never evidence of specificity.

### Broad pairs (superseded)

`scripts/build_pgk2_pair_dataset.py` produced 21,385 pairs: 8,057
control-confirmed pairs (weight 1.0) and 13,328 lower-confidence target-only
pairs (weight 0.35). It used a sampled counter pool plus random low-target
negatives and did not transfer to ASMS.

### Matched-SAR pairs (current)

`scripts/build_pgk2_matched_sar_pairs.py` requires a lower-ranked candidate to:

1. come from the same OpenDEL library;
2. share two of three building blocks with the positive;
3. have Morgan radius-2/2048-bit Tanimoto similarity >= 0.60;
4. show inhibitor or NTC evidence; and
5. be dominated by the positive in target and control evidence.

Output: **11,953** pairs from **7,333** positives. Similarity: 10th percentile
0.622, median 0.698, 90th percentile 0.802. This is a building-block/Morgan
proxy for local SELFIES/SELFormer neighbor mining; an all-release SELFormer
embedding index has not yet been built.

## Model and validation evidence

Pair win rates are DEL-derived training diagnostics, not ASMS performance.

| Variant | DEL scaffold-group pair win rate | Blind validation result |
|---|---:|---|
| Broad-pair LogiCA | 0.9324 | 0 hits / 50 |
| Matched-SAR LogiCA | 0.8697 | 1 hit, 1 cluster, `p=0.05201186181800366` |
| Matched-SAR fingerprint logistic regression | 0.8270 | 0 hits / 50 |
| Matched-SAR, one-per-series portfolio | Same scores; alternative selection | 1 hit, 1 cluster, same p value |

Matched-SAR LogiCA used five A100 epochs, 11,953 pairs, batch size 16, encoder
batch size 64, and adapter/cross-attention fine-tuning only. Loss fell from
0.4427 to 0.1200.

## Portal submission history

The **Blind validation** queue expects exactly 50 `CatalogID`s, one per line,
with no header, score, or SMILES. The Blind test queue has another format and
only two allowed submissions.

| Submission ID | Variant | Result |
|---:|---|---|
| 9779037 | Broad-pair LogiCA | 0 hits |
| 9779038 | Matched-SAR LogiCA | 1 hit, 1 cluster, `p=0.05201186181800366` |
| 9779039 | Fingerprint baseline | 0 hits |
| 9779051 | Matched-SAR one-per-series | 1 hit, 1 cluster, `p=0.05201186181800366` |
| 9780724 | SELFormer + Morgan dual-gated LogiCA | 0 hits |
| 9780732 | Active-site competition-refined LogiCA | 0 hits |

## Chemical-series portfolio

The challenge cites Kruger, Fechner, and Stiefl, *Automated Identification of
Chemical Series: Classifying like a Medicinal Chemist* (2020), DOI
`10.1021/acs.jcim.0c00204`. Its reported best approach is UPGMA clustering plus
maximum-common-substructure (MCS) series definitions.

`scripts/select_pgk2_series_portfolio.py` is a **Kruger-inspired**, not exact,
implementation: UPGMA/average linkage on Morgan Tanimoto distance, followed by
MCS definitions for audit. Obtain the paper's supplementary code or the
organizers' exact parameters before claiming exact cluster agreement.

On the top 1,000 matched-SAR candidates (similarity cut 0.50):

- 613 inferred series;
- the raw top-50 has 43 inferred series;
- caps of 3 or 5 compounds/series reproduce the raw top-50;
- a cap of 1 swaps seven molecules to make 50 series, without improving the
  observed validation score.

Thus diversity is currently not the obvious bottleneck; improve hit
probability while retaining a cluster-aware final selection rule.

## Important artifacts

| Purpose | Path |
|---|---|
| Strict positives | `artifacts/pgk2_training/strict_positive_hits.parquet` |
| Broad pairs | `artifacts/pgk2_pairs/pgk2_ranking_pairs.parquet` |
| Matched-SAR pairs | `artifacts/pgk2_matched_sar_pairs/pgk2_matched_sar_pairs.parquet` |
| Matched-SAR validation ranking | `artifacts/pgk2_matched_sar_validation_scores/merged/validation_ranked_scores.csv` |
| Fingerprint validation ranking | `artifacts/pgk2_fingerprint_baseline/validation_ranked_scores.csv` |
| Series-cluster audit | `artifacts/pgk2_series_portfolios/series_definitions.csv` |
| One-per-series submission | `artifacts/pgk2_series_portfolios/validation_series_cap1_submission.txt` |

Cluster-side matched-SAR adapter:

```text
~/pgk2_logica/artifacts/pgk2_matched_sar_finetune/finetuned_adapter.pt
```

## Reproduction

```bash
.venv/bin/python scripts/build_pgk2_matched_sar_pairs.py \
  --pairs-per-positive 2 \
  --output-dir artifacts/pgk2_matched_sar_pairs

.venv/bin/python scripts/run_pgk2_fingerprint_baseline.py \
  --output-dir artifacts/pgk2_fingerprint_baseline

.venv/bin/python scripts/select_pgk2_series_portfolio.py \
  --pool-size 1000 --minimum-similarity 0.50 \
  --output-dir artifacts/pgk2_series_portfolios
```

On McCleary:

```bash
sbatch scripts/submit_pgk2_matched_sar_finetune.slurm
sbatch scripts/submit_pgk2_matched_sar_validation_scoring.slurm
```

## Next steps

1. Obtain and run the exact Kruger supplementary implementation or organizer
   cluster settings.
2. Mine hard local pairs using frozen SELFormer/SELFIES embeddings across the
   DEL release or a defensible high-information subset.
3. Add permitted public PGK2 ligands and protein–ligand structures from CACHE
   as an independent feature/source, and validate each ablation.
4. Reproduce the organizer's DEL baseline before more architecture changes.
5. Freeze one model and one selection rule only after a clear validation gain;
   then use the two blind-test submissions.

## Guardrails

- Never use the test set for model selection.
- Never treat aggregate validation feedback as compound labels.
- Never describe pair loss or pair win rate as ASMS performance.
- Do not submit test predictions until validation is materially stronger.
