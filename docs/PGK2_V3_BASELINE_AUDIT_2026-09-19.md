# PGK2 v3 baseline and supervision audit

Both authorized jobs completed successfully: supervision audit `10936707`
(45 seconds), matched development comparison `10936708` (3m45s). All 56 local
tests pass. No holdout was reevaluated, labels changed, challenge candidates
scored, or submission made.

## Main result

The evaluation reproduced the full-trained model's original development
competition metric exactly. All models use the same 4,086 competition pairs,
weights, 238 development batches across the three channels, and scaffold split.
The fingerprint baseline replays all 754,240 supervised molecule visits from
the full run, but has no molecular self-supervised objective.

| Model | Competition weighted pair accuracy | Combined confidence-scaled pair loss (lower is better) |
| --- | ---: | ---: |
| Full-trained Graph LogiCA | 61.25% | 0.003249 |
| Same Graph LogiCA before PGK2 training | 60.93% | 0.002798 |
| Starting pocket-restricted LogiCA without graph conditioning | 59.88% | 0.002776 |
| Linear Morgan-2048 ranker | 73.34% | 0.003767 |

Training improved competition ordering by only 0.32 percentage points relative
to its starting graph initialization and worsened the combined development
loss. The Morgan baseline improved ordering by 12.09 points over the trained
graph model but had worse pair loss. Ordering and margin-sensitive loss must
not be conflated; raw scores are not calibrated kinase-hit probabilities.
This is one prespecified baseline fit, not a tuned leaderboard or a causal
graph-ablation experiment. No confidence interval or significance claim is made.

The full trainer compared epoch-end checkpoints only. With one epoch, it did
not test whether the starting checkpoint had better development loss. Before
another run, checkpoint selection should include epoch zero and intermediate
development checks. This does not imply that the starting model will produce
kinase hits: no model has demonstrated that here.

## Supervision coverage

Of 6,026,213 training molecules, 20,208 receive competition-ranking supervision,
and 24,445 receive any of the three observed-control ranking signals. All train
molecules receive ligand-only auxiliary learning. Those are distinct claims.

| Mean target-read threshold | Training molecules meeting threshold | Also have zero competitor reads, excluded from competition supervision |
| --- | ---: | ---: |
| At least 5 | 3,640 | 2,886 |
| At least 10 | 1,279 | 969 |
| At least 20 | 360 | 248 |
| At least 50 | 66 | 37 |
| At least 100 | 20 | 6 |

All 248 excluded molecules with mean target reads at least 20 have a nonzero
NTC or historic-hit signal. None passes the simultaneous zero-NTC/no-history
screen used for this descriptive audit. These flags do not prove promiscuity or
non-inhibition, and zero reads do not prove absence of background. Automatic
positive relabeling is not justified. A count/depth-aware sensitivity analysis
is warranted before changing the objective.

The competition evaluation includes 1,891 distinct pair-participating molecules.
Its weight-only effective sample size is about 2,307 pairs, and the ten largest
compound contributions account for 7.4% of incidence weight. NTC evaluations are
more concentrated: the top ten compounds contribute about 34–35%. Pair sharing
and scaffold dependence prevent treating weight-only ESS as an independent
sample size.

An exploratory check over 2,966 unique evaluated development molecules gave
Spearman correlations between capped token length and score of -0.108 (trained
graph), -0.230 (initial graph), -0.234 (initial no-graph), and -0.116 (Morgan).
These marginal associations do not establish or exclude length-related bias,
and do not evaluate transfer to ASMS. Full challenge-library transfer inspection
remains pending; the 128-token truncation limitation remains unresolved.

## Recommended next gate

Do not launch another full-library graph fit or promote a submission based on
these results alone. Keep the fingerprint model as the ligand-only benchmark,
include the starting checkpoint in future selection, and test evidence/zero-read
handling in bounded development-only experiments. Only a model passing those
checks should proceed to ASMS score-distribution/diversity inspection and a
separately reviewed challenge submission.

Artifacts: `artifacts/pgk2_v3_baselines/report.json`,
`artifacts/pgk2_v3_baselines/development_scores.parquet`, and
`artifacts/pgk2_v3_supervision_audit/report.json`.
