# PGK2 Graph LogiCA: code review package

This repository is a reviewable snapshot of our work adapting Graph LogiCA to
the DREAM × CACHE PGK2 DEL-to-ASMS challenge. It is intended for scientific and
code review, especially of the supervision assumptions, structural masking, and
evaluation design. It is **not** a claim that the included model predicts kinase
inhibition, binding affinity, IC50, or validated challenge hits.

## Current scientific status

The latest externally scored top-50 ranking returned zero hits (user-reported
Synapse submission `9780912`). The resulting audit found two central issues:

1. The internal DEL ordering score could be strongly explained by count patterns
   in the surrogate comparisons. An inverse inhibitor-count diagnostic achieved
   88.0% weighted ordering on the same development comparisons where the selected
   model achieved 80.2%. The diagnostic is not a deployable predictor; it directly
   uses the counts defining the comparison target.
2. The supervised DEL molecules and challenge validation molecules have a large
   chemistry gap. Median nearest-supervised-training Morgan similarity was about
   0.70 for sampled internal-development molecules and 0.31 for the submitted 50.

The repository records these negative findings deliberately. Please read
[`docs/PGK2_FAILURE_AUDIT_9780912.md`](docs/PGK2_FAILURE_AUDIT_9780912.md) and
[`docs/PGK2_SUBSTRATE_COMPARISON_PROTOCOL_2026-09-20.md`](docs/PGK2_SUBSTRATE_COMPARISON_PROTOCOL_2026-09-20.md)
before interpreting any historical internal metrics.

## What is included

- `src/logica_binding/`: the adapted LogiCA model, frozen-encoder interface,
  RDKit graph implementation, packed graph I/O, score functions, active-site
  masking and diagnostic utilities.
- `scripts/`: PGK2-only data preparation, graph caching, training, evaluation,
  inference, structural derivation, Slurm launchers, and audits.
- `tests/`: regression, provenance, graph construction, score parity and
  diagnostic tests.
- `provenance/`: small JSON pocket definitions and evidence-coverage reports.
- `docs/`: chronological experiment records, failures, submissions, and the
  proposed substrate-pocket comparison protocol.

## What is intentionally excluded

This repository contains **no** raw DEL/ASMS challenge tables, private candidate
lists, submitted CatalogIDs, checkpoints, ESM/SELFormer weights, Hugging Face
caches, generated RDKit graph archives, embeddings, Boltz outputs, or model
predictions. Some scripts therefore require separately authorized data and
official model assets. Do not infer that omission of an input file means it can
be redistributed.

## Model design, briefly

The historical full-library model used all 7,487,567 cached **2D** RDKit ligand
graphs. Full human PGK2 sequence was encoded by frozen ESM. Candidate ligands
were never assigned protein-complex poses, and the graph model did not use
intermolecular ligand-protein distances. Protein-ligand messages and
cross-attention were restricted to a fixed PGK2 region.

The older mask was based on residues around co-crystallized PGK2 inhibitors 21
and 47. The current preparation adds an inhibitor-independent alternative based
on ATP and 3-PG neighborhoods in mouse PGK2 structure 2PAA, mapped to human PGK2.
The proposed matched comparison uses identical human protein coordinates and a
common context shell, changing only the permitted direct-contact residue mask.

The key implementation entrypoints are:

```text
src/logica_binding/graph_model.py          graph conditioning and pocket graphs
src/logica_binding/streaming_graph_logica.py packed scoring and DEL loss
scripts/run_pgk2_full_logica.py             full-library training runner
scripts/score_pgk2_v4_validation.py         validation scoring runner
scripts/derive_pgk2_substrate_pocket.py     ATP + 3-PG mask derivation
scripts/audit_pgk2_evidence_coverage.py     count-evidence coverage audit
```

## Reproducibility notes

The historical pipeline used PyTorch 2.1.2, ESM2 8M and SELFormer, plus RDKit,
Polars, NumPy and PyTest. Install the Python dependencies:

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Several tests require model/data assets intentionally excluded from this package.
The code includes smoke-test gates so an incomplete environment should fail
clearly rather than create a purported full-library result.

## Important interpretation constraints

- A DEL count proxy is not a confirmed PGK2 kinase-inhibition label.
- The organizer states the released inhibitor column is censored to PGK2-positive
  compounds; missing inhibitor-arm-only rows are unavailable.
- Low or zero inhibitor counts are uncertain, not confirmed competition.
- The internal development split has already informed historical selection; it is
  not a pristine external validation set.
- The substrate-pocket comparison is prepared but **not trained**. No revised
  model should be submitted merely because an internal DEL metric improves.

## Review request

The most useful review questions are:

1. Is the proposed ATP + 3-PG mask and matched-context comparison scientifically
   appropriate for active-site inhibition?
2. Is there a defensible way to use the released censored DEL data without
   overinterpreting zero counts?
3. Is there independent permissible activity or assay metadata that would enable
   evaluation against kinase inhibition rather than only DEL ordering?

