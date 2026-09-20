#!/usr/bin/env bash
# Benchmark and audit packed RDKit graphs on 100k molecules sampled across the
# complete 7.7M-row PGK2 DEL release. This is CPU preprocessing, not training.
#SBATCH --job-name=pgk2-rdkit-100k
#SBATCH --partition=day
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=artifacts/pgk2_rdkit_graph_pilot/slurm-%j.out
#SBATCH --error=artifacts/pgk2_rdkit_graph_pilot/slurm-%j.err

set -euo pipefail

cd "$HOME/pgk2_logica"
module load PyTorch/2.1.2-foss-2022b-CUDA-12.1.1
export PYTHONPATH="/vast/palmer/apps/avx2/software/PyTorch/2.1.2-foss-2022b-CUDA-12.1.1/lib/python3.10/site-packages:$PWD/src"
export TOKENIZERS_PARALLELISM=false

.runtime/bin/python scripts/precompute_pgk2_rdkit_graphs.py \
  --selection PGK2_selection.parquet \
  --candidate-panels DREAM_challenge_2026/Val-Test-set.zip \
  --output-dir artifacts/pgk2_rdkit_graph_pilot/data \
  --max-molecules 100000 \
  --shard-size 25000 \
  --workers 10 \
  --optimization-max-iters 50 \
  --spatial-cutoff 4.5
