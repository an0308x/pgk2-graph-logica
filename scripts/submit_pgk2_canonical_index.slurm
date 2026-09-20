#!/usr/bin/env bash
#SBATCH --job-name=pgk2-canonical-index
#SBATCH --partition=day
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=artifacts/pgk2_full_canonical_index/slurm-%j.out
#SBATCH --error=artifacts/pgk2_full_canonical_index/slurm-%j.err

set -euo pipefail

cd "$HOME/pgk2_logica"
mkdir -p artifacts/pgk2_full_canonical_index
module load PyTorch/2.1.2-foss-2022b-CUDA-12.1.1
export PYTHONPATH="/vast/palmer/apps/avx2/software/PyTorch/2.1.2-foss-2022b-CUDA-12.1.1/lib/python3.10/site-packages:$PWD/src"

.runtime/bin/python scripts/build_pgk2_canonical_index.py \
  --selection PGK2_selection.parquet \
  --ntc-supplement PGK2_NTC_supplement.parquet \
  --candidate-panels DREAM_challenge_2026/Val-Test-set.zip \
  --output-dir artifacts/pgk2_full_canonical_index \
  --workers 10 \
  --batch-size 100000 \
  --resume
