#!/usr/bin/env bash
#SBATCH --job-name=pgk2-v4-val-merge
#SBATCH --partition=day
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=artifacts/pgk2_full_logica_v3_logs/v4-validation-merge-%j.out
#SBATCH --error=artifacts/pgk2_full_logica_v3_logs/v4-validation-merge-%j.err
set -euo pipefail
cd /vast/palmer/home.mccleary/an923/pgk2_logica
module load PyTorch/2.1.2-foss-2022b-CUDA-12.1.1
export PYTHONPATH="/vast/palmer/apps/avx2/software/PyTorch/2.1.2-foss-2022b-CUDA-12.1.1/lib/python3.10/site-packages:$PWD/src"
export POLARS_MAX_THREADS=2 OMP_NUM_THREADS=2
.runtime/bin/python scripts/merge_pgk2_v4_validation.py \
  --input artifacts/pgk2_v4_validation/input.json \
  --chunks artifacts/pgk2_v4_validation/chunks \
  --verification artifacts/pgk2_v4_validation/verification.json \
  --output-dir artifacts/pgk2_v4_validation/merged
