#!/usr/bin/env bash
#SBATCH --job-name=pgk2-logica-v3-manifest
#SBATCH --partition=day
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=artifacts/pgk2_full_logica_v3_logs/manifest-%j.out
#SBATCH --error=artifacts/pgk2_full_logica_v3_logs/manifest-%j.err
set -euo pipefail
cd /vast/palmer/home.mccleary/an923/pgk2_logica
module load PyTorch/2.1.2-foss-2022b-CUDA-12.1.1
export PYTHONPATH="/vast/palmer/apps/avx2/software/PyTorch/2.1.2-foss-2022b-CUDA-12.1.1/lib/python3.10/site-packages:$PWD/src"
export POLARS_MAX_THREADS=4
export OMP_NUM_THREADS=1
.runtime/bin/python -u scripts/prepare_pgk2_full_logica_manifest.py \
  --graph-dir artifacts/pgk2_rdkit_2d_full/data \
  --output-dir artifacts/pgk2_full_logica_v3_manifest --workers 8
