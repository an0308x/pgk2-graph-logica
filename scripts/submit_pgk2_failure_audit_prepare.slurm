#!/usr/bin/env bash
#SBATCH --job-name=pgk2-failure-prep
#SBATCH --partition=day
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=artifacts/pgk2_full_logica_v3_logs/failure-prep-%j.out
#SBATCH --error=artifacts/pgk2_full_logica_v3_logs/failure-prep-%j.err
set -euo pipefail
cd /vast/palmer/home.mccleary/an923/pgk2_logica
module load PyTorch/2.1.2-foss-2022b-CUDA-12.1.1
export PYTHONPATH="/vast/palmer/apps/avx2/software/PyTorch/2.1.2-foss-2022b-CUDA-12.1.1/lib/python3.10/site-packages:$PWD/src"
export OMP_NUM_THREADS=4 POLARS_MAX_THREADS=4
.runtime/bin/python -u scripts/prepare_pgk2_failure_audit.py \
  --manifest artifacts/pgk2_full_logica_v3_manifest/manifest.json \
  --merged artifacts/pgk2_v4_validation/merged \
  --submission artifacts/pgk2_v4_validation/merged/graph_logica_v4_cap5_validation_submission.txt \
  --output-dir artifacts/pgk2_failure_audit_9780912
