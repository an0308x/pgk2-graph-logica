#!/usr/bin/env bash
#SBATCH --job-name=pgk2-failure-audit
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=r109u04n01
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=artifacts/pgk2_full_logica_v3_logs/failure-audit-%j.out
#SBATCH --error=artifacts/pgk2_full_logica_v3_logs/failure-audit-%j.err
set -euo pipefail
cd /vast/palmer/home.mccleary/an923/pgk2_logica
module load PyTorch/2.1.2-foss-2022b-CUDA-12.1.1
export PYTHONPATH="/vast/palmer/apps/avx2/software/PyTorch/2.1.2-foss-2022b-CUDA-12.1.1/lib/python3.10/site-packages:$PWD/src"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 POLARS_MAX_THREADS=4
.runtime/bin/python -u scripts/run_pgk2_failure_audit.py \
  --audit-dir artifacts/pgk2_failure_audit_9780912 \
  --run-dir artifacts/pgk2_dev_v4_cap5_graph_aux0 --device cuda
