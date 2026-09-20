#!/usr/bin/env bash
#SBATCH --job-name=pgk2-logica-v3-smoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a5000:1
#SBATCH --exclude=r109u04n01
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=00:30:00
#SBATCH --output=artifacts/pgk2_full_logica_v3_logs/smoke-%j.out
#SBATCH --error=artifacts/pgk2_full_logica_v3_logs/smoke-%j.err
set -euo pipefail
cd /vast/palmer/home.mccleary/an923/pgk2_logica
module load PyTorch/2.1.2-foss-2022b-CUDA-12.1.1
export PYTHONPATH="/vast/palmer/apps/avx2/software/PyTorch/2.1.2-foss-2022b-CUDA-12.1.1/lib/python3.10/site-packages:$PWD/src"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 POLARS_MAX_THREADS=4
.runtime/bin/python -u scripts/run_pgk2_full_logica.py \
  --manifest artifacts/pgk2_full_logica_v3_manifest/manifest.json \
  --output-dir artifacts/pgk2_full_logica_v3_smoke \
  --device cuda --batch-size 64 --rank-batch-size 32 --rank-every 4 \
  --epochs 1 --max-train-batches 64 --max-eval-batches 12
