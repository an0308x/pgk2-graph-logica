#!/usr/bin/env bash
#SBATCH --job-name=pgk2-v4-validation
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=r109u04n01
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --array=0-7%2
#SBATCH --output=artifacts/pgk2_full_logica_v3_logs/v4-validation-%A_%a.out
#SBATCH --error=artifacts/pgk2_full_logica_v3_logs/v4-validation-%A_%a.err
set -euo pipefail
cd /vast/palmer/home.mccleary/an923/pgk2_logica
module load PyTorch/2.1.2-foss-2022b-CUDA-12.1.1
export PYTHONPATH="/vast/palmer/apps/avx2/software/PyTorch/2.1.2-foss-2022b-CUDA-12.1.1/lib/python3.10/site-packages:$PWD/src"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 POLARS_MAX_THREADS=4
if [[ "${VERIFY_ONLY:-0}" == 1 ]]; then
  .runtime/bin/python -u scripts/score_pgk2_v4_validation.py \
    --input artifacts/pgk2_v4_validation/input.json \
    --run-dir artifacts/pgk2_dev_v4_cap5_graph_aux0 \
    --manifest artifacts/pgk2_full_logica_v3_manifest/manifest.json \
    --output-dir artifacts/pgk2_v4_validation --authorize-validation --verify-only
else
  .runtime/bin/python -u scripts/score_pgk2_v4_validation.py \
    --input artifacts/pgk2_v4_validation/input.json \
    --run-dir artifacts/pgk2_dev_v4_cap5_graph_aux0 \
    --manifest artifacts/pgk2_full_logica_v3_manifest/manifest.json \
    --output-dir artifacts/pgk2_v4_validation/chunks --authorize-validation \
    --verification artifacts/pgk2_v4_validation/verification.json \
    --shard-id "${SLURM_ARRAY_TASK_ID:?}" --num-shards 8
fi
