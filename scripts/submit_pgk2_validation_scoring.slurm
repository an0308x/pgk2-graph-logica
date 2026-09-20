#!/usr/bin/env bash
#SBATCH --job-name=pgk2-val-score
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --array=0-48%4
#SBATCH --output=artifacts/pgk2_validation_scores/slurm-%A_%a.out
#SBATCH --error=artifacts/pgk2_validation_scores/slurm-%A_%a.err

set -euo pipefail
cd "$HOME/pgk2_logica"
module load PyTorch/2.1.2-foss-2022b-CUDA-12.1.1
export PYTHONPATH="/vast/palmer/apps/avx2/software/PyTorch/2.1.2-foss-2022b-CUDA-12.1.1/lib/python3.10/site-packages:$PWD/src"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

chunk_size=5000
offset=$((SLURM_ARRAY_TASK_ID * chunk_size))
mkdir -p artifacts/pgk2_validation_scores/chunks
.runtime/bin/python scripts/run_pgk2_logica.py \
  --panel validation \
  --candidate-panels DREAM_challenge_2026/Val-Test-set.zip \
  --adapter artifacts/pgk2_pairwise_full/finetuned_adapter.pt \
  --offset "$offset" \
  --max-compounds "$chunk_size" \
  --batch-size 16 \
  --encoder-batch-size 64 \
  --device cuda \
  --output-dir artifacts/pgk2_validation_scores/chunks
