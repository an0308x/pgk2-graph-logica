#!/usr/bin/env bash
#SBATCH --job-name=pgk2-fp-production
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a5000:1
#SBATCH --array=0-2%3
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=06:00:00
#SBATCH --output=artifacts/pgk2_full_signal_fingerprint_production/slurm-%A_%a.out
#SBATCH --error=artifacts/pgk2_full_signal_fingerprint_production/slurm-%A_%a.err

set -euo pipefail

cd "$HOME/pgk2_logica"
mkdir -p artifacts/pgk2_full_signal_fingerprint_production
module load PyTorch/2.1.2-foss-2022b-CUDA-12.1.1
export PYTHONPATH="/vast/palmer/apps/avx2/software/PyTorch/2.1.2-foss-2022b-CUDA-12.1.1/lib/python3.10/site-packages:$PWD/src"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

seed=$((2026 + SLURM_ARRAY_TASK_ID))

.runtime/bin/python scripts/run_pgk2_full_graph_ablation.py \
  --data-dir artifacts/pgk2_rdkit_2d_full/data \
  --index artifacts/pgk2_full_canonical_index/canonical_training_index.parquet \
  --model-mode fingerprint \
  --graph-mode 2d \
  --data-scope all \
  --selection-metric active_score_spearman \
  --output-dir "artifacts/pgk2_full_signal_fingerprint_production/seed${seed}" \
  --epochs 5 \
  --batch-size 2048 \
  --graph-width 128 \
  --seed "$seed" \
  --device cuda
