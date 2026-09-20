#!/usr/bin/env bash
#SBATCH --job-name=pgk2-etkdg-ablation
#SBATCH --partition=day
#SBATCH --array=0-7%8
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=artifacts/pgk2_etkdg_ablation/slurm-%A_%a.out
#SBATCH --error=artifacts/pgk2_etkdg_ablation/slurm-%A_%a.err

set -euo pipefail

cd "$HOME/pgk2_logica"
mkdir -p artifacts/pgk2_etkdg_ablation/data
module load PyTorch/2.1.2-foss-2022b-CUDA-12.1.1
export PYTHONPATH="/vast/palmer/apps/avx2/software/PyTorch/2.1.2-foss-2022b-CUDA-12.1.1/lib/python3.10/site-packages:$PWD/src"

.runtime/bin/python scripts/precompute_pgk2_graph_shard.py \
  --index artifacts/pgk2_full_canonical_index/canonical_training_index.parquet \
  --output-dir artifacts/pgk2_etkdg_ablation/data \
  --graph-mode etkdg \
  --subset ablation \
  --shard-id "$SLURM_ARRAY_TASK_ID" \
  --num-shards 8 \
  --workers 10 \
  --optimization-max-iters 0 \
  --spatial-cutoff 4.5
