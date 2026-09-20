#!/usr/bin/env bash
#SBATCH --job-name=pgk2-graph-smoke
#SBATCH --partition=gpu_devel
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=01:00:00
#SBATCH --output=artifacts/pgk2_full_graph_ablation_smoke/slurm-%j.out
#SBATCH --error=artifacts/pgk2_full_graph_ablation_smoke/slurm-%j.err

set -euo pipefail

cd "$HOME/pgk2_logica"
mkdir -p artifacts/pgk2_full_graph_ablation_smoke
module load PyTorch/2.1.2-foss-2022b-CUDA-12.1.1
export PYTHONPATH="/vast/palmer/apps/avx2/software/PyTorch/2.1.2-foss-2022b-CUDA-12.1.1/lib/python3.10/site-packages:$PWD/src"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

.runtime/bin/python scripts/run_pgk2_full_graph_ablation.py \
  --data-dir artifacts/pgk2_rdkit_2d_full/data \
  --index artifacts/pgk2_full_canonical_index/canonical_training_index.parquet \
  --model-mode pocket \
  --graph-mode 2d \
  --output-dir artifacts/pgk2_full_graph_ablation_smoke/run \
  --epochs 1 \
  --batch-size 192 \
  --graph-width 128 \
  --device cuda \
  --max-shards 2 \
  --max-train-batches 20 \
  --max-eval-batches 5
