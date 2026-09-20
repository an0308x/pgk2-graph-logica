#!/usr/bin/env bash
# Diagnostic only: identical split/objective, graph branch on versus off.
#SBATCH --job-name=pgk2-graph-v2-audit
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a5000:1
#SBATCH --array=0-5%3
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --output=artifacts/pgk2_graph_logica_v2_diagnostic/slurm-%A_%a.out
#SBATCH --error=artifacts/pgk2_graph_logica_v2_diagnostic/slurm-%A_%a.err

set -euo pipefail
cd "$HOME/pgk2_logica"
module load PyTorch/2.1.2-foss-2022b-CUDA-12.1.1
export PYTHONPATH="/vast/palmer/apps/avx2/software/PyTorch/2.1.2-foss-2022b-CUDA-12.1.1/lib/python3.10/site-packages:$PWD/src"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8

seed=$((2026 + SLURM_ARRAY_TASK_ID % 3))
conditioning=on
if (( SLURM_ARRAY_TASK_ID >= 3 )); then conditioning=off; fi
.runtime/bin/python -u scripts/run_pgk2_graph_logica_finetune.py \
  --pairs artifacts/pgk2_corrected_pairs_v2/pretrain.parquet \
  --epochs 3 --batch-size 16 --encoder-batch-size 64 \
  --ligand-mode 2d --graph-conditioning "$conditioning" \
  --run-purpose diagnostic --seed "$seed" --device cuda \
  --output-dir "artifacts/pgk2_graph_logica_v2_diagnostic/graph_${conditioning}_seed${seed}"
