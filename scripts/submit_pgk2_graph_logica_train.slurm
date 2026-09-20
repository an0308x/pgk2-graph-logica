#!/usr/bin/env bash
# Two-stage Graph LogiCA: broad matched-SAR pretraining followed by active-site
# competition refinement. No Boltz poses or candidate complex structures.
#SBATCH --job-name=pgk2-graph-logica
#SBATCH --partition=gpu_devel
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=01:00:00
#SBATCH --output=artifacts/pgk2_graph_logica_train/slurm-%j.out
#SBATCH --error=artifacts/pgk2_graph_logica_train/slurm-%j.err

set -euo pipefail

cd "$HOME/pgk2_logica"
module load PyTorch/2.1.2-foss-2022b-CUDA-12.1.1
export PYTHONPATH="/vast/palmer/apps/avx2/software/PyTorch/2.1.2-foss-2022b-CUDA-12.1.1/lib/python3.10/site-packages:$PWD/src"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
mkdir -p artifacts/pgk2_graph_logica_train

.runtime/bin/python scripts/run_pgk2_graph_logica_finetune.py \
  --pairs artifacts/pgk2_matched_sar_pairs/pgk2_matched_sar_pairs.parquet \
  --max-pairs 11953 \
  --epochs 3 \
  --batch-size 8 \
  --encoder-batch-size 64 \
  --graph-workers 8 \
  --learning-rate 3e-5 \
  --device cuda \
  --output-dir artifacts/pgk2_graph_logica_pretrain

.runtime/bin/python scripts/run_pgk2_graph_logica_finetune.py \
  --pairs artifacts/pgk2_competition_pairs/pgk2_competition_pairs.parquet \
  --initial-graph-adapter artifacts/pgk2_graph_logica_pretrain/graph_logica_adapter.pt \
  --max-pairs 279 \
  --epochs 15 \
  --batch-size 8 \
  --encoder-batch-size 64 \
  --graph-workers 8 \
  --learning-rate 3e-5 \
  --device cuda \
  --output-dir artifacts/pgk2_graph_logica_competition_finetune
