#!/usr/bin/env bash
#SBATCH --job-name=pgk2-graph-ablation
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a5000:1
#SBATCH --array=0-11%4
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=06:00:00
#SBATCH --output=artifacts/pgk2_full_graph_ablation/slurm-%A_%a.out
#SBATCH --error=artifacts/pgk2_full_graph_ablation/slurm-%A_%a.err

set -euo pipefail

cd "$HOME/pgk2_logica"
mkdir -p artifacts/pgk2_full_graph_ablation
module load PyTorch/2.1.2-foss-2022b-CUDA-12.1.1
export PYTHONPATH="/vast/palmer/apps/avx2/software/PyTorch/2.1.2-foss-2022b-CUDA-12.1.1/lib/python3.10/site-packages:$PWD/src"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

configuration=$((SLURM_ARRAY_TASK_ID / 3))
seed=$((2026 + SLURM_ARRAY_TASK_ID % 3))
model_modes=(fingerprint ligand pocket pocket)
graph_modes=(2d 2d 2d etkdg)
data_dirs=(
  artifacts/pgk2_rdkit_2d_full/data
  artifacts/pgk2_rdkit_2d_full/data
  artifacts/pgk2_rdkit_2d_full/data
  artifacts/pgk2_etkdg_ablation/data
)
labels=(fingerprint_2d ligand_2d pocket_2d pocket_etkdg)

model_mode="${model_modes[$configuration]}"
graph_mode="${graph_modes[$configuration]}"
data_dir="${data_dirs[$configuration]}"
label="${labels[$configuration]}"

.runtime/bin/python scripts/run_pgk2_full_graph_ablation.py \
  --data-dir "$data_dir" \
  --index artifacts/pgk2_full_canonical_index/canonical_training_index.parquet \
  --model-mode "$model_mode" \
  --graph-mode "$graph_mode" \
  --output-dir "artifacts/pgk2_full_graph_ablation/${label}_seed${seed}" \
  --epochs 5 \
  --batch-size 192 \
  --graph-width 128 \
  --seed "$seed" \
  --device cuda
