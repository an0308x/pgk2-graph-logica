#!/usr/bin/env bash
#SBATCH --job-name=pgk2-logica-rank
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=artifacts/pgk2_pairwise_full/slurm-%j.out
#SBATCH --error=artifacts/pgk2_pairwise_full/slurm-%j.err

set -euo pipefail

cd "$HOME/pgk2_logica"
module load PyTorch/2.1.2-foss-2022b-CUDA-12.1.1
export PYTHONPATH="/vast/palmer/apps/avx2/software/PyTorch/2.1.2-foss-2022b-CUDA-12.1.1/lib/python3.10/site-packages:$PWD/src"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
mkdir -p artifacts/pgk2_pairwise_full

.runtime/bin/python - <<'PY'
import torch
assert torch.cuda.is_available(), "Slurm allocation did not expose CUDA"
print({"torch": torch.__version__, "device": torch.cuda.get_device_name(0)})
PY

.runtime/bin/python scripts/run_pgk2_pairwise_finetune.py \
  --pairs artifacts/pgk2_pairs/pgk2_ranking_pairs.parquet \
  --max-pairs 21385 \
  --epochs 5 \
  --batch-size 16 \
  --encoder-batch-size 64 \
  --device cuda \
  --output-dir artifacts/pgk2_pairwise_full
