#!/usr/bin/env bash
#SBATCH --job-name=pgk2-selfies-sar
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=artifacts/pgk2_selfies_embeddings/slurm-%j.out
#SBATCH --error=artifacts/pgk2_selfies_embeddings/slurm-%j.err

set -euo pipefail

cd "$HOME/pgk2_logica"
module load PyTorch/2.1.2-foss-2022b-CUDA-12.1.1
export PYTHONPATH="/vast/palmer/apps/avx2/software/PyTorch/2.1.2-foss-2022b-CUDA-12.1.1/lib/python3.10/site-packages:$PWD/src"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
mkdir -p artifacts/pgk2_selfies_embeddings artifacts/pgk2_selfies_neighbor_pairs

.runtime/bin/python - <<'PY'
import torch
assert torch.cuda.is_available(), "Slurm allocation did not expose CUDA"
print({"torch": torch.__version__, "device": torch.cuda.get_device_name(0)})
PY

.runtime/bin/python scripts/embed_pgk2_selfies_pool.py \
  --pool artifacts/pgk2_selfies_neighbor_pool/selfies_neighbor_pool.parquet \
  --batch-size 128 \
  --device cuda \
  --output-dir artifacts/pgk2_selfies_embeddings

.runtime/bin/python scripts/mine_pgk2_selfies_neighbors.py \
  --embeddings artifacts/pgk2_selfies_embeddings/embeddings.npy \
  --metadata artifacts/pgk2_selfies_embeddings/metadata.parquet \
  --minimum-cosine 0.90 \
  --neighbors-per-anchor 32 \
  --pairs-per-anchor 2 \
  --output-dir artifacts/pgk2_selfies_neighbor_pairs
