#!/usr/bin/env bash
# Clean adapter-only baseline using LogiCA's official DTI score.  This starts
# from the released pretraining checkpoint (not a prior PGK2 adapter), reads all
# valid protein/drug tokens exactly as pair_scores_from_tokens does, and learns
# pair_alpha.  It intentionally does not claim to reproduce the official LoRA
# recipe because the cached-encoder runner freezes both language models.
#SBATCH --job-name=pgk2-official-score-ft
#SBATCH --partition=gpu_devel
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=artifacts/pgk2_official_score_finetune/slurm-%j.out
#SBATCH --error=artifacts/pgk2_official_score_finetune/slurm-%j.err

set -euo pipefail

cd "$HOME/pgk2_logica"
module load PyTorch/2.1.2-foss-2022b-CUDA-12.1.1
export PYTHONPATH="/vast/palmer/apps/avx2/software/PyTorch/2.1.2-foss-2022b-CUDA-12.1.1/lib/python3.10/site-packages:$PWD/src"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
mkdir -p artifacts/pgk2_official_score_finetune

.runtime/bin/python - <<'PY'
import torch
assert torch.cuda.is_available(), "Slurm allocation did not expose CUDA"
print({"torch": torch.__version__, "device": torch.cuda.get_device_name(0)})
PY

.runtime/bin/python scripts/run_pgk2_pairwise_finetune.py \
  --pairs artifacts/pgk2_competition_pairs/pgk2_competition_pairs.parquet \
  --max-pairs 279 \
  --epochs 20 \
  --batch-size 16 \
  --encoder-batch-size 64 \
  --learning-rate 3e-5 \
  --score-mode token_likelihood \
  --protein-mask-region random \
  --device cuda \
  --output-dir artifacts/pgk2_official_score_finetune
