#!/usr/bin/env bash
#SBATCH --job-name=pgk2-selfies-val
#SBATCH --partition=gpu_devel
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --array=0-1%2
#SBATCH --output=artifacts/pgk2_selfies_dualgated_validation_scores/slurm-%A_%a.out
#SBATCH --error=artifacts/pgk2_selfies_dualgated_validation_scores/slurm-%A_%a.err

set -euo pipefail

cd "$HOME/pgk2_logica"
module load PyTorch/2.1.2-foss-2022b-CUDA-12.1.1
export PYTHONPATH="/vast/palmer/apps/avx2/software/PyTorch/2.1.2-foss-2022b-CUDA-12.1.1/lib/python3.10/site-packages:$PWD/src"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

chunk_size=10000
chunks_per_task=13
first_chunk=$((SLURM_ARRAY_TASK_ID * chunks_per_task))
last_chunk=$((first_chunk + chunks_per_task))
if (( last_chunk > 25 )); then
  last_chunk=25
fi
mkdir -p artifacts/pgk2_selfies_dualgated_validation_scores/chunks
for chunk in $(seq "$first_chunk" "$((last_chunk - 1))"); do
  offset=$((chunk * chunk_size))
  .runtime/bin/python scripts/run_pgk2_logica.py \
    --panel validation \
    --candidate-panels DREAM_challenge_2026/Val-Test-set.zip \
    --adapter artifacts/pgk2_selfies_dualgated_finetune/finetuned_adapter.pt \
    --offset "$offset" \
    --max-compounds "$chunk_size" \
    --batch-size 16 \
    --encoder-batch-size 64 \
    --device cuda \
    --output-dir artifacts/pgk2_selfies_dualgated_validation_scores/chunks
done
