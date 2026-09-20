#!/usr/bin/env bash
# Resumable blind-validation scoring with the trained WALLE-style Graph LogiCA.
# Produces score chunks only; it does not submit anything to the challenge.
# Submit with --export=ALL,ROUND=N. Each round uses two immediate A5000 workers,
# with three 8k chunks per worker, and stays within the observed two-GPU /
# ten-CPU McCleary user limits and one-hour gpu_devel limit.
#SBATCH --job-name=pgk2-graph-logica-val
#SBATCH --partition=gpu_devel
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=40G
#SBATCH --time=01:00:00
#SBATCH --array=0-1%2
#SBATCH --output=artifacts/pgk2_graph_logica_validation/slurm-%A_%a.out
#SBATCH --error=artifacts/pgk2_graph_logica_validation/slurm-%A_%a.err

set -euo pipefail

cd "$HOME/pgk2_logica"
module load PyTorch/2.1.2-foss-2022b-CUDA-12.1.1
export PYTHONPATH="/vast/palmer/apps/avx2/software/PyTorch/2.1.2-foss-2022b-CUDA-12.1.1/lib/python3.10/site-packages:$PWD/src"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

round="${ROUND:?Submit with --export=ALL,ROUND=N}"
chunk_size=8000
chunks_per_task=3
total_chunks=31
first_chunk=$((round * 2 * chunks_per_task + SLURM_ARRAY_TASK_ID * chunks_per_task))
if (( first_chunk >= total_chunks )); then
  echo "No chunks remain for round $round task $SLURM_ARRAY_TASK_ID"
  exit 0
fi
last_chunk=$((first_chunk + chunks_per_task))
if (( last_chunk > total_chunks )); then
  last_chunk=$total_chunks
fi

outdir="artifacts/pgk2_graph_logica_validation/chunks"
mkdir -p "$outdir"
for chunk in $(seq "$first_chunk" "$((last_chunk - 1))"); do
  offset=$((chunk * chunk_size))
  .runtime/bin/python scripts/run_pgk2_graph_logica.py \
    --panel validation \
    --candidate-panels DREAM_challenge_2026/Val-Test-set.zip \
    --adapter artifacts/pgk2_graph_logica_competition_finetune/graph_logica_adapter.pt \
    --offset "$offset" \
    --max-compounds "$chunk_size" \
    --batch-size 16 \
    --encoder-batch-size 64 \
    --graph-workers 5 \
    --device cuda \
    --output-dir "$outdir"
done
