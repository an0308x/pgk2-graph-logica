#!/usr/bin/env bash
# Full-panel validation scoring with the protein score localized to the
# compound-21/47 inhibitor pocket instead of a random 15% of all residues.
#
# LOGICA Eq. 2 defines the score over an evaluation position set A, and the
# paper states A "can be restricted to mutation sites, binding-interface
# residues, or other biologically meaningful subsets". Every submission so far
# used a seeded random 15% of the 417 residues, which is neither the full
# sequence nor the interface. This run scores the 26-residue direct pocket and
# the 49-residue context shell so the two can be compared against the existing
# random-mask ranking.
#
# Diagnostic only. Do not submit any ranking from this run before the adapter
# norm pathology in artifacts/pgk2_gate_diagnostic/adapter_norms.json is
# resolved.
#SBATCH --job-name=pgk2-pocket-val
#SBATCH --partition=gpu_devel
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=20G
#SBATCH --time=01:00:00
#SBATCH --array=0-1%2
#SBATCH --output=artifacts/pgk2_pocket_validation_scores/slurm-%A_%a.out
#SBATCH --error=artifacts/pgk2_pocket_validation_scores/slurm-%A_%a.err

set -euo pipefail

cd "$HOME/pgk2_logica"
module load PyTorch/2.1.2-foss-2022b-CUDA-12.1.1
export PYTHONPATH="/vast/palmer/apps/avx2/software/PyTorch/2.1.2-foss-2022b-CUDA-12.1.1/lib/python3.10/site-packages:$PWD/src"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

region="${PGK2_MASK_REGION:-inhibitor_pocket}"
chunk_size=10000
chunks_per_task=13
first_chunk=$((SLURM_ARRAY_TASK_ID * chunks_per_task))
last_chunk=$((first_chunk + chunks_per_task))
if (( last_chunk > 25 )); then
  last_chunk=25
fi

outdir="artifacts/pgk2_pocket_validation_scores/${region}/chunks"
mkdir -p "$outdir"
for chunk in $(seq "$first_chunk" "$((last_chunk - 1))"); do
  offset=$((chunk * chunk_size))
  .runtime/bin/python scripts/run_pgk2_logica.py \
    --panel validation \
    --candidate-panels DREAM_challenge_2026/Val-Test-set.zip \
    --adapter artifacts/pgk2_competition_finetune/finetuned_adapter.pt \
    --protein-mask-region "$region" \
    --offset "$offset" \
    --max-compounds "$chunk_size" \
    --batch-size 16 \
    --encoder-batch-size 64 \
    --device cuda \
    --output-dir "$outdir"
done
