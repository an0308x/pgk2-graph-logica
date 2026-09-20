#!/usr/bin/env bash
#SBATCH --job-name=pgk2-dev-v4
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=r109u04n01
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --array=0-8%2
#SBATCH --output=artifacts/pgk2_full_logica_v3_logs/dev-v4-%A_%a.out
#SBATCH --error=artifacts/pgk2_full_logica_v3_logs/dev-v4-%A_%a.err
set -euo pipefail
cd /vast/palmer/home.mccleary/an923/pgk2_logica
module load PyTorch/2.1.2-foss-2022b-CUDA-12.1.1
export PYTHONPATH="/vast/palmer/apps/avx2/software/PyTorch/2.1.2-foss-2022b-CUDA-12.1.1/lib/python3.10/site-packages:$PWD/src"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 POLARS_MAX_THREADS=4
task_index=${SLURM_ARRAY_TASK_ID:?}
caps=(0 1 5)
cap=${caps[$((task_index / 3))]}
arm=$((task_index % 3))
model=graph
aux=0
if (( arm == 0 )); then model=fingerprint; fi
if (( arm == 2 )); then aux=0.01; fi
.runtime/bin/python -u scripts/run_pgk2_development.py \
  --manifest artifacts/pgk2_full_logica_v3_manifest/manifest.json \
  --output-dir "artifacts/pgk2_dev_v4_cap${cap}_${model}_aux${aux}" \
  --model "$model" --zero-cap "$cap" --auxiliary-weight "$aux" \
  --passes 2 --eval-every-shards 16 --patience 4 --device cuda
