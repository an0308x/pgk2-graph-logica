#!/usr/bin/env bash
# Override --time using the measured smoke recommendation before submission.
#SBATCH --job-name=pgk2-full-logica-v3
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a5000:1
#SBATCH --exclude=r109u04n01
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=2-00:00:00
#SBATCH --output=artifacts/pgk2_full_logica_v3_logs/train-%j.out
#SBATCH --error=artifacts/pgk2_full_logica_v3_logs/train-%j.err
set -euo pipefail
cd /vast/palmer/home.mccleary/an923/pgk2_logica
module load PyTorch/2.1.2-foss-2022b-CUDA-12.1.1
export PYTHONPATH="/vast/palmer/apps/avx2/software/PyTorch/2.1.2-foss-2022b-CUDA-12.1.1/lib/python3.10/site-packages:$PWD/src"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 POLARS_MAX_THREADS=4
.runtime/bin/python scripts/check_pgk2_full_logica_smoke.py \
  --manifest artifacts/pgk2_full_logica_v3_manifest/manifest.json \
  --smoke-report artifacts/pgk2_full_logica_v3_smoke/report.json
tar -czf "artifacts/pgk2_full_logica_v3_logs/source-${SLURM_JOB_ID}.tar.gz" \
  src/logica_binding scripts/run_pgk2_full_logica.py \
  scripts/check_pgk2_full_logica_smoke.py scripts/prepare_pgk2_full_logica_manifest.py \
  scripts/build_pgk2_canonical_index.py scripts/submit_pgk2_full_logica_train.slurm
resume_args=()
if [[ "${1:-}" == "--resume" ]]; then resume_args=(--resume); fi
.runtime/bin/python -u scripts/run_pgk2_full_logica.py \
  --manifest artifacts/pgk2_full_logica_v3_manifest/manifest.json \
  --output-dir artifacts/pgk2_full_logica_v3_train \
  --device cuda --batch-size 64 --rank-batch-size 32 --rank-every 4 \
  --epochs 1 --threads 4 "${resume_args[@]}"
