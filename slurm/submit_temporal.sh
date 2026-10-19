#!/bin/bash
# Train on 2018, predict the other survey years, draw the transfer hexbin, sync figures.
# Uses the hyperparameters in eval/cv_*_hyperparams_v20261002.csv.
#
#   ./slurm/submit_temporal.sh
#   ./slurm/submit_temporal.sh smoke

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    cd "${SLURM_SUBMIT_DIR}"
else
    cd "$(cd "$(dirname "$0")/.." && pwd)"
fi
ROOT="$(pwd)"
mkdir -p output

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    if [[ "${1:-}" == "smoke" ]]; then
        exec sbatch --parsable \
            --job-name=temporal_smoke \
            --partition=big,work \
            --ntasks=1 --cpus-per-task=2 --mem=32G --time=02:00:00 \
            --export=ALL,EASYDENSITY_SMOKE=1,EASYDENSITY_SMOKE_N=200,MLDATADEVICES_SILENCE_WARN_NO_GPU=1 \
            --output="${ROOT}/output/temporal_smoke.out" \
            --error="${ROOT}/output/temporal_smoke.err" \
            "$0"
    fi
    exec sbatch --parsable \
        --job-name=temporal_transfer \
        --partition=big,work \
        --ntasks=1 --cpus-per-task=20 --mem=192G --time=48:00:00 \
        --export=ALL,EASYDENSITY_SMOKE=0,MLDATADEVICES_SILENCE_WARN_NO_GPU=1 \
        --output="${ROOT}/output/temporal_transfer.out" \
        --error="${ROOT}/output/temporal_transfer.err" \
        "$0"
fi

set -euo pipefail
export JULIA_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
echo "CPU threads: ${JULIA_NUM_THREADS}"
echo "EASYDENSITY_SMOKE: ${EASYDENSITY_SMOKE:-0}"

source "${SLURM_SUBMIT_DIR}/slurm/run_julia.sh"
run_julia cv/temporal_transfer.jl
run_julia viz-makie/temporal_transferability.jl

rclone sync --ca-cert ~/.config/rclone/certs/ca-bundle-plus-harica.crt results/ nextcloud:people/Xuemeng_Tian/results_EasyDensity/
