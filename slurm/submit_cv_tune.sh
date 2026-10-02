#!/bin/bash
# Hyperparameter tuning only. Does not train the ensemble.
#
#   ./slurm/submit_cv_tune.sh
#   ./slurm/submit_cv_tune.sh smoke

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
            --job-name=cv_tune_smoke \
            --partition=big,work \
            --ntasks=1 --cpus-per-task=4 --mem=32G --time=02:00:00 \
            --export=ALL,EASYDENSITY_SMOKE=1,EASYDENSITY_SMOKE_N=200,EASYDENSITY_SCRIPT=all \
            --output="${ROOT}/output/cv_tune_smoke.out" \
            --error="${ROOT}/output/cv_tune_smoke.err" \
            "$0"
    fi
    for model in UniNN MultiNN SiNN; do
        sbatch --parsable \
            --job-name="cv_tune_${model}" \
            --partition=big \
            --ntasks=1 --cpus-per-task=128 --mem=384G --time=24:00:00 \
            --export=ALL,EASYDENSITY_SCRIPT="cv/tune_$(echo "$model" | tr '[:upper:]' '[:lower:]').jl" \
            --output="${ROOT}/output/cv_tune_${model}.out" \
            --error="${ROOT}/output/cv_tune_${model}.err" \
            "$0"
    done
    exit 0
fi

set -euo pipefail
export JULIA_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
echo "CPU threads: ${JULIA_NUM_THREADS}"
echo "EASYDENSITY_SMOKE: ${EASYDENSITY_SMOKE:-0}"
echo "EASYDENSITY_SCRIPT: ${EASYDENSITY_SCRIPT}"

run_one() {
    julia --project=. --threads="${SLURM_CPUS_PER_TASK}" "$1"
}

if [[ "${EASYDENSITY_SCRIPT}" == "all" ]]; then
    run_one cv/tune_uninn.jl
    run_one cv/tune_multinn.jl
    run_one cv/tune_sinn.jl
else
    run_one "${EASYDENSITY_SCRIPT}"
fi
