#!/bin/bash
# Deep ensemble only. Run after cv/tune_*.jl has written the hyperparameter files.
#
#   ./slurm/submit_cv_ensemble.sh
#   ./slurm/submit_cv_ensemble.sh smoke

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
            --job-name=cv_ensemble_smoke \
            --partition=big,work \
            --ntasks=1 --cpus-per-task=2 --mem=32G --time=02:00:00 \
            --export=ALL,EASYDENSITY_SMOKE=1,EASYDENSITY_SMOKE_N=200,EASYDENSITY_SCRIPT=all,MLDATADEVICES_SILENCE_WARN_NO_GPU=1 \
            --output="${ROOT}/output/cv_ensemble_smoke.out" \
            --error="${ROOT}/output/cv_ensemble_smoke.err" \
            "$0"
    fi
    ids=()
    for model in UniNN MultiNN SiNN; do
        id=$(sbatch --parsable \
            --job-name="cv_ensemble_${model}" \
            --partition=big,work \
            --ntasks=1 --cpus-per-task=20 --mem=192G --time=48:00:00 \
            --export=ALL,EASYDENSITY_SCRIPT="cv/ensemble_$(echo "$model" | tr '[:upper:]' '[:lower:]').jl",MLDATADEVICES_SILENCE_WARN_NO_GPU=1 \
            --output="${ROOT}/output/cv_ensemble_${model}.out" \
            --error="${ROOT}/output/cv_ensemble_${model}.err" \
            "$0")
        ids+=("$id")
    done
    dep=$(IFS=:; echo "${ids[*]}")
    sbatch --parsable \
        --dependency="afterok:${dep}" \
        --job-name=cv_ensemble_figures \
        --partition=big,work \
        --ntasks=1 --cpus-per-task=4 --mem=32G --time=02:00:00 \
        --export=ALL,EASYDENSITY_SCRIPT=figures,MLDATADEVICES_SILENCE_WARN_NO_GPU=1 \
        --output="${ROOT}/output/cv_ensemble_figures.out" \
        --error="${ROOT}/output/cv_ensemble_figures.err" \
        "$0"
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
    run_one cv/ensemble_uninn.jl
    run_one cv/ensemble_multinn.jl
    run_one cv/ensemble_sinn.jl
elif [[ "${EASYDENSITY_SCRIPT}" == "figures" ]]; then
    run_one cv/summarize.jl
    run_one viz-makie/ensemble_uncertainty.jl
    run_one viz-makie/model_accuracy.jl
    run_one viz-makie/join_distribution.jl
    run_one viz-makie/join_distribution_outliers.jl
    run_one viz-makie/temporal.jl
    run_one viz-makie/porosity.jl
    run_one viz-makie/plausibility_oBD_mBD.jl
else
    run_one "${EASYDENSITY_SCRIPT}"
fi
