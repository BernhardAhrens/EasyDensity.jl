#!/bin/bash
# Deep ensemble only. Run after cv/tune_*.jl has written the hyperparameter files.
#
#   ./slurm/submit_cv_ensemble.sh
#   ./slurm/submit_cv_ensemble.sh 100
#   ./slurm/submit_cv_ensemble.sh smoke
#
# The member count defaults to 100. The run date is the day this script is submitted.
# Hyperparameters and the observation table are copied into results/YYYYMMDD/.

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

copy_inputs() {
    local day="${EASYDENSITY_VERSION#v}"
    local dest="${ROOT}/results/${day}"
    local hp="${EASYDENSITY_HYPERPARAMS_VERSION:-v20261002}"
    if [[ "${EASYDENSITY_SMOKE:-0}" == 1 && -z "${EASYDENSITY_HYPERPARAMS_VERSION:-}" ]]; then
        hp="v20261002_smoke"
    fi
    local hp_day="${hp#v}"
    mkdir -p "$dest"
    local model src
    for model in UniNN MultiNN SiNN; do
        [[ -f "${dest}/cv_${model}_hyperparams.csv" ]] && continue
        src="${ROOT}/results/${hp_day}/cv_${model}_hyperparams.csv"
        [[ -f "$src" ]] || src="${ROOT}/eval/cv_${model}_hyperparams_${hp}.csv"
        [[ -f "$src" ]] || { echo "missing hyperparameters for ${model} (${hp})" >&2; return 1; }
        cp -a "$src" "${dest}/cv_${model}_hyperparams.csv"
    done
    if [[ ! -f "${dest}/all_cv.pred_with.lc.csv" ]]; then
        [[ -f "${ROOT}/eval/all_cv.pred_with.lc_v20251209.csv" ]] || { echo "missing observation table" >&2; return 1; }
        cp -a "${ROOT}/eval/all_cv.pred_with.lc_v20251209.csv" "${dest}/all_cv.pred_with.lc.csv"
    fi
}

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    n_ensemble="${EASYDENSITY_N_ENSEMBLE:-100}"
    smoke=0
    for arg in "$@"; do
        if [[ "$arg" == "smoke" ]]; then
            smoke=1
        elif [[ "$arg" =~ ^[1-9][0-9]*$ ]]; then
            n_ensemble="$arg"
        else
            echo "usage: $0 [smoke] [N]" >&2
            exit 1
        fi
    done
    if [[ -z "${EASYDENSITY_VERSION:-}" ]]; then
        EASYDENSITY_VERSION="v$(date +%Y%m%d)"
    fi
    if [[ "$smoke" == 1 && "$EASYDENSITY_VERSION" != *_smoke ]]; then
        EASYDENSITY_VERSION="${EASYDENSITY_VERSION}_smoke"
    fi
    export EASYDENSITY_VERSION EASYDENSITY_N_ENSEMBLE="$n_ensemble"
    [[ "$smoke" == 1 ]] && export EASYDENSITY_SMOKE=1
    set -euo pipefail
    copy_inputs

    if [[ "$smoke" == 1 ]]; then
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
        --ntasks=1 --cpus-per-task=4 --mem=64G --time=04:00:00 \
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
echo "EASYDENSITY_N_ENSEMBLE: ${EASYDENSITY_N_ENSEMBLE:-100}"
echo "EASYDENSITY_VERSION: ${EASYDENSITY_VERSION}"
echo "EASYDENSITY_SCRIPT: ${EASYDENSITY_SCRIPT}"
copy_inputs

source "${SLURM_SUBMIT_DIR}/slurm/run_julia.sh"
run_one() {
    run_julia "$1"
}

if [[ "${EASYDENSITY_SCRIPT}" == "all" ]]; then
    run_one cv/ensemble_uninn.jl
    run_one cv/ensemble_multinn.jl
    run_one cv/ensemble_sinn.jl
elif [[ "${EASYDENSITY_SCRIPT}" == "figures" ]]; then
    run_one cv/summarize.jl
    run_one cv/ensemble_uncertainty.jl
    run_one viz-makie/ensemble_calibration.jl
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

rclone sync --ca-cert ~/.config/rclone/certs/ca-bundle-plus-harica.crt results/ nextcloud:people/Xuemeng_Tian/results_EasyDensity/
