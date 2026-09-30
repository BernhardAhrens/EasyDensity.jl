#!/bin/bash
#SBATCH --job-name=easydensity
#SBATCH -p big,work
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=384G
#SBATCH --time=24:00:00
#SBATCH --mail-type=END,FAIL

# Login node:
#   ./slurm/submit_easydensity.sh          SiNN, MultiNN, and UniNN in parallel
#   ./slurm/submit_easydensity.sh smoke    200-row smoke test, one job, all three in order
#   ./slurm/submit_easydensity.sh ensemble MC dropout and a 5-member deep ensemble
#
# A full job uses 128 CPUs, one big node. Each fold has 648 configs and one
# thread per config, so 128 is the largest node size and keeps those threads busy.
# The 64-CPU run peaked near 128 GB, so 384 GB leaves room if memory grows with threads.

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
    mode="${1:-full}"
    if [[ "$mode" == "smoke" ]]; then
        exec sbatch --parsable \
            --job-name=easydensity_smoke \
            --partition=big,work \
            --ntasks=1 \
            --cpus-per-task=4 \
            --mem=32G \
            --time=02:00:00 \
            --export=ALL,EASYDENSITY_SMOKE=1,EASYDENSITY_SMOKE_N=200 \
            --output="${ROOT}/output/easydensity_smoke.out" \
            --error="${ROOT}/output/easydensity_smoke.err" \
            "$0"
    fi
    if [[ "$mode" == "ensemble" ]]; then
        exec sbatch --parsable \
            --job-name=easydensity_ensemble \
            --partition=big,work \
            --ntasks=1 \
            --cpus-per-task=5 \
            --mem=64G \
            --time=12:00:00 \
            --export=ALL,EASYDENSITY_MODEL=ensemble,MLDATADEVICES_SILENCE_WARN_NO_GPU=1 \
            --output="${ROOT}/output/easydensity_ensemble.out" \
            --error="${ROOT}/output/easydensity_ensemble.err" \
            "$0"
    fi
    for model in SiNN MultiNN UniNN; do
        sbatch --parsable \
            --job-name="easydensity_${model}" \
            --partition=big \
            --ntasks=1 \
            --cpus-per-task=128 \
            --mem=384G \
            --time=24:00:00 \
            --export=ALL,EASYDENSITY_MODEL="${model}" \
            --output="${ROOT}/output/easydensity_${model}.out" \
            --error="${ROOT}/output/easydensity_${model}.err" \
            "$0"
    done
    exit 0
fi

export JULIA_NUM_THREADS=${SLURM_CPUS_PER_TASK}
echo "CPU threads: ${JULIA_NUM_THREADS}"
echo "SLURM_NODELIST: ${SLURM_NODELIST}"
echo "EASYDENSITY_SMOKE: ${EASYDENSITY_SMOKE:-0}"
echo "EASYDENSITY_MODEL: ${EASYDENSITY_MODEL:-all}"

run_model() {
    julia --project=. --threads="${SLURM_CPUS_PER_TASK}" "$1"
}

case "${EASYDENSITY_MODEL:-all}" in
    SiNN) run_model SiNN.jl ;;
    MultiNN) run_model MultiNN.jl ;;
    UniNN) run_model UniNN.jl ;;
    ensemble)
        run_model deep_ensemble.jl
        run_model prediction_uncertainty.jl
        ;;
    all)
        run_model SiNN.jl
        run_model MultiNN.jl
        run_model UniNN.jl
        ;;
    *)
        echo "Unknown EASYDENSITY_MODEL=${EASYDENSITY_MODEL}" >&2
        exit 1
        ;;
esac
