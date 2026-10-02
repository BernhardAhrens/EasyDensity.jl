#!/bin/bash
# Submit tuning, then the ensemble, then the figures.
# Ensemble jobs start only after every tuning job succeeds.
# The figure job starts only after every ensemble job succeeds.
#
#   ./slurm/submit_cv.sh
#   ./slurm/submit_cv.sh smoke

cd "$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p output

if [[ "${1:-}" == "smoke" ]]; then
    tune_id=$(sbatch --parsable \
        --job-name=cv_tune_smoke \
        --partition=big,work \
        --ntasks=1 --cpus-per-task=4 --mem=32G --time=02:00:00 \
        --export=ALL,EASYDENSITY_SMOKE=1,EASYDENSITY_SMOKE_N=200,EASYDENSITY_SCRIPT=all \
        --output="output/cv_tune_smoke.out" \
        --error="output/cv_tune_smoke.err" \
        slurm/submit_cv_tune.sh)
    ens_id=$(sbatch --parsable \
        --dependency="afterok:${tune_id}" \
        --job-name=cv_ensemble_smoke \
        --partition=big,work \
        --ntasks=1 --cpus-per-task=2 --mem=32G --time=02:00:00 \
        --export=ALL,EASYDENSITY_SMOKE=1,EASYDENSITY_SMOKE_N=200,EASYDENSITY_SCRIPT=all,MLDATADEVICES_SILENCE_WARN_NO_GPU=1 \
        --output="output/cv_ensemble_smoke.out" \
        --error="output/cv_ensemble_smoke.err" \
        slurm/submit_cv_ensemble.sh)
    echo "tune ${tune_id}"
    echo "ensemble ${ens_id}"
    exit 0
fi

tune_ids=()
for model in UniNN MultiNN SiNN; do
    id=$(sbatch --parsable \
        --job-name="cv_tune_${model}" \
        --partition=big \
        --ntasks=1 --cpus-per-task=128 --mem=384G --time=24:00:00 \
        --export=ALL,EASYDENSITY_SCRIPT="cv/tune_$(echo "$model" | tr '[:upper:]' '[:lower:]').jl" \
        --output="output/cv_tune_${model}.out" \
        --error="output/cv_tune_${model}.err" \
        slurm/submit_cv_tune.sh)
    tune_ids+=("$id")
    echo "tune ${model} ${id}"
done
tune_dep=$(IFS=:; echo "${tune_ids[*]}")

ens_ids=()
for model in UniNN MultiNN SiNN; do
    id=$(sbatch --parsable \
        --dependency="afterok:${tune_dep}" \
        --job-name="cv_ensemble_${model}" \
        --partition=big,work \
        --ntasks=1 --cpus-per-task=20 --mem=192G --time=48:00:00 \
        --export=ALL,EASYDENSITY_SCRIPT="cv/ensemble_$(echo "$model" | tr '[:upper:]' '[:lower:]').jl",MLDATADEVICES_SILENCE_WARN_NO_GPU=1 \
        --output="output/cv_ensemble_${model}.out" \
        --error="output/cv_ensemble_${model}.err" \
        slurm/submit_cv_ensemble.sh)
    ens_ids+=("$id")
    echo "ensemble ${model} ${id}"
done
ens_dep=$(IFS=:; echo "${ens_ids[*]}")

fig_id=$(sbatch --parsable \
    --dependency="afterok:${ens_dep}" \
    --job-name=cv_ensemble_figures \
    --partition=big,work \
    --ntasks=1 --cpus-per-task=4 --mem=32G --time=02:00:00 \
    --export=ALL,EASYDENSITY_SCRIPT=figures,MLDATADEVICES_SILENCE_WARN_NO_GPU=1 \
    --output="output/cv_ensemble_figures.out" \
    --error="output/cv_ensemble_figures.err" \
    slurm/submit_cv_ensemble.sh)
echo "figures ${fig_id}"
