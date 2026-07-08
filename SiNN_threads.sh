#!/bin/bash
# =============================================================================
# Single node, multithreaded run of SiNN.jl
#
# One Slurm job. Outer CV folds run serially; the hyperparameter configs of
# each fold are trained in parallel with Julia threads (Threads.@threads).
#
# Submit from the EasyDensity.jl project root:
#   sbatch SiNN_threads.sh
# =============================================================================
#SBATCH --job-name=SiNN_threads
#SBATCH -p big
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128                 # threads used for the config sweep
#SBATCH --mem=1000G                         # NN sweep + data replicated per thread
#SBATCH --time=48:00:00
#SBATCH --output=output/SiNN_threads.out
#SBATCH --error=output/SiNN_threads.err
#SBATCH --mail-type END,FAIL

mkdir -p output

# Avoid BLAS over-subscription when the config loop already spawns threads
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export JULIA_NUM_THREADS=${SLURM_CPUS_PER_TASK}

echo "CPU threads: ${JULIA_NUM_THREADS}"
echo "SLURM_NODELIST: ${SLURM_NODELIST}"
echo "Running SiNN hyperparameter search (all folds)…"

julia --threads=${SLURM_CPUS_PER_TASK} SiNN.jl
