# EasyDensity.jl — SiNN hybrid bulk-density / SOC model

`SiNN.jl` trains a hybrid neural + mechanistic model (`SOCD_model`) that jointly
predicts bulk density (`BD`), SOC concentration (`SOCconc`), coarse fragments
(`CF`) and SOC density (`SOCdensity`). It runs an exhaustive hyperparameter grid
search inside a 5-fold cross-validation. For each outer fold the configs are
trained in parallel with Julia threads (via `tune`); the best config per fold is
kept for out-of-fold predictions.

## Requirements

- Julia (matching the version used to build `Manifest.toml`).
- The `EasyHybrid_Porosity` package checked out **next to** this repo, i.e. the
  two folders share a parent directory:

  ```
  <parent>/
    ├── EasyDensity.jl/        # this repo
    └── EasyHybrid_Porosity/   # the hybrid-modelling package (dev'd by SiNN.jl)
  ```

  `SiNN.jl` calls `Pkg.develop` on `../EasyHybrid_Porosity` and
  `Pkg.instantiate()` at startup, so dependencies (including `Hyperopt`, pulled
  in transitively by `EasyHybrid`) are resolved automatically on first run.

## Input data

`SiNN.jl` reads `data/lucas_preprocessed_v20251125.csv`. The predictor columns
are selected positionally with `predictors = Symbol.(names(df))[18:end-6]` —
**re-check this slice whenever the input file/columns change** (see the
`# CHECK EVERY TIME` comment in `SiNN.jl`).

## How to run

### On an HPC cluster (Slurm) — recommended

Submit from the project root:

```bash
sbatch SiNN_threads.sh
```

This runs one job: outer CV folds serial, the hyperparameter configs of each
fold trained in parallel with Julia threads. Edit the `#SBATCH` header in
`SiNN_threads.sh` to match your cluster (partition, `--cpus-per-task`, `--mem`,
`--time`). `JULIA_NUM_THREADS` is set from `--cpus-per-task`, and BLAS/OMP/MKL
threads are pinned to 1 to avoid over-subscription while the config loop is
already multithreaded.

Job stdout/stderr go to `output/SiNN_threads.out` / `.err`.

### Locally / interactively

Run with as many threads as you want to devote to the config sweep:

```bash
julia --threads=8 SiNN.jl
```

Plotting is headless (`ENV["GKSwstype"] = "100"`), so it works fine over SSH
without an X server; figures are written to disk (see Outputs).

## Outputs

All paths are relative to the project root:

- `eval/mend_hybridNN_cv.pred_v20251209.csv` — out-of-fold predictions
  (`pred_*` columns) for the best config of each fold.
- `eval/mend_hybridNN_hyperparams_v20251209.csv` — the winning hyperparameters
  and validation metrics per fold.
- `eval/plots/scatter_<target>_v20251209.png` — predicted-vs-true 2D histograms.
- `eval/plots/hist_pred_<target>_v20251209.png` — histograms of predictions.
- `model/best_model_<testid>_config<i>_fold<f>.jld2` — saved best model per fold.
- `output_tmp/` — per-config scratch, wiped after each fold.

(`output_tmp*`, `output/`, `model/` and `eval/` are git-ignored.)

## Tuning the run

Key knobs live near the top of `SiNN.jl`:

- `hidden_configs`, `batch_sizes`, `lrs`, `activations` — the search grid
  (their Cartesian product is the number of configs trained per fold).
- `k` — number of CV folds.
- `nepochs`, `patience` — passed to `tune` in the training loop.
- `testid` / `version` — string tags used in all output filenames.
