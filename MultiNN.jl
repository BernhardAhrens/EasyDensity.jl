# =============================================================================
# MultiNN.jl — converted from MultiNN.ipynb (02 - multivariate NN)
# =============================================================================
# A single multivariate neural network predicts all targets directly.
#
# CHANGE (reviewer UQ request): the hyperparameter search now includes a dropout
# probability `p`, and every candidate network is built WITH dropout layers, so
# the best model supports Monte-Carlo dropout uncertainty quantification.
#
# Run the full study on the real LUCAS data, or a fast smoke test with synthetic
# data via `EASYDENSITY_SMOKE=1`.
#
# NOTE: MC dropout requires EasyHybrid's uncertainty API (PR #2, branch
# `cursor/aleatoric-uncertainty-quantification-c142`).

using EasyHybrid
using Lux, NNlib, Optimisers, LuxCore
using CSV, DataFrames, Random, Statistics, JLD2

include(joinpath(@__DIR__, "dropout_uq.jl"))

testid = "02_multiNN_dropout"
version = "v20251125"
targets = [:BD, :SOCconc, :SOCdensity, :CF]
results_dir = joinpath(@__DIR__, "eval"); mkpath(results_dir)
models_dir = joinpath(@__DIR__, "models"); mkpath(models_dir)

# scales (kept from the notebook for reference)
scalers = Dict(
    :SOCconc => 0.151, :CF => 0.263, :BD => 0.529, :SOCdensity => 0.167,
)

# ---------------------------------------------------------------------------
# Data + search space
# ---------------------------------------------------------------------------
if smoke_mode()
    df, predictors = synthetic_lucas(400; nfeatures = 6, targets = targets)
    k = 2; nepochs = 15; patience = 5
    hidden_configs = [(16, 8)]
    batch_sizes = [64]; lrs = [1e-2]; activations = [relu]
    dropouts = [0.2]
else
    df = CSV.read(joinpath(@__DIR__, "data/lucas_preprocessed_$version.csv"), DataFrame; normalizenames = true)
    predictors = Symbol.(names(df))[18:(end - 6)]   # CHECK EVERY TIME
    k = 5; nepochs = 200; patience = 15
    hidden_configs = [
        (512, 256, 128, 64, 32, 16), (512, 256, 128, 64, 32),
        (256, 128, 64, 32, 16), (256, 128, 64, 32), (256, 128, 64),
        (128, 64, 32, 16), (128, 64, 32), (64, 32, 16),
    ]
    batch_sizes = [128, 256, 512]
    lrs = [1e-3, 5e-4, 1e-4]
    activations = [relu, swish, gelu]
    dropouts = [0.1, 0.2, 0.3]        # NEW: dropout in the search space (MC-dropout UQ)
end
nf = length(predictors)
configs = build_configs(hidden_configs, batch_sizes, lrs, activations, dropouts)
@info "MultiNN: $(length(configs)) configs over $nf predictors, k=$k"

# ---------------------------------------------------------------------------
# Model builder: multivariate NN WITH dropout
# ---------------------------------------------------------------------------
build_model(cfg) = constructNNModel(
    predictors, targets;
    hidden_layers = make_dropout_chain(cfg.h, cfg.act, cfg.p),
    activation = cfg.act, scale_nn_outputs = true, input_batchnorm = false,
)

# ---------------------------------------------------------------------------
# Hyperparameter optimization (k-fold CV grid search) with dropout
# ---------------------------------------------------------------------------
fold_best, folds = cv_grid_hpo(build_model, df, configs; k = k, nepochs = nepochs, seed = 42, patience = patience)
best = overall_best(fold_best)
@info "Best config" cfg = best.cfg val_loss = best.loss

# out-of-fold cross-validated predictions
rlt_pred = oof_predictions(fold_best, folds, df, targets)
CSV.write(joinpath(results_dir, "$(testid)_cv.pred_$version.csv"), rlt_pred)

# ---------------------------------------------------------------------------
# Retrain best config WITH dropout, then MC-dropout uncertainty quantification
# ---------------------------------------------------------------------------
best_model, best_res = retrain_best(build_model, best.cfg, df; nepochs = nepochs, seed = 42, patience = patience)
jldsave(joinpath(models_dir, "$(testid)_best_$version.jld2"); cfg = best.cfg, ps = best_res.ps, st = best_res.st)

u = estimate_uncertainty(MCDropout(n_samples = smoke_mode() ? 30 : 100), best_model, df, best_res)
@info "MC-dropout mean σ per target" (; (t => round(mean(u.std[t]); digits = 4) for t in targets)...)

# save per-observation predictive mean/std/interval per target
unc = DataFrame()
for t in targets
    unc[!, Symbol(t, "_mean")]  = u.mean[t]
    unc[!, Symbol(t, "_std")]   = u.std[t]
    unc[!, Symbol(t, "_lower")] = u.lower[t]
    unc[!, Symbol(t, "_upper")] = u.upper[t]
end
CSV.write(joinpath(results_dir, "$(testid)_mcdropout_$version.csv"), unc)
@info "MultiNN done."
