# =============================================================================
# UniNN.jl — converted from UniNN.ipynb (per-target NNs, "mend_uniNN")
# =============================================================================
# A separate univariate neural network is trained per target. Each target gets
# its own k-fold CV grid-search hyperparameter optimization.
#
# CHANGE (reviewer UQ request): the per-target hyperparameter search now includes
# a dropout probability `p`, and every candidate network is built WITH dropout
# layers, so each best model supports Monte-Carlo dropout uncertainty
# quantification.
#
# Run the full study on the real LUCAS data, or a fast smoke test with synthetic
# data via `EASYDENSITY_SMOKE=1`.

using EasyHybrid
using Lux, NNlib, Optimisers, LuxCore
using CSV, DataFrames, Random, Statistics, JLD2

include(joinpath(@__DIR__, "dropout_uq.jl"))

testid = "mend_uniNN_dropout"
version = "v20251209"
target_names = [:BD, :SOCconc, :CF, :SOCdensity]
results_dir = joinpath(@__DIR__, "eval"); mkpath(results_dir)
models_dir = joinpath(@__DIR__, "models"); mkpath(models_dir)

scalers = Dict(:SOCconc => 0.151, :CF => 0.263, :BD => 0.529, :SOCdensity => 0.167)

# ---------------------------------------------------------------------------
# Data + search space
# ---------------------------------------------------------------------------
if smoke_mode()
    df, predictors = synthetic_lucas(400; nfeatures = 6, targets = target_names)
    k = 2; nepochs = 15; patience = 5
    hidden_configs = [(16, 8)]
    batch_sizes = [64]; lrs = [1e-2]; activations = [relu]
    dropouts = [0.2]
else
    df = CSV.read(joinpath(@__DIR__, "data/lucas_preprocessed_v20251125.csv"), DataFrame; normalizenames = true)
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
@info "UniNN: $(length(configs)) configs over $nf predictors, k=$k, targets=$target_names"

# ---------------------------------------------------------------------------
# Per-target HPO (with dropout), retrain best, MC-dropout UQ
# ---------------------------------------------------------------------------
summary = DataFrame(target = Symbol[], h = String[], bs = Int[], lr = Float64[],
    act = String[], p = Float64[], val_loss = Float64[], mc_mean_std = Float64[])
unc = DataFrame()

for tgt in target_names
    @info "===== Target $tgt ====="
    df_t = dropmissing(df, tgt)
    if nrow(df_t) == 0
        @warn "No rows for $tgt, skipping."
        continue
    end

    build_model(cfg) = constructNNModel(
        predictors, [tgt];
        hidden_layers = make_dropout_chain(cfg.h, cfg.act, cfg.p),
        activation = cfg.act, scale_nn_outputs = true, input_batchnorm = false,
    )

    fold_best, folds = cv_grid_hpo(build_model, df_t, configs; k = k, nepochs = nepochs, seed = 42, patience = patience)
    best = overall_best(fold_best)
    @info "Best config for $tgt" cfg = best.cfg val_loss = best.loss

    best_model, best_res = retrain_best(build_model, best.cfg, df_t; nepochs = nepochs, seed = 42, patience = patience)
    jldsave(joinpath(models_dir, "$(testid)_$(tgt)_best_$version.jld2"); cfg = best.cfg, ps = best_res.ps, st = best_res.st)

    u = estimate_uncertainty(MCDropout(n_samples = smoke_mode() ? 30 : 100), best_model, df_t, best_res)
    mc_std = mean(u.std[tgt])
    @info "MC-dropout mean σ for $tgt = $(round(mc_std; digits = 4))"

    push!(summary, (tgt, string(best.cfg.h), best.cfg.bs, best.cfg.lr,
        string(best.cfg.act), best.cfg.p, best.loss, mc_std))
end

CSV.write(joinpath(results_dir, "$(testid)_summary_$version.csv"), summary)
@info "UniNN done." summary
