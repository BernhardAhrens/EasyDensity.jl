# =============================================================================
# SiNN.jl — converted from SiNN.ipynb (single-NN hybrid, "mend_hybridNN")
# =============================================================================
# A single neural network predicts the parameters (SOCconc, CF, mBD, oBD) that
# feed the mechanistic `SOCD_model`, which returns BD / SOCconc / CF / SOCdensity.
#
# CHANGE (reviewer UQ request): the hyperparameter search now includes a dropout
# probability `p`, and every candidate network is built WITH dropout layers, so
# the best hybrid model supports Monte-Carlo dropout uncertainty quantification.
# (The hybrid model has no estimated global parameters, so MC dropout applies.)
#
# Run the full study on the real LUCAS data, or a fast smoke test with synthetic
# data via `EASYDENSITY_SMOKE=1`.

using EasyHybrid
using Lux, NNlib, Optimisers, LuxCore
using CSV, DataFrames, Random, Statistics, JLD2

include(joinpath(@__DIR__, "dropout_uq.jl"))

testid = "mend_hybridNN_dropout"
version = "v20251209"
results_dir = joinpath(@__DIR__, "eval"); mkpath(results_dir)
models_dir = joinpath(@__DIR__, "models"); mkpath(models_dir)

# scales
scalers = Dict(
    :SOCconc => 0.151, :CF => 0.263, :BD => 0.529, :SOCdensity => 0.167,
)

# mechanistic model (unchanged from the notebook)
function SOCD_model(; SOCconc, CF, oBD, mBD)
    ϵ = 1.0e-7
    soct = (exp.(SOCconc ./ scalers[:SOCconc]) .- 1) ./ 1000
    soct = clamp.(soct, ϵ, Inf)
    cft = (exp.(CF ./ scalers[:CF]) .- 1) ./ 100
    cft = clamp.(cft, 0, 0.99)
    som = 1.724f0 .* soct
    som = clamp.(som, 0, 1)
    denom = som .* mBD .+ (1.0f0 .- som) .* oBD
    BD = (oBD .* mBD) ./ denom
    BD = clamp.(BD, ϵ, Inf)
    SOCdensity = soct .* 1000 .* BD .* (1 .- cft)
    SOCdensity = clamp.(SOCdensity, 1, Inf)
    SOCdensity = log.(SOCdensity) .* scalers[:SOCdensity]
    BD = BD .* scalers[:BD]
    return (; BD, SOCconc, CF, SOCdensity, oBD, mBD)
end

# parameter bounds (default, lower, upper)
parameters = (
    SOCconc = (0.01f0, 0.0f0, 1.0f0),
    CF      = (0.15f0, 0.0f0, 1.0f0),
    oBD     = (0.20f0, 0.05f0, 0.40f0),
    mBD     = (1.20f0, 0.75f0, 2.0f0),
)
neural_param_names = [:SOCconc, :CF, :mBD, :oBD]
global_param_names = Symbol[]        # none estimated globally -> MC dropout applies
forcing = Symbol[]
targets = [:BD, :SOCconc, :SOCdensity, :CF]

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
@info "SiNN: $(length(configs)) configs over $nf predictors, k=$k"

# ---------------------------------------------------------------------------
# Model builder: single-NN hybrid WITH dropout
# ---------------------------------------------------------------------------
build_model(cfg) = constructHybridModel(
    predictors, forcing, targets, SOCD_model, parameters,
    neural_param_names, global_param_names;
    hidden_layers = make_dropout_chain(cfg.h, cfg.act, cfg.p),
    activation = cfg.act, scale_nn_outputs = true, input_batchnorm = false,
    start_from_default = true,
)

# ---------------------------------------------------------------------------
# Hyperparameter optimization (k-fold CV grid search) with dropout
# ---------------------------------------------------------------------------
fold_best, folds = cv_grid_hpo(build_model, df, configs; k = k, nepochs = nepochs, seed = 42,
    patience = patience, monitor_names = [:oBD, :mBD])
best = overall_best(fold_best)
@info "Best config" cfg = best.cfg val_loss = best.loss

rlt_pred = oof_predictions(fold_best, folds, df, [:BD, :SOCconc, :CF, :SOCdensity, :oBD, :mBD])
CSV.write(joinpath(results_dir, "$(testid)_cv.pred_$version.csv"), rlt_pred)

# ---------------------------------------------------------------------------
# Retrain best config WITH dropout, then MC-dropout uncertainty quantification
# ---------------------------------------------------------------------------
best_model, best_res = retrain_best(build_model, best.cfg, df; nepochs = nepochs, seed = 42,
    patience = patience, monitor_names = [:oBD, :mBD])
jldsave(joinpath(models_dir, "$(testid)_best_$version.jld2"); cfg = best.cfg, ps = best_res.ps, st = best_res.st)

u = estimate_uncertainty(MCDropout(n_samples = smoke_mode() ? 30 : 100), best_model, df, best_res)
@info "MC-dropout mean σ per target" (; (t => round(mean(u.std[t]); digits = 4) for t in targets)...)

unc = DataFrame()
for t in targets
    unc[!, Symbol(t, "_mean")]  = u.mean[t]
    unc[!, Symbol(t, "_std")]   = u.std[t]
    unc[!, Symbol(t, "_lower")] = u.lower[t]
    unc[!, Symbol(t, "_upper")] = u.upper[t]
end
CSV.write(joinpath(results_dir, "$(testid)_mcdropout_$version.csv"), unc)
@info "SiNN done."
