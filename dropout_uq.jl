# =============================================================================
# Shared helpers for dropout-enabled hyperparameter optimization + MC-dropout UQ
# =============================================================================
#
# Used by the converted notebook scripts `SiNN.jl`, `UniNN.jl` and `MultiNN.jl`.
#
# Motivation (reviewer request): perform aleatoric/epistemic uncertainty
# quantification. To enable **Monte-Carlo dropout** later, the hyperparameter
# search now (a) includes a dropout probability in the search space and
# (b) builds every candidate network with `Dropout` layers. The best
# configuration is then retrained with dropout so `EasyHybrid.estimate_uncertainty`
# with `MCDropout(...)` can be applied.
#
# MC dropout requires the EasyHybrid uncertainty API (branch
# `cursor/aleatoric-uncertainty-quantification-c142`, PR #2).

using EasyHybrid
using EasyHybrid: constructNNModel
using LuxCore
using Random, Statistics, DataFrames

"""
    make_dropout_chain(h, act, p) -> Chain

Build the *hidden* portion of a network for `constructNNModel` /
`constructHybridModel` as a `Chain` of `Dense(h[i] => h[i+1], act)` blocks with a
`Dropout(p)` after each hidden layer. `prepare_hidden_chain` then wraps this with
the input projection `Dense(in_dim => h[1])` and the output layer
`Dense(h[end] => out_dim)`, so the resulting architecture matches the original
`collect(h)` sizes with dropout inserted after every hidden layer.

`p == 0` yields a dropout-free network (useful as a baseline in the search).
"""
function make_dropout_chain(h, act, p)
    layers = Any[]
    for i in 1:(length(h) - 1)
        push!(layers, Dense(h[i], h[i + 1], act))
        p > 0 && push!(layers, Dropout(Float32(p)))
    end
    if isempty(layers)                     # single hidden size -> one Dense (+dropout)
        push!(layers, Dense(h[1], h[1], act))
        p > 0 && push!(layers, Dropout(Float32(p)))
    end
    return Chain(layers...)
end

"""
    build_configs(hidden_configs, batch_sizes, lrs, activations, dropouts) -> Vector{NamedTuple}

Cartesian product of the search space, now including a dropout probability `p`.
"""
function build_configs(hidden_configs, batch_sizes, lrs, activations, dropouts)
    return [(h = h, bs = bs, lr = lr, act = act, p = p)
            for h in hidden_configs
            for bs in batch_sizes
            for lr in lrs
            for act in activations
            for p in dropouts]
end

"""
    cv_grid_hpo(build_model, df, configs; k, nepochs, seed, patience, train_kwargs...)

Nested k-fold grid-search hyperparameter optimization, mirroring the original
notebooks: for each outer fold, every configuration is trained (with an internal
train/validation split) and the configuration with the lowest validation loss is
kept. `build_model(cfg)` must return a fresh model for a given config (with
dropout). Returns a vector of `(; loss, cfg, res, model)` — the best per fold.
"""
function cv_grid_hpo(build_model, df, configs; k = 5, nepochs = 200, seed = 42,
        patience = 15, train_kwargs...)
    Random.seed!(seed)
    folds = make_folds(df, k = k, shuffle = true)
    fold_best = Vector{NamedTuple}(undef, k)

    for test_fold in 1:k
        @info "Outer fold $test_fold / $k"
        train_idx = findall(!=(test_fold), folds)
        train_df = df[train_idx, :]

        best = (; loss = Inf, cfg = nothing, res = nothing, model = nothing)
        for cfg in configs
            m = build_model(cfg)
            res = train(m, train_df, ();
                nepochs = nepochs, batchsize = cfg.bs, opt = AdamW(cfg.lr),
                training_loss = :mse, loss_types = [:mse, :r2], shuffleobs = true,
                random_seed = seed, patience = patience, agg = mean,
                return_model = :best, show_progress = false, plotting = false,
                save_training = false, train_kwargs...)
            res === nothing && continue
            if res.best_loss < best.loss
                best = (; loss = res.best_loss, cfg = cfg, res = res, model = deepcopy(m))
            end
        end
        fold_best[test_fold] = best
    end
    return fold_best, folds
end

"""
    oof_predictions(fold_best, folds, df, targets) -> DataFrame

Out-of-fold predictions: for each outer fold, predict the held-out rows with that
fold's best (dropout) model in **test mode** (dropout off) and collect
`pred_<target>` columns. Mirrors the notebooks' cross-validated prediction export.
"""
function oof_predictions(fold_best, folds, df, targets)
    pieces = DataFrame[]
    for test_fold in eachindex(fold_best)
        fb = fold_best[test_fold]
        fb.model === nothing && continue
        test_df = df[findall(==(test_fold), folds), :]
        x_test = prepare_data(fb.model, test_df)[1]
        ŷ, _ = fb.model(x_test, fb.res.ps, LuxCore.testmode(fb.res.st))
        out = copy(test_df)
        for t in targets
            if hasproperty(ŷ, t)
                val = getproperty(ŷ, t)
                if val isa AbstractVector && length(val) == nrow(test_df)
                    out[!, Symbol("pred_", t)] = collect(val)
                elseif (val isa Number) || (val isa AbstractVector && length(val) == 1)
                    out[!, Symbol("pred_", t)] = fill(Float32(val isa AbstractVector ? first(val) : val), nrow(test_df))
                end
            end
        end
        push!(pieces, out)
    end
    return isempty(pieces) ? DataFrame() : vcat(pieces...; cols = :union)
end

"""
    overall_best(fold_best) -> NamedTuple

Pick the single best `(; loss, cfg, res, model)` across folds (lowest validation loss).
"""
overall_best(fold_best) = fold_best[argmin([fb.loss for fb in fold_best])]

"""
    retrain_best(build_model, cfg, df; nepochs, seed, patience, train_kwargs...)

Retrain the best configuration (with dropout) on `df` and return the `TrainResults`
plus the freshly built model. This is the model to use for MC-dropout UQ.
"""
function retrain_best(build_model, cfg, df; nepochs = 200, seed = 42, patience = 15, train_kwargs...)
    model = build_model(cfg)
    res = train(model, df, ();
        nepochs = nepochs, batchsize = cfg.bs, opt = AdamW(cfg.lr),
        training_loss = :mse, loss_types = [:mse, :r2], shuffleobs = true,
        random_seed = seed, patience = patience, agg = mean,
        return_model = :best, show_progress = false, plotting = false,
        save_training = false, train_kwargs...)
    return model, res
end

# --- Smoke-test utilities ----------------------------------------------------
# When the real LUCAS CSV is unavailable (e.g. CI), generate a small synthetic
# dataset with the same target columns so the whole pipeline can be exercised.
smoke_mode() = get(ENV, "EASYDENSITY_SMOKE", "0") == "1"

"""
    synthetic_lucas(n; nfeatures, targets, seed) -> (df, predictors)

Small synthetic dataset with `nfeatures` predictor columns `f1..fN` and the
requested `targets`, for smoke-testing the pipeline without the real data.
"""
function synthetic_lucas(n; nfeatures = 6, targets = [:BD, :SOCconc, :CF, :SOCdensity], seed = 1)
    rng = MersenneTwister(seed)
    df = DataFrame()
    X = randn(rng, Float32, n, nfeatures)
    predictors = [Symbol("f", i) for i in 1:nfeatures]
    for (i, p) in enumerate(predictors)
        df[!, p] = X[:, i]
    end
    w = randn(rng, Float32, nfeatures)
    base = X * w
    for t in targets
        df[!, t] = Float32.(0.2 .* base .+ 0.1 .* randn(rng, Float32, n) .+ 0.5)
    end
    return df, predictors
end
