using EasyHybrid, Lux, NNlib, Optimisers, CSV, DataFrames, Statistics

include(joinpath(@__DIR__, "dropout_uq.jl"))

const ACTS = Dict("relu" => relu, "swish" => swish, "gelu" => gelu)
const MULTI_TARGETS = [:BD, :SOCconc, :SOCdensity, :CF]
const SINN_TARGETS = [:BD, :SOCconc, :SOCdensity, :CF]

function n_ensemble(study)
    default = study.smoke ? "2" : "5"
    return parse(Int, get(ENV, "EASYDENSITY_N_ENSEMBLE", default))
end

function parse_hidden(text)
    ex = Meta.parse(String(text))
    nums = ex isa Expr ? ex.args : [ex]
    return Tuple(Int.(nums))
end

function parse_act(text)
    key = split(String(text), '.')[end]
    haskey(ACTS, key) && return ACTS[key]
    name = Symbol(key)
    isdefined(NNlib, name) || error("unknown activation $text")
    return getproperty(NNlib, name)
end

function parse_cfg(row)
    return (;
        h = parse_hidden(row.h),
        bs = Int(row.bs),
        lr = Float64(row.lr),
        act = parse_act(row.act),
        p = Float64(row.p),
    )
end

function study_folds(df, k)
    Random.seed!(42)
    return make_folds(df, k = k, shuffle = true)
end

function train_kwargs(study, cfg; monitor)
    return (;
        nepochs = study.nepochs,
        batchsize = cfg.bs,
        opt = AdamW(cfg.lr),
        training_loss = :mse,
        loss_types = [:mse, :r2],
        shuffleobs = true,
        patience = study.patience,
        agg = mean,
        return_model = :best,
        show_progress = false,
        plotting = false,
        save_training = false,
        monitor_names = monitor,
    )
end

function put_spread!(out, stem, method, μ, σ, lower, upper)
    n = nrow(out)
    out[!, Symbol("$(stem)_$(method)")] = as_column(μ, n)
    out[!, Symbol("$(stem)_$(method)_std")] = as_column(σ, n)
    out[!, Symbol("$(stem)_$(method)_lower")] = as_column(lower, n)
    out[!, Symbol("$(stem)_$(method)_upper")] = as_column(upper, n)
    return out
end

function add_targets!(out, u, model, targets, method)
    for t in targets
        put_spread!(out, "$(model)_$(t)", method, u.mean[t], u.std[t], u.lower[t], u.upper[t])
    end
    return out
end

function add_latents!(out, u, method)
    for name in SINN_LATENTS
        lat = u.latents[name]
        samples = lat.samples
        lo = [quantile(view(samples, i, :), u.quantiles[1]) for i in 1:size(samples, 1)]
        hi = [quantile(view(samples, i, :), u.quantiles[2]) for i in 1:size(samples, 1)]
        put_spread!(out, "pred_$(name)", method, lat.mean, lat.std, lo, hi)
    end
    return out
end

function estimate_fold(model, train_df, test_df, study, cfg; monitor, n_ens)
    kw = train_kwargs(study, cfg; monitor)
    res = train(model, train_df; merge(kw, (; random_seed = 42))...)
    res === nothing && error("MC dropout training produced no model")
    eval_data = prepare_data(model, test_df; drop_missing_rows = false)
    n_mc = study.smoke ? 4 : 100
    mc = estimate_uncertainty(MCDropout(n_samples = n_mc), model, train_df, res; eval_data)
    ens = estimate_uncertainty(
        DeepEnsemble(n_models = n_ens), model, train_df;
        parallel = true, eval_data, kw...,
    )
    return mc, ens
end

function fold_frames(build, df, study, params, targets, model_name; monitor, latents, target)
    folds = study_folds(df, study.k)
    n_ens = n_ensemble(study)
    pieces = DataFrame[]
    for fold in 1:study.k
        rows = filter(r -> r.test_fold == fold, params)
        if target !== nothing
            rows = filter(r -> Symbol(replace(String(r.target), ":" => "")) == target, rows)
        end
        nrow(rows) == 1 || error("$model_name fold $fold target=$target has $(nrow(rows)) hyperparameter rows")
        cfg = parse_cfg(rows[1, :])
        train_df = observed_rows(df[folds .!= fold, :], targets)
        test_df = df[folds .== fold, :]
        @info "$model_name fold $fold / $(study.k)" target n_ensemble = n_ens
        mc, ens = estimate_fold(build(cfg), train_df, test_df, study, cfg; monitor, n_ens)
        out = select(test_df, :row_id)
        add_targets!(out, mc, model_name, targets, "MC")
        add_targets!(out, ens, model_name, targets, "Ens")
        latents && (add_latents!(out, mc, "MC"); add_latents!(out, ens, "Ens"))
        push!(pieces, out)
    end
    return vcat(pieces...; cols = :union)
end

function attach(base, piece)
    fresh = setdiff(propertynames(piece), (:row_id,))
    return leftjoin(base, select(piece, :row_id, fresh...), on = :row_id)
end

function load_params(root, testid)
    path = joinpath(root, "eval", "$(testid)_hyperparams_$(RESULT_VERSION).csv")
    isfile(path) || error("missing tuned hyperparameters: $path")
    return CSV.read(path, DataFrame)
end

function main()
    study = study_grid()
    out = select(study.df, [:row_id, :SOCconc, :CF, :BD, :SOCdensity])
    uni = load_params(study.root, "mend_uniNN_dropout")
    for tgt in TARGETS
        piece = fold_frames(
            cfg -> build_uninn(study.predictors, tgt, cfg),
            dropmissing(study.df, tgt), study, uni, [tgt], "UniNN";
            monitor = Symbol[], latents = false, target = tgt,
        )
        out = attach(out, piece)
    end
    multi = load_params(study.root, "02_multiNN_dropout")
    out = attach(out, fold_frames(
        cfg -> build_multinn(study.predictors, MULTI_TARGETS, cfg),
        study.df, study, multi, MULTI_TARGETS, "MultiNN";
        monitor = Symbol[], latents = false, target = nothing,
    ))
    sinn = load_params(study.root, "mend_hybridNN_dropout")
    out = attach(out, fold_frames(
        cfg -> build_sinn(study.predictors, SINN_TARGETS, cfg),
        study.df, study, sinn, SINN_TARGETS, "SiNN";
        monitor = SINN_LATENTS, latents = true, target = nothing,
    ))
    path = joinpath(study.root, "eval", "uq_cv.pred_$(RESULT_VERSION).csv")
    CSV.write(path, out)
    @info "wrote uncertainty predictions" path rows = nrow(out) cols = ncol(out)
    return out
end

main()
