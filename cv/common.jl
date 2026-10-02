using EasyHybrid, Lux, NNlib, Optimisers, LuxCore
using CSV, DataFrames, Random, Statistics, LinearAlgebra

# Dropout-free cross-validation. Tuning and the ensemble are separate scripts.
# Outer test fold is untouched. The next fold is validation. The other folds train.

const DATA_VERSION = "v20251125"
const RESULT_VERSION = get(ENV, "EASYDENSITY_SMOKE", "0") == "1" ? "v20261002_smoke" : "v20261002"
const TUNING_SEED = 42
const TARGETS = [:BD, :SOCconc, :CF, :SOCdensity]
const SINN_LATENTS = [:oBD, :mBD]
const SCALERS = Dict(:SOCconc => 0.151, :CF => 0.263, :BD => 0.529, :SOCdensity => 0.167)
const ACTS = Dict("relu" => relu, "swish" => swish, "gelu" => gelu)
const HIDDEN = [
    (512, 256, 128, 64, 32, 16), (512, 256, 128, 64, 32),
    (256, 128, 64, 32, 16), (256, 128, 64, 32), (256, 128, 64),
    (128, 64, 32, 16), (128, 64, 32), (64, 32, 16),
]

const SINN_PARAMETERS = (
    SOCconc = (0.01f0, 0.0f0, 1.0f0),
    CF = (0.15f0, 0.0f0, 1.0f0),
    oBD = (0.20f0, 0.05f0, 0.40f0),
    mBD = (1.20f0, 0.75f0, 2.0f0),
)
const SINN_NEURAL = [:SOCconc, :CF, :mBD, :oBD]

function SOCD_model(; SOCconc, CF, oBD, mBD)
    ϵ = 1.0e-7
    soct = (exp.(SOCconc ./ SCALERS[:SOCconc]) .- 1) ./ 1000
    soct = clamp.(soct, ϵ, Inf)
    cft = (exp.(CF ./ SCALERS[:CF]) .- 1) ./ 100
    cft = clamp.(cft, 0, 0.99)
    som = clamp.(1.724f0 .* soct, 0, 1)
    BD = (oBD .* mBD) ./ (som .* mBD .+ (1.0f0 .- som) .* oBD)
    BD = clamp.(BD, ϵ, Inf)
    SOCdensity = log.(clamp.(soct .* 1000 .* BD .* (1 .- cft), 1, Inf)) .* SCALERS[:SOCdensity]
    BD = BD .* SCALERS[:BD]
    return (; BD, SOCconc, CF, SOCdensity, oBD, mBD)
end

finite_value(v) = v isa Real && isfinite(Float64(v))

function act_name(f)
    for (name, act) in ACTS
        f === act && return name
    end
    error("unknown activation $f")
end

function parse_hidden(text)
    ex = Meta.parse(String(text))
    nums = ex isa Expr ? ex.args : [ex]
    return Tuple(Int.(nums))
end

function to_physical(name, y)
    name == :SOCconc && return @. exp(y / SCALERS[:SOCconc]) - 1
    name == :CF && return @. exp(y / SCALERS[:CF]) - 1
    name == :BD && return y ./ SCALERS[:BD]
    name == :SOCdensity && return @. exp(y / SCALERS[:SOCdensity])
    return Float64.(y)
end

function finite_rows(frame, cols)
    keep = trues(nrow(frame))
    for col in cols
        hasproperty(frame, col) || continue
        keep .&= map(finite_value, frame[!, col])
    end
    return keep
end

function take_smoke(df, predictors, n)
    visits = combine(groupby(df, :id), nrow => :nvisits)
    triple = Set(visits.id[visits.nvisits .== 3])
    ok = trues(nrow(df))
    for t in TARGETS
        ok .&= map(finite_value, df[!, t])
    end
    df = df[in.(df.id, Ref(intersect(triple, Set(df.id[ok])))), :]
    nrow(df) >= 160 || error("smoke subset has $(nrow(df)) complete three-visit rows; need 160")
    order = unique(df.id)
    n_ids = min(length(order), cld(max(n, 160), 3))
    function blank(frame)
        for col in predictors
            any(finite_value, frame[!, col]) || return col
        end
        return nothing
    end
    chosen = df[in.(df.id, Ref(Set(order[1:n_ids]))), :]
    while (missing_col = blank(chosen)) !== nothing && n_ids < length(order)
        n_ids = min(length(order), n_ids + 20)
        chosen = df[in.(df.id, Ref(Set(order[1:n_ids]))), :]
    end
    missing_col = blank(chosen)
    missing_col === nothing || error("smoke column $missing_col has no values")
    @info "smoke subset" rows = nrow(chosen) sites = length(unique(chosen.id))
    return chosen
end

function load_study()
    root = dirname(@__DIR__)
    smoke = get(ENV, "EASYDENSITY_SMOKE", "0") == "1"
    src = joinpath(root, "data", "lucas_preprocessed_$(DATA_VERSION).csv")
    df = CSV.read(src, DataFrame; normalizenames = true)
    hasproperty(df, :row_id) || (df.row_id = 1:nrow(df))
    predictors = Symbol.(names(df))[18:(end - 6)]
    if smoke
        n = parse(Int, get(ENV, "EASYDENSITY_SMOKE_N", "200"))
        df = take_smoke(df, predictors, n)
        configs = [(h = (32, 16), bs = 32, lr = 1e-2, act = relu)]
        # One training fold, one validation fold, one test fold.
        k, nepochs, patience, seeds = 3, 4, 2, 1:2
    else
        configs = [(h = h, bs = bs, lr = lr, act = act)
                   for h in HIDDEN for bs in (128, 256, 512)
                   for lr in (1e-3, 5e-4, 1e-4) for act in (relu, swish, gelu)]
        k, nepochs, patience, seeds = 5, 200, 15, 1:20
    end
    return (; df, predictors, configs, k, nepochs, patience, seeds, smoke, root)
end

function assign_folds(df, k)
    Random.seed!(42)
    return make_folds(df, k = k, shuffle = true)
end

function fold_roles(df, folds, test_fold, k, targets, predictors)
    val_fold = mod1(test_fold + 1, k)
    train_folds = setdiff(collect(1:k), [test_fold, val_fold])
    inside = folds .!= test_fold
    fit_df = df[inside, :]
    fit_folds = folds[inside]
    keep = finite_rows(fit_df, (targets..., predictors...))
    fit_df = fit_df[keep, :]
    fit_folds = fit_folds[keep]
    any(==(val_fold), fit_folds) || error("validation fold $val_fold has no complete rows")
    any(x -> x in train_folds, fit_folds) || error("training folds have no complete rows")
    test_df = df[folds .== test_fold, :]
    test_df = test_df[finite_rows(test_df, predictors), :]
    return (; val_fold, train_folds, fit_df, fit_folds, test_df)
end

# Model construction and EasyHybrid `train` live here so the six scripts stay in step.

function uninn_model(predictors, target, cfg)
    return constructNNModel(predictors, [target];
        hidden_layers = collect(Int, cfg.h), activation = cfg.act,
        scale_nn_outputs = true, input_batchnorm = false)
end

function multinn_model(predictors, targets, cfg)
    return constructNNModel(predictors, targets;
        hidden_layers = collect(Int, cfg.h), activation = cfg.act,
        scale_nn_outputs = true, input_batchnorm = false)
end

function sinn_model(predictors, targets, cfg)
    return constructHybridModel(predictors, Symbol[], targets, SOCD_model, SINN_PARAMETERS,
        SINN_NEURAL, Symbol[];
        hidden_layers = collect(Int, cfg.h), activation = cfg.act,
        scale_nn_outputs = true, input_batchnorm = false, start_from_default = true)
end

function fit(model, roles, study, cfg; seed)
    return train(model, deepcopy(roles.fit_df);
        nepochs = study.nepochs, batchsize = cfg.bs, opt = AdamW(cfg.lr),
        training_loss = :mse, loss_types = [:mse, :r2],
        folds = roles.fit_folds, val_fold = roles.val_fold,
        random_seed = seed, patience = study.patience, agg = mean,
        return_model = :best, show_progress = false, plotting = false,
        save_training = false, keep_history = false)
end

function predict_test(model, trained, frame, names)
    (x_test, _) = prepare_data(model, frame; drop_missing_rows = false)
    yhat, _ = model(x_test, trained.ps, LuxCore.testmode(trained.st))
    n = nrow(frame)
    return Dict(name => column_values(yhat, name, n) for name in names)
end

function hyperparams_path(root, architecture)
    return joinpath(root, "eval", "cv_$(architecture)_hyperparams_$(RESULT_VERSION).csv")
end

function remember(target, test_fold, val_fold, best)
    return (;
        target = String(target), test_fold, val_fold,
        h = string(best.h), bs = best.bs, lr = best.lr, act = act_name(best.act),
        val_mse = best.val_mse, best_epoch = best.best_epoch,
    )
end

function write_table(path, rows)
    mkpath(dirname(path))
    CSV.write(path, rows isa DataFrame ? rows : DataFrame(rows))
    @info "wrote" path
    return path
end

function read_hyperparams(root, architecture)
    path = hyperparams_path(root, architecture)
    isfile(path) || error("missing $path")
    return CSV.read(path, DataFrame)
end

function saved_config(params, test_fold; target = nothing)
    rows = filter(r -> r.test_fold == test_fold, params)
    target === nothing || (rows = filter(r -> String(r.target) == String(target), rows))
    nrow(rows) == 1 || error("test fold $test_fold target=$target has $(nrow(rows)) hyperparameter rows")
    row = rows[1, :]
    return (;
        h = parse_hidden(row.h), bs = Int(row.bs), lr = Float64(row.lr),
        act = ACTS[String(row.act)], val_fold = Int(row.val_fold),
    )
end

function column_values(yhat, name, n)
    values = Float64.(vec(collect(getproperty(yhat, name))))
    length(values) == n && return values
    length(values) == 1 && return fill(values[1], n)
    error("$name has length $(length(values)), expected $n")
end

function column_stats(M)
    med = [median(@view(M[i, :])) for i in axes(M, 1)]
    μ = vec(mean(M; dims = 2))
    σ = vec(std(M; dims = 2))
    iqr = [quantile(@view(M[i, :]), 0.75) - quantile(@view(M[i, :]), 0.25) for i in axes(M, 1)]
    return med, μ, σ, iqr
end

function stem(model, name)
    return name in SINN_LATENTS ? "pred_$(name)" : "$(model)_$(name)"
end

function spread_frame(row_ids, model, slots, names)
    out = DataFrame(row_id = collect(row_ids))
    for name in names
        M = reduce(hcat, (to_physical(name, slot[name]) for slot in slots))
        med, μ, σ, iqr = column_stats(M)
        col = stem(model, name)
        out[!, "$(col)_ens_median"] = med
        out[!, "$(col)_ens_mean"] = μ
        out[!, "$(col)_ens_sd"] = σ
        out[!, "$(col)_ens_iqr"] = iqr
    end
    return out
end

function member_frame(row_ids, model, slots, names, seeds)
    ids = collect(row_ids)
    pieces = DataFrame[]
    for (slot, seed) in zip(slots, seeds), name in names
        push!(pieces, DataFrame(
            row_id = ids, model = model, target = String(name),
            seed = seed, prediction = to_physical(name, slot[name]),
        ))
    end
    return vcat(pieces...)
end

function join_on_row(pieces)
    out = pieces[1]
    for piece in pieces[2:end]
        out = outerjoin(out, piece, on = :row_id)
    end
    return out
end

function ensemble_path(root, architecture)
    return joinpath(root, "eval", "cv_ensemble_$(architecture)_$(RESULT_VERSION).csv")
end

function members_path(root, architecture)
    return joinpath(root, "eval", "cv_ensemble_members_$(architecture)_$(RESULT_VERSION).csv")
end
