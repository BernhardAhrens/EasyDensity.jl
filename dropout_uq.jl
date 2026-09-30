using EasyHybrid
using Lux, NNlib, Optimisers, LuxCore
using CSV, DataFrames, Random, Statistics, JLD2, LinearAlgebra

const DATA_VERSION = "v20251125"
const RESULT_VERSION = "v20251209"
const ROW_KEYS = [:row_id, :time, :lat, :lon, :id, :nuts0, :maxdiff,
    :bd, :clay, :sand, :silt, :cf, :ocd, :soc, :SOCconc, :CF, :BD, :SOCdensity]

const SINN_SCALERS = Dict(:SOCconc => 0.151, :CF => 0.263, :BD => 0.529, :SOCdensity => 0.167)

function SOCD_model(; SOCconc, CF, oBD, mBD)
    ϵ = 1.0e-7
    soct = (exp.(SOCconc ./ SINN_SCALERS[:SOCconc]) .- 1) ./ 1000
    soct = clamp.(soct, ϵ, Inf)
    cft = (exp.(CF ./ SINN_SCALERS[:CF]) .- 1) ./ 100
    cft = clamp.(cft, 0, 0.99)
    som = 1.724f0 .* soct
    som = clamp.(som, 0, 1)
    denom = som .* mBD .+ (1.0f0 .- som) .* oBD
    BD = (oBD .* mBD) ./ denom
    BD = clamp.(BD, ϵ, Inf)
    SOCdensity = soct .* 1000 .* BD .* (1 .- cft)
    SOCdensity = clamp.(SOCdensity, 1, Inf)
    SOCdensity = log.(SOCdensity) .* SINN_SCALERS[:SOCdensity]
    BD = BD .* SINN_SCALERS[:BD]
    return (; BD, SOCconc, CF, SOCdensity, oBD, mBD)
end

const SINN_PARAMETERS = (
    SOCconc = (0.01f0, 0.0f0, 1.0f0),
    CF = (0.15f0, 0.0f0, 1.0f0),
    oBD = (0.20f0, 0.05f0, 0.40f0),
    mBD = (1.20f0, 0.75f0, 2.0f0),
)
const SINN_NEURAL_PARAMS = [:SOCconc, :CF, :mBD, :oBD]
const SINN_LATENTS = [:oBD, :mBD]
const TARGETS = [:BD, :SOCconc, :CF, :SOCdensity]

function build_uninn(predictors, target, cfg)
    return constructNNModel(predictors, [target];
        hidden_layers = make_dropout_chain(cfg.h, cfg.act, cfg.p),
        activation = cfg.act, scale_nn_outputs = true, input_batchnorm = false)
end

function build_multinn(predictors, targets, cfg)
    return constructNNModel(predictors, targets;
        hidden_layers = make_dropout_chain(cfg.h, cfg.act, cfg.p),
        activation = cfg.act, scale_nn_outputs = true, input_batchnorm = false)
end

function build_sinn(predictors, targets, cfg)
    return constructHybridModel(predictors, Symbol[], targets, SOCD_model, SINN_PARAMETERS,
        SINN_NEURAL_PARAMS, Symbol[];
        hidden_layers = make_dropout_chain(cfg.h, cfg.act, cfg.p),
        activation = cfg.act, scale_nn_outputs = true, input_batchnorm = false,
        start_from_default = true)
end

function make_dropout_chain(h, act, p)
    h = collect(Int, h)
    layers = []
    if length(h) == 1
        push!(layers, Dense(h[1], h[1], act))
        p > 0 && push!(layers, Dropout(Float32(p)))
    else
        for i in 1:(length(h) - 1)
            push!(layers, Dense(h[i], h[i + 1], act))
            p > 0 && push!(layers, Dropout(Float32(p)))
        end
    end
    return Chain(layers...)
end

function study_grid()
    root = @__DIR__
    smoke = get(ENV, "EASYDENSITY_SMOKE", "0") == "1"
    src = joinpath(root, "data", "lucas_preprocessed_$(DATA_VERSION).csv")
    df = CSV.read(src, DataFrame; normalizenames = true)
    predictors = Symbol.(names(df))[18:(end - 6)]
    if smoke
        n = parse(Int, get(ENV, "EASYDENSITY_SMOKE_N", "200"))
        visits = combine(groupby(df, :id), nrow => :nvisits)
        triple = Set(visits.id[visits.nvisits .== 3])
        finite_target = trues(nrow(df))
        for t in [:BD, :SOCconc, :CF, :SOCdensity]
            finite_target .&= map(v -> v isa Real && isfinite(v), df[!, t])
        end
        complete = Set(df.id[finite_target])
        df = df[in.(df.id, Ref(intersect(triple, complete))), :]
        min_rows = 160
        nrow(df) < min_rows && error("smoke data has $(nrow(df)) rows from three-visit sites; need at least $min_rows")
        id_order = unique(df.id)
        n_ids = cld(max(n, min_rows), 3)
        function column_filled(frame)
            for name in [ROW_KEYS; predictors]
                hasproperty(frame, name) || continue
                col = frame[!, name]
                nval = eltype(col) <: Union{Missing, Real} ? count(v -> v isa Real && isfinite(v), col) : count(!ismissing, col)
                nval > 0 || return name
            end
            return nothing
        end
        chosen = df[in.(df.id, Ref(Set(id_order[1:min(n_ids, length(id_order))]))), :]
        while (missing_col = column_filled(chosen)) !== nothing && n_ids < length(id_order)
            n_ids = min(length(id_order), n_ids + 20)
            chosen = df[in.(df.id, Ref(Set(id_order[1:n_ids]))), :]
        end
        missing_col = column_filled(chosen)
        missing_col === nothing || error("smoke column $missing_col has no values in $(nrow(chosen)) three-visit rows")
        df = chosen
        @info "smoke subset" rows = nrow(df) sites = length(unique(df.id)) visits = 3
        hidden = [(32, 16)]
        batches, rates, acts, drops = [32], [1e-2], [relu], [0.2]
        k, nepochs, patience = 2, 4, 2
    else
        hidden = [
            (512, 256, 128, 64, 32, 16), (512, 256, 128, 64, 32),
            (256, 128, 64, 32, 16), (256, 128, 64, 32), (256, 128, 64),
            (128, 64, 32, 16), (128, 64, 32), (64, 32, 16),
        ]
        batches = [128, 256, 512]
        rates = [1e-3, 5e-4, 1e-4]
        acts = [relu, swish, gelu]
        drops = [0.1, 0.2, 0.3]
        k, nepochs, patience = 5, 200, 15
    end
    configs = [(h = h, bs = bs, lr = lr, act = act, p = p)
               for h in hidden for bs in batches for lr in rates for act in acts for p in drops]
    return (; df, predictors, configs, k, nepochs, patience, smoke, root)
end

function as_column(point, n)
    values = Float32.(vec(collect(point)))
    length(values) == n && return values
    length(values) == 1 && return fill(values[1], n)
    error("prediction length $(length(values)) does not match $n rows")
end

function prediction_spread(model, x, ps, st, names, n_mc)
    st = LuxCore.trainmode(st)
    draws = Dict(name => Vector{Float32}[] for name in names)
    for _ in 1:n_mc
        yhat, st = model(x, ps, st)
        for name in names
            push!(draws[name], Float32.(vec(collect(getproperty(yhat, name)))))
        end
    end
    out = Dict{Symbol, NamedTuple}()
    for name in names
        M = reduce(hcat, draws[name])
        out[name] = (;
            std = Float32.(vec(std(M; dims = 2))),
            lower = Float32[quantile(view(M, i, :), 0.025) for i in 1:size(M, 1)],
            upper = Float32[quantile(view(M, i, :), 0.975) for i in 1:size(M, 1)],
        )
    end
    return out
end

function observed_rows(frame, targets)
    keep = trues(nrow(frame))
    for t in targets
        hasproperty(frame, t) || continue
        keep .&= map(v -> v isa Real && isfinite(v), frame[!, t])
    end
    return frame[keep, :]
end

function run_study(build_model, df, study; prefix, targets, latents = Symbol[], testid,
        monitor = Symbol[], into = nothing, write = true, target = nothing, append_params = false)
    t0 = time()
    Random.seed!(42)
    folds = make_folds(df, k = study.k, shuffle = true)
    ncfg = length(study.configs)
    fold_best = Vector{NamedTuple}(undef, study.k)
    n_mc = study.smoke ? 4 : 100
    blas = BLAS.get_num_threads()
    BLAS.set_num_threads(1)
    try
        for fold in 1:study.k
            @info "$testid fold $fold / $(study.k)" configs = ncfg threads = Threads.nthreads()
            train_df = observed_rows(df[folds .!= fold, :], targets)
            nrow(train_df) == 0 && error("$testid fold $fold has no rows with finite targets")
            slot = Vector{Any}(nothing, ncfg)
            Threads.@threads for i in 1:ncfg
                cfg = study.configs[i]
                model = build_model(cfg)
                res = train(model, train_df;
                    nepochs = study.nepochs, batchsize = cfg.bs, opt = AdamW(cfg.lr),
                    training_loss = :mse, loss_types = [:mse, :r2], shuffleobs = true,
                    random_seed = 42, patience = study.patience, agg = mean,
                    return_model = :best, show_progress = false, plotting = false,
                    save_training = false, monitor_names = monitor)
                res === nothing && continue
                slot[i] = (; loss = res.best_loss, cfg, res, model = deepcopy(model))
            end
            done = filter(!isnothing, slot)
            isempty(done) && error("$testid fold $fold produced no trained model")
            fold_best[fold] = done[argmin([d.loss for d in done])]
        end
    finally
        BLAS.set_num_threads(blas)
    end

    pieces = DataFrame[]
    params = DataFrame(target = Symbol[], h = String[], bs = Int[], lr = Float64[], act = String[],
        p = Float64[], mse = Float64[], best_epoch = Int[], test_fold = Int[])
    for fold in eachindex(fold_best)
        fb = fold_best[fold]
        test_df = df[folds .== fold, :]
        n = nrow(test_df)
        x = prepare_data(fb.model, test_df; drop_missing_rows = false)[1]
        yhat, _ = fb.model(x, fb.res.ps, LuxCore.testmode(fb.res.st))
        spread = prediction_spread(fb.model, x, fb.res.ps, fb.res.st, [targets; latents], n_mc)
        keys = filter(k -> hasproperty(test_df, k), ROW_KEYS)
        out = select(test_df, keys)
        for t in targets
            out[!, Symbol(prefix, t)] = as_column(getproperty(yhat, t), n)
            out[!, Symbol(prefix, t, "_std")] = spread[t].std
            out[!, Symbol(prefix, t, "_lower")] = spread[t].lower
            out[!, Symbol(prefix, t, "_upper")] = spread[t].upper
        end
        for t in latents
            out[!, Symbol("pred_", t)] = as_column(getproperty(yhat, t), n)
            out[!, Symbol("pred_", t, "_std")] = spread[t].std
            out[!, Symbol("pred_", t, "_lower")] = spread[t].lower
            out[!, Symbol("pred_", t, "_upper")] = spread[t].upper
        end
        push!(pieces, out)
        push!(params, (; target = something(target, :all), h = string(fb.cfg.h), bs = fb.cfg.bs,
            lr = fb.cfg.lr, act = string(fb.cfg.act), p = fb.cfg.p, mse = fb.loss,
            best_epoch = fb.res.best_epoch, test_fold = fold))
    end
    pred = vcat(pieces...; cols = :union)
    if into !== nothing
        fresh = setdiff(Symbol.(names(pred)), ROW_KEYS)
        pred = leftjoin(into, select(pred, :row_id, fresh), on = :row_id)
    end
    if !hasproperty(pred, :LC1)
        lc_path = joinpath(study.root, "data", "lc_by_row_id.csv")
        if isfile(lc_path)
            pred = leftjoin(pred, CSV.read(lc_path, DataFrame), on = :row_id)
        end
    end

    eval_dir = joinpath(study.root, "eval")
    models_dir = joinpath(study.root, "models")
    mkpath(eval_dir)
    mkpath(models_dir)
    param_path = joinpath(eval_dir, "$(testid)_hyperparams_$(RESULT_VERSION).csv")
    CSV.write(param_path, params; append = append_params, writeheader = !append_params)
    tag = target === nothing ? "" : "_$(target)"
    best = fold_best[argmin([fb.loss for fb in fold_best])]
    retrained = build_model(best.cfg)
    res = train(retrained, observed_rows(df, targets);
        nepochs = study.nepochs, batchsize = best.cfg.bs, opt = AdamW(best.cfg.lr),
        training_loss = :mse, loss_types = [:mse, :r2], shuffleobs = true,
        random_seed = 42, patience = study.patience, agg = mean,
        return_model = :best, show_progress = false, plotting = false,
        save_training = false, monitor_names = monitor)
    res === nothing && error("$testid retrain produced no model")
    jldsave(joinpath(models_dir, "$(testid)$(tag)_best_$(RESULT_VERSION).jld2");
        cfg = best.cfg, ps = res.ps, st = res.st)
    if write
        CSV.write(joinpath(eval_dir, "$(testid)_cv.pred_$(RESULT_VERSION).csv"), pred)
    end
    @info "$testid finished" seconds = round(time() - t0; digits = 1) trainings = study.k * ncfg + 1
    return pred
end
