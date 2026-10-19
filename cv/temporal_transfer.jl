# Train once on every complete 2018 row and predict 2009, 2012, and 2015.
# Spatial CV stored one configuration per fold; this run keeps the lowest validation MSE.
# No spatial fold is held out. EasyHybrid early-stops on its own validation split.
# The written median is that single physical prediction. No 2018 row is a test row.

include(joinpath(@__DIR__, "common.jl"))

const TRAIN_YEAR = 2018
const OBS = Dict(:SOCconc => :soc, :CF => :cf, :BD => :bd, :SOCdensity => :ocd)

function tc_models()
    name = get(ENV, "EASYDENSITY_TC_MODEL", "all")
    name == "all" && return ("UniNN", "MultiNN", "SiNN")
    name in ("UniNN", "MultiNN", "SiNN") ||
        error("EASYDENSITY_TC_MODEL must be all, UniNN, MultiNN, or SiNN, got $name")
    return (name,)
end

function chosen_config(params; target = nothing)
    rows = target === nothing ? params : filter(r -> String(r.target) == String(target), params)
    nrow(rows) > 0 || error("no hyperparameter rows for target=$target")
    row = rows[argmin(Float64.(rows.val_mse)), :]
    return (;
        h = parse_hidden(row.h), bs = Int(row.bs), lr = Float64(row.lr),
        act = ACTS[String(row.act)],
        val_mse = Float64(row.val_mse), test_fold = Int(row.test_fold),
    )
end

function year_split(df, predictors, targets, year)
    pool = df[df.time .== year, :]
    keep = finite_rows(pool, (targets..., predictors...))
    fit_df = pool[keep, :]
    nrow(fit_df) > 1 || error("training year $year has $(nrow(fit_df)) complete rows")
    test_df = df[df.time .!= year, :]
    test_df = test_df[finite_rows(test_df, predictors), :]
    nrow(test_df) > 0 || error("no rows outside $year")
    return (; fit_df, test_df)
end

function tc_ensemble_path(root, architecture)
    return joinpath(results_dir(root), "tc_ensemble_$(architecture).csv")
end

function observed_physical(frame, target)
    y = frame[!, OBS[target]]
    target == :CF && return y .* 100
    return y
end

function regression_scores(pred, obs)
    err = pred .- obs
    denom = sum((obs .- mean(obs)) .^ 2)
    r2 = denom == 0 ? NaN : 1 - sum(err .^ 2) / denom
    return (; n = length(obs), r2, rmse = sqrt(mean(err .^ 2)), bias = mean(err))
end

function score_transfer(test_df, pred, model, names)
    pred.row_id == test_df.row_id || error("$model predictions are not aligned with the test rows")
    rows = NamedTuple[]
    years = sort(unique(skipmissing(test_df.time)))
    for name in names
        name in TARGETS || continue
        yhat = pred[!, "$(stem(model, name))_ens_median"]
        y = observed_physical(test_df, name)
        for year in Any[nothing; years]
            keep = year === nothing ? trues(nrow(test_df)) : (test_df.time .== year)
            ok = keep .& map(finite_value, yhat) .& map(finite_value, y)
            count(ok) < 2 && continue
            stats = regression_scores(Float64.(yhat[ok]), Float64.(y[ok]))
            push!(rows, (;
                model, target = String(name),
                year = year === nothing ? "all" : string(year),
                stats...,
            ))
        end
    end
    return rows
end

function config_row(model, target, cfg, roles)
    return (;
        model, target = String(target),
        h = string(cfg.h), bs = cfg.bs, lr = cfg.lr, act = act_name(cfg.act),
        cv_test_fold = cfg.test_fold, cv_val_mse = cfg.val_mse,
        n_fit = nrow(roles.fit_df), n_test = nrow(roles.test_df),
        seed = TUNING_SEED,
    )
end

function prediction_frame(row_ids, model, slot, names)
    out = DataFrame(row_id = collect(row_ids))
    for name in names
        out[!, "$(stem(model, name))_ens_median"] = to_physical(name, slot[name])
    end
    return out
end

function transfer_one(build, label, roles, names, study, cfg)
    @info "$label temporal transfer" year = TRAIN_YEAR n_fit = nrow(roles.fit_df) n_test = nrow(roles.test_df) seed = TUNING_SEED h = cfg.h bs = cfg.bs lr = cfg.lr act = act_name(cfg.act) cv_test_fold = cfg.test_fold cv_val_mse = cfg.val_mse
    model = build()
    trained = train(model, deepcopy(roles.fit_df);
        nepochs = study.nepochs, batchsize = cfg.bs, opt = AdamW(cfg.lr),
        training_loss = :mse, loss_types = [:mse, :r2],
        shuffleobs = true, random_seed = TUNING_SEED,
        patience = study.patience, agg = mean,
        return_model = :best, show_progress = false, plotting = false,
        save_training = false, keep_history = false)
    trained === nothing && error("$label produced no model")
    slot = predict_test(model, trained, roles.test_df, names)
    pred = prediction_frame(roles.test_df.row_id, label, slot, names)
    return pred, score_transfer(roles.test_df, pred, label, names)
end

function run_uninn(study)
    params = read_hyperparams(study.root, "UniNN")
    pieces = DataFrame[]
    configs = NamedTuple[]
    metrics = NamedTuple[]
    for target in TARGETS
        roles = year_split(study.df, study.predictors, [target], TRAIN_YEAR)
        cfg = chosen_config(params; target)
        pred, scores = transfer_one("UniNN", roles, [target], study, cfg) do
            uninn_model(study.predictors, target, cfg)
        end
        push!(pieces, pred)
        append!(metrics, scores)
        push!(configs, config_row("UniNN", target, cfg, roles))
    end
    write_table(tc_ensemble_path(study.root, "UniNN"), join_on_row(pieces))
    return configs, metrics
end

function run_joint(study, architecture, names, build)
    params = read_hyperparams(study.root, architecture)
    roles = year_split(study.df, study.predictors, TARGETS, TRAIN_YEAR)
    cfg = chosen_config(params)
    pred, scores = transfer_one(architecture, roles, names, study, cfg) do
        build(cfg)
    end
    write_table(tc_ensemble_path(study.root, architecture), pred)
    return [config_row(architecture, "all", cfg, roles)], scores
end

study = load_study()
wanted = tc_models()
configs = NamedTuple[]
metrics = NamedTuple[]

if "UniNN" in wanted
    c, m = run_uninn(study)
    append!(configs, c)
    append!(metrics, m)
end
if "MultiNN" in wanted
    c, m = run_joint(study, "MultiNN", TARGETS, cfg -> multinn_model(study.predictors, TARGETS, cfg))
    append!(configs, c)
    append!(metrics, m)
end
if "SiNN" in wanted
    c, m = run_joint(study, "SiNN", [TARGETS; SINN_LATENTS], cfg -> sinn_model(study.predictors, TARGETS, cfg))
    append!(configs, c)
    append!(metrics, m)
end

results = results_dir(study.root)
mkpath(results)
stem = length(wanted) == 3 ? "temporal_transfer" : "temporal_transfer_$(join(wanted, "_"))"
write_table(joinpath(results, "$(stem)_hyperparams.csv"), configs)
write_table(joinpath(results, "$(stem)_metrics.csv"), metrics)
show(DataFrame(metrics); allrows = true, allcols = true)
println()
