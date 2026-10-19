include(joinpath(@__DIR__, "../cv/paths.jl"))

const ENS_VERSION = result_version()
const MODELS = ("UniNN", "MultiNN", "SiNN")

ens_col(model, target, stat = "median") = "$(model)_$(target)_ens_$(stat)"
latent_col(name, stat = "median") = "pred_$(name)_ens_$(stat)"

function figure_results(root = normpath(joinpath(@__DIR__, "..")))
    dir = results_dir(root)
    mkpath(dir)
    return dir
end

function attach_ensemble(df, root)
    for model in MODELS
        path = ensemble_path(root, model)
        isfile(path) || error("missing ensemble table $path")
        piece = CSV.read(path, DataFrame)
        fresh = setdiff(propertynames(piece), (:row_id, propertynames(df)...))
        df = leftjoin(df, select(piece, :row_id, fresh...), on = :row_id)
    end
    return df
end

# Point predictions in the cross-validation table are on the training scale.
const POINT_SCALERS = (SOCconc = 0.151, CF = 0.263, BD = 0.529, SOCdensity = 0.167)

point_col(model, target) = "$(model)_$(target)_phys"

function physical_value(target, v)
    v isa Real && isfinite(Float64(v)) || return missing
    x = Float64(v)
    target == "SOCconc" && return exp(x / POINT_SCALERS.SOCconc) - 1
    target == "CF" && return exp(x / POINT_SCALERS.CF) - 1
    target == "BD" && return x / POINT_SCALERS.BD
    target == "SOCdensity" && return exp(x / POINT_SCALERS.SOCdensity)
    error("unknown target $target")
end

function add_point_predictions!(df)
    for model in MODELS, target in ("BD", "SOCconc", "CF", "SOCdensity")
        src = "$(model)_$(target)"
        hasproperty(df, src) || error("missing prediction $src")
        y = df[!, src]
        df[!, point_col(model, target)] = Union{Missing,Float64}[physical_value(target, v) for v in y]
    end
    return df
end

function prediction_sources()
    src = get(ENV, "EASYDENSITY_PRED", "")
    src == "" && return ("median", "tune")
    src in ("median", "tune") || error("EASYDENSITY_PRED must be median or tune, got $src")
    return (src,)
end

function scale_modes()
    src = get(ENV, "EASYDENSITY_SCALE", "")
    src == "" && return ("scaled", "unscaled")
    src in ("scaled", "unscaled") || error("EASYDENSITY_SCALE must be scaled or unscaled, got $src")
    return (src,)
end

function scaled_value(target, v)
    v isa Real && isfinite(Float64(v)) || return missing
    x = Float64(v)
    s = getproperty(POINT_SCALERS, Symbol(target))
    target == "BD" && return x * s
    target == "SOCdensity" && return x > 0 ? s * log(x) : missing
    return x > -1 ? s * log(x + 1) : missing
end

function add_scaled_medians!(df)
    for model in MODELS, target in ("BD", "SOCconc", "CF", "SOCdensity")
        src = ens_col(model, target)
        hasproperty(df, src) || error("missing ensemble column $src")
        y = df[!, src]
        df[!, "$(src)_scaled"] = Union{Missing,Float64}[scaled_value(target, v) for v in y]
    end
    return df
end

function load_figure_frame(root)
    path = observation_path(root)
    isfile(path) || error("missing $path")
    df = CSV.read(path, DataFrame)
    sources = prediction_sources()
    "tune" in sources && add_point_predictions!(df)
    "median" in sources && (df = attach_ensemble(df, root))
    return df
end

function value_col(model, target, source)
    source == "median" && return ens_col(model, target)
    source == "tune" && return point_col(model, target)
    error("unknown prediction source $source")
end

function latent_value_col(name, source)
    source == "median" && return latent_col(name)
    source == "tune" && return "pred_$name"
    error("unknown prediction source $source")
end

function joint_pairs(source)
    pairs = [["bd", "soc"]]
    for model in MODELS
        push!(pairs, [value_col(model, "BD", source), value_col(model, "SOCconc", source)])
    end
    return pairs
end

ocd_cols(source) = [value_col(model, "SOCdensity", source) for model in MODELS]
