# Ensemble median versus observations, in physical units.
# Ensemble SD is model disagreement, not a prediction interval.

using CSV, DataFrames, Statistics

const VERSION = "v20261002"
const MODELS = ("UniNN", "MultiNN", "SiNN")
const TARGETS = ("SOCconc", "CF", "BD", "SOCdensity")
const OBS = Dict("SOCconc" => :soc, "CF" => :cf, "BD" => :bd, "SOCdensity" => :ocd)

ens_col(model, target, stat) = "$(model)_$(target)_ens_$(stat)"
finite_value(v) = v isa Real && isfinite(Float64(v))
rmse(pred, obs) = sqrt(mean((pred .- obs) .^ 2))

function load_joined(root)
    obs = CSV.read(joinpath(root, "eval", "all_cv.pred_with.lc_v20251209.csv"), DataFrame)
    for model in MODELS
        path = joinpath(root, "eval", "cv_ensemble_$(model)_$(VERSION).csv")
        isfile(path) || error("missing $path")
        piece = CSV.read(path, DataFrame)
        fresh = setdiff(propertynames(piece), (:row_id, propertynames(obs)...))
        obs = leftjoin(obs, select(piece, :row_id, fresh...), on = :row_id)
    end
    return obs
end

function paired(df, model, target)
    pred = df[!, ens_col(model, target, "median")]
    obs = df[!, OBS[target]]
    sd = df[!, ens_col(model, target, "sd")]
    iqr = df[!, ens_col(model, target, "iqr")]
    mask = map(finite_value, pred) .& map(finite_value, obs) .& map(finite_value, sd) .& map(finite_value, iqr)
    return Float64.(pred[mask]), Float64.(obs[mask]), Float64.(sd[mask]), Float64.(iqr[mask])
end

function summary_table(df)
    rows = NamedTuple[]
    for target in TARGETS, model in MODELS
        pred, obs, sd, iqr = paired(df, model, target)
        push!(rows, (;
            target, model, n = length(obs),
            rmse = rmse(pred, obs),
            median_sd = median(sd), median_iqr = median(iqr),
        ))
    end
    return DataFrame(rows)
end

function landcover_column(df)
    for name in (:LC_group, :LC1)
        hasproperty(df, name) && return name
    end
    return nothing
end

function landcover_table(df, lc)
    rows = NamedTuple[]
    for class in unique(df[!, lc]), target in TARGETS, model in MODELS
        sub = df[isequal.(df[!, lc], class), :]
        pred, obs, sd, iqr = paired(sub, model, target)
        isempty(obs) && continue
        push!(rows, (;
            landcover = class, target, model, n = length(obs),
            rmse = rmse(pred, obs), median_sd = median(sd), median_iqr = median(iqr),
        ))
    end
    return DataFrame(rows)
end

function finite_numbers(x)
    return Float64[v for v in x if finite_value(v)]
end

function temporal_table(df)
    hasproperty(df, :id) || return DataFrame()
    counts = combine(groupby(df, :id), nrow => :n)
    sites = Set(counts.id[counts.n .== 3])
    df3 = df[in.(df.id, Ref(sites)), :]
    isempty(df3) && return DataFrame()
    rows = NamedTuple[]
    for model in MODELS
        med = ens_col(model, "SOCdensity", "median")
        sd = ens_col(model, "SOCdensity", "sd")
        by_site = combine(groupby(df3, :id),
            med => (x -> (y = finite_numbers(x); isempty(y) ? missing : maximum(y) - minimum(y))) => :range,
            sd => (x -> (y = finite_numbers(x); isempty(y) ? missing : mean(y))) => :mean_sd,
        )
        for row in eachrow(by_site)
            finite_value(row.range) && finite_value(row.mean_sd) || continue
            push!(rows, (; model, id = row.id, range = Float64(row.range), mean_sd = Float64(row.mean_sd)))
        end
    end
    return DataFrame(rows)
end

root = dirname(@__DIR__)
df = load_joined(root)
summary = summary_table(df)
eval_dir = joinpath(root, "eval")
CSV.write(joinpath(eval_dir, "cv_ensemble_summary_$(VERSION).csv"), summary)
@info "wrote" path = joinpath(eval_dir, "cv_ensemble_summary_$(VERSION).csv")
lc = landcover_column(df)
if lc === nothing
    @warn "no land-cover column; skipping land-cover table"
else
    CSV.write(joinpath(eval_dir, "cv_ensemble_landcover_$(VERSION).csv"), landcover_table(df, lc))
end
temporal = temporal_table(df)
CSV.write(joinpath(eval_dir, "cv_ensemble_temporal_$(VERSION).csv"), temporal)
show(summary; allrows = true, allcols = true)
println()
for model in MODELS
    sub = filter(r -> r.model == model, temporal)
    nrow(sub) < 3 && continue
    @info "temporal range vs mean ensemble SD" model r = cor(sub.range, sub.mean_sd) n = nrow(sub)
end
