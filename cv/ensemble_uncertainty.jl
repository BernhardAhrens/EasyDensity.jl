# Sharpness, PICP, PIW, QCP, and RMSE in bins of ensemble SD.
# Same summaries as prediction_uncertainty.jl, for the dropout-free deep ensembles.
# Predictions and observations are in physical units. The point prediction is the
# ensemble median. The 90% interval is the 0.05 and 0.95 quantiles of the members.
# Member count is the number of predictions stored for each sample.
# PIW is upper minus lower. The summary stores its mean (MPIW) and median.
# QCP is the fraction of observations at or below one member quantile.
# Decile tables use quantile levels and central PI levels 10, 20, ..., 90 percent.
# Coarse fraction is stored as a fraction; the ensemble is in percent.

using CSV, DataFrames, Statistics

include(joinpath(@__DIR__, "paths.jl"))

const MODELS = ("UniNN", "MultiNN", "SiNN")
const TARGETS = ("SOCconc", "CF", "BD", "SOCdensity")
const OBS = Dict("SOCconc" => :soc, "CF" => :cf, "BD" => :bd, "SOCdensity" => :ocd)
const N_BINS = 10
const NOMINAL = 0.90
const DECILE_PERCENTS = 10:10:90

function level_percents()
    percents = Int[]
    for percent in DECILE_PERCENTS
        push!(percents, percent, div(100 - percent, 2), div(100 + percent, 2))
    end
    return sort!(unique(percents))
end

const LEVEL_PERCENTS = level_percents()

level_name(percent::Integer) = Symbol("q", lpad(Int(percent), 2, '0'))

ens_col(model, target, stat) = "$(model)_$(target)_ens_$(stat)"
finite_value(v) = v isa Real && isfinite(Float64(v))
rmse(pred, obs) = sqrt(mean((pred .- obs) .^ 2))

function observed_column(df, target)
    y = df[!, OBS[target]]
    target == "CF" || return y
    return [finite_value(v) ? 100 * Float64(v) : missing for v in y]
end

function load_joined(root)
    obs_path = observation_path(root)
    isfile(obs_path) || error("missing $obs_path")
    obs = CSV.read(obs_path, DataFrame)
    for model in MODELS
        path = ensemble_path(root, model)
        isfile(path) || error("missing $path")
        piece = CSV.read(path, DataFrame)
        fresh = setdiff(propertynames(piece), (:row_id, propertynames(obs)...))
        obs = leftjoin(obs, select(piece, :row_id, fresh...), on = :row_id)
    end
    return obs
end

function member_quantiles(path)
    isfile(path) || error("missing $path")
    raw = CSV.read(path, DataFrame; select = [:row_id, :target, :seed, :prediction])
    filter!(r -> String(r.target) in TARGETS, raw)
    names = Tuple(level_name(percent) for percent in LEVEL_PERCENTS)
    probs = LEVEL_PERCENTS ./ 100
    tall = combine(groupby(raw, [:row_id, :target]),
        :prediction => (p -> NamedTuple{names}(Tuple(quantile(Float64.(p), probs)))) => AsTable,
        nrow => :n_members,
    )
    counts = unique(tall.n_members)
    length(counts) == 1 || error("$path has mixed member counts: $(sort(counts))")
    n_members = only(counts)
    n_seeds = length(unique(raw.seed))
    n_members == n_seeds || error("$path has $n_seeds seeds but $n_members predictions per sample")
    return select(tall, Not(:n_members)), n_members
end

# Rank bins keep about the same count in each bin when SD values tie.
function quantile_bin_ids(x, nbins)
    n = length(x)
    ids = Vector{Int}(undef, n)
    for (rank, i) in enumerate(sortperm(x))
        ids[i] = 1 + div((rank - 1) * nbins, n)
    end
    return ids
end

function model_rows(df, quantiles, model, target)
    pred = df[!, ens_col(model, target, "median")]
    spread = df[!, ens_col(model, target, "sd")]
    obs = observed_column(df, target)
    iv = filter(r -> String(r.target) == target, quantiles)
    nrow(iv) == nrow(unique(iv, :row_id)) || error("$model $target quantiles are not one row per id")
    qcols = setdiff(propertynames(iv), (:row_id, :target))
    joined = leftjoin(
        DataFrame(row_id = df.row_id, pred = pred, sd = spread, obs = obs),
        select(iv, :row_id, qcols...),
        on = :row_id,
    )
    mask = map(finite_value, joined.pred) .& map(finite_value, joined.sd) .& map(finite_value, joined.obs)
    for name in qcols
        mask .&= map(finite_value, joined[!, name])
    end
    return joined[mask, :]
end

function qvalues(frame, percent::Integer)
    return Float64.(getproperty(frame, level_name(percent)))
end

function calibration_row(rows, target, model, obs, level_pct)
    lo = qvalues(rows, div(100 - level_pct, 2))
    hi = qvalues(rows, div(100 + level_pct, 2))
    width = hi .- lo
    return (;
        target, model, level_pct, n = length(obs),
        qcp = mean(obs .<= qvalues(rows, level_pct)),
        picp = mean((obs .>= lo) .& (obs .<= hi)),
        mean_piw = mean(width),
    )
end

function climatology_row(target, obs, level_pct)
    τ = level_pct / 100
    lo = quantile(obs, (100 - level_pct) / 200)
    hi = quantile(obs, (100 + level_pct) / 200)
    mark = quantile(obs, τ)
    return (;
        target, model = "Climatology", level_pct, n = length(obs),
        qcp = mean(obs .<= mark),
        picp = mean((obs .>= lo) .& (obs .<= hi)),
        mean_piw = hi - lo,
    )
end

function uncertainty_tables(df, by_model)
    summary = DataFrame(
        target = String[], model = String[], method = String[], n = Int[],
        rmse = Float64[], mean_std = Float64[], median_std = Float64[],
        mean_std_over_rmse = Float64[], nominal = Float64[], picp = Float64[],
        mean_piw = Float64[], median_piw = Float64[],
        qcp_lower = Float64[], qcp_upper = Float64[],
    )
    by_bin = DataFrame(
        target = String[], model = String[], method = String[], bin = Int[], n = Int[],
        std_min = Float64[], std_median = Float64[], std_max = Float64[],
        rmse = Float64[],
    )
    calibration = NamedTuple[]
    piw_parts = DataFrame[]
    prepared = Dict{Tuple{String, String}, DataFrame}()
    for target in TARGETS, model in MODELS
        prepared[(target, model)] = model_rows(df, by_model[model], model, target)
    end
    for target in TARGETS
        shared = intersect((Set(prepared[(target, model)].row_id) for model in MODELS)...)
        climatology_obs = nothing
        for model in MODELS
            rows = prepared[(target, model)]
            rows = rows[in.(rows.row_id, Ref(shared)), :]
            obs = Float64.(rows.obs)
            pred = Float64.(rows.pred)
            spread = Float64.(rows.sd)
            lower = qvalues(rows, 5)
            upper = qvalues(rows, 95)
            piw = upper .- lower
            err = rmse(pred, obs)
            mean_std = mean(spread)
            push!(summary, (;
                target, model, method = "Ens", n = nrow(rows),
                rmse = err, mean_std, median_std = median(spread),
                mean_std_over_rmse = mean_std / err,
                nominal = NOMINAL,
                picp = mean((obs .>= lower) .& (obs .<= upper)),
                mean_piw = mean(piw), median_piw = median(piw),
                qcp_lower = mean(obs .<= lower), qcp_upper = mean(obs .<= upper),
            ))
            bins = quantile_bin_ids(spread, N_BINS)
            for b in 1:N_BINS
                inbin = bins .== b
                s = spread[inbin]
                push!(by_bin, (;
                    target, model, method = "Ens", bin = b, n = count(inbin),
                    std_min = minimum(s), std_median = median(s), std_max = maximum(s),
                    rmse = rmse(pred[inbin], obs[inbin]),
                ))
            end
            for level_pct in DECILE_PERCENTS
                push!(calibration, calibration_row(rows, target, model, obs, level_pct))
            end
            push!(piw_parts, DataFrame(target = fill(target, length(piw)), model = fill(model, length(piw)), piw = piw))
            climatology_obs === nothing && (climatology_obs = obs)
        end
        for level_pct in DECILE_PERCENTS
            push!(calibration, climatology_row(target, climatology_obs, level_pct))
        end
    end
    return summary, by_bin, DataFrame(calibration), vcat(piw_parts...)
end

root = dirname(@__DIR__)
df = load_joined(root)
quantiles = Dict{String, DataFrame}()
for model in MODELS
    table, n_members = member_quantiles(members_path(root, model))
    quantiles[model] = table
    @info "ensemble members" model n_members
end
summary, by_bin, calibration, piw = uncertainty_tables(df, quantiles)
out = results_dir(root)
mkpath(out)
summary_path = joinpath(out, "cv_ensemble_uncertainty_summary.csv")
bins_path = joinpath(out, "cv_ensemble_uncertainty_rmse_by_std.csv")
calibration_path = joinpath(out, "cv_ensemble_calibration.csv")
piw_path = joinpath(out, "cv_ensemble_piw.csv")
CSV.write(summary_path, summary)
CSV.write(bins_path, by_bin)
CSV.write(calibration_path, calibration)
CSV.write(piw_path, piw)
@info "wrote" summary_path bins_path calibration_path piw_path
show(summary; allrows = true, allcols = true)
println()
show(filter(r -> r.level_pct == 90, calibration); allrows = true, allcols = true)
println()
