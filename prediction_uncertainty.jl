using CSV, DataFrames
using Statistics

# Sharpness, 95% interval coverage, and RMSE in bins of predictive std
# for the _MC and _Ens columns written by deep_ensemble.jl.

const VERSION = "v20251209"
const TARGETS = ["SOCconc", "CF", "BD", "SOCdensity"]
const MODELS = ["UniNN", "MultiNN", "SiNN"]
const METHODS = ["MC", "Ens"]
const N_BINS = 5

finite_value(v) = v isa Real && isfinite(Float64(v))

uq_name(model, target, method, suffix = "") = "$(model)_$(target)_$(method)$(suffix)"

function methods_present(df)
    found = String[]
    for method in METHODS
        needed = [uq_name(model, target, method, suffix)
                  for model in MODELS for target in TARGETS
                  for suffix in ("", "_std", "_lower", "_upper")]
        all(name -> hasproperty(df, name), needed) && push!(found, method)
    end
    isempty(found) && error("uq table has no complete _MC or _Ens columns")
    return found
end

function common_mask(df, target, methods)
    mask = map(finite_value, df[!, target])
    for method in methods
        for model in MODELS
            for suffix in ("", "_std", "_lower", "_upper")
                mask .&= map(finite_value, df[!, uq_name(model, target, method, suffix)])
            end
        end
    end
    return mask
end

rmse(pred, obs) = sqrt(mean((pred .- obs) .^ 2))

# Rank bins keep about the same count in each bin when std values tie.
function quantile_bin_ids(x, nbins)
    n = length(x)
    ids = Vector{Int}(undef, n)
    for (rank, i) in enumerate(sortperm(x))
        ids[i] = 1 + div((rank - 1) * nbins, n)
    end
    return ids
end

function uncertainty_tables(df, methods)
    summary = DataFrame(
        target = String[], model = String[], method = String[], n = Int[],
        rmse = Float64[], mean_std = Float64[], median_std = Float64[],
        mean_std_over_rmse = Float64[], coverage = Float64[],
    )
    by_bin = DataFrame(
        target = String[], model = String[], method = String[], bin = Int[], n = Int[],
        std_min = Float64[], std_median = Float64[], std_max = Float64[],
        rmse = Float64[],
    )
    for target in TARGETS
        mask = common_mask(df, target, methods)
        obs = Float64.(df[mask, target])
        for method in methods, model in MODELS
            pred = Float64.(df[mask, uq_name(model, target, method)])
            spread = Float64.(df[mask, uq_name(model, target, method, "_std")])
            lower = Float64.(df[mask, uq_name(model, target, method, "_lower")])
            upper = Float64.(df[mask, uq_name(model, target, method, "_upper")])
            err = rmse(pred, obs)
            mean_std = mean(spread)
            push!(summary, (;
                target, model, method, n = length(obs),
                rmse = err, mean_std, median_std = median(spread),
                mean_std_over_rmse = mean_std / err,
                coverage = mean((obs .>= lower) .& (obs .<= upper)),
            ))
            bins = quantile_bin_ids(spread, N_BINS)
            for b in 1:N_BINS
                inbin = bins .== b
                s = spread[inbin]
                push!(by_bin, (;
                    target, model, method, bin = b, n = count(inbin),
                    std_min = minimum(s), std_median = median(s), std_max = maximum(s),
                    rmse = rmse(pred[inbin], obs[inbin]),
                ))
            end
        end
    end
    return summary, by_bin
end

function main()
    src = joinpath(@__DIR__, "eval", "uq_cv.pred_$(VERSION).csv")
    isfile(src) || error("missing $src")
    df = CSV.read(src, DataFrame)
    summary, by_bin = uncertainty_tables(df, methods_present(df))
    eval_dir = joinpath(@__DIR__, "eval")
    CSV.write(joinpath(eval_dir, "prediction_uncertainty_summary_$(VERSION).csv"), summary)
    CSV.write(joinpath(eval_dir, "prediction_uncertainty_rmse_by_std_$(VERSION).csv"), by_bin)
    show(summary; allrows = true, allcols = true)
    println()
    show(by_bin; allrows = true, allcols = true)
    println()
    return summary, by_bin
end

main()
