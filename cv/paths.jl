# Run date and output paths. The shell sets EASYDENSITY_VERSION to vYYYYMMDD.
# Products of that run live in results/YYYYMMDD/.

using Dates

function result_version()
    tag = get(ENV, "EASYDENSITY_VERSION", "")
    tag == "" && (tag = "v" * Dates.format(today(), dateformat"yyyymmdd"))
    if get(ENV, "EASYDENSITY_SMOKE", "0") == "1" && !endswith(tag, "_smoke")
        tag *= "_smoke"
    end
    occursin(r"^v\d{8}(_smoke)?$", tag) ||
        error("EASYDENSITY_VERSION must look like vYYYYMMDD, got $tag")
    return tag
end

function result_folder(version::AbstractString = result_version())
    m = match(r"^v(\d{8}(_smoke)?)$", version)
    m === nothing && error("EASYDENSITY_VERSION must look like vYYYYMMDD, got $version")
    return m.captures[1]
end

function results_dir(root, version::AbstractString = result_version())
    return joinpath(root, "results", result_folder(version))
end

function observation_path(root, version::AbstractString = result_version())
    return joinpath(results_dir(root, version), "all_cv.pred_with.lc.csv")
end

function hyperparams_path(root, architecture, version::AbstractString = result_version())
    return joinpath(results_dir(root, version), "cv_$(architecture)_hyperparams.csv")
end

function ensemble_path(root, architecture, version::AbstractString = result_version())
    return joinpath(results_dir(root, version), "cv_ensemble_$(architecture).csv")
end

function members_path(root, architecture, version::AbstractString = result_version())
    return joinpath(results_dir(root, version), "cv_ensemble_members_$(architecture).csv")
end
