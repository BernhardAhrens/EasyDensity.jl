const ENS_VERSION = "v20261002"
const MODELS = ("UniNN", "MultiNN", "SiNN")

ens_col(model, target, stat = "median") = "$(model)_$(target)_ens_$(stat)"
latent_col(name, stat = "median") = "pred_$(name)_ens_$(stat)"

function attach_ensemble(df, root)
    for model in MODELS
        path = joinpath(root, "eval", "cv_ensemble_$(model)_$(ENS_VERSION).csv")
        isfile(path) || error("missing ensemble table $path")
        piece = CSV.read(path, DataFrame)
        fresh = setdiff(propertynames(piece), (:row_id, propertynames(df)...))
        df = leftjoin(df, select(piece, :row_id, fresh...), on = :row_id)
    end
    return df
end

function joint_pairs()
    pairs = [["bd", "soc"]]
    for model in MODELS
        push!(pairs, [ens_col(model, "BD"), ens_col(model, "SOCconc")])
    end
    return pairs
end

ocd_cols() = [ens_col(model, "SOCdensity") for model in MODELS]
