const UQ_VERSION = "v20251209"

pred_col(model, target, method, suffix = "") = "$(model)_$(target)_$(method)$(suffix)"

function attach_uq(df, root)
    path = joinpath(root, "eval", "uq_cv.pred_$(UQ_VERSION).csv")
    if !isfile(path)
        @warn "uncertainty table not found" path
        return df
    end
    uq = CSV.read(path, DataFrame)
    fresh = setdiff(propertynames(uq), (:row_id,))
    return leftjoin(df, select(uq, :row_id, fresh...), on = :row_id)
end

function present_methods(df, probes)
    found = String[]
    for method in ("MC", "Ens")
        all(name -> hasproperty(df, Symbol(name)), probes(method)) || continue
        push!(found, method)
    end
    isempty(found) && @warn "no _MC or _Ens columns for this figure"
    return found
end

function add_scaled_predictions!(df, method, models, scalers)
    for mod in models
        df[!, "$(mod)_soc_$(method)"] = @. exp(df[!, pred_col(mod, "SOCconc", method)] / scalers["SOCconc"]) - 1
        df[!, "$(mod)_cf_$(method)"] = @. exp(df[!, pred_col(mod, "CF", method)] / scalers["CF"]) - 1
        df[!, "$(mod)_bd_$(method)"] = @. df[!, pred_col(mod, "BD", method)] / scalers["BD"]
        df[!, "$(mod)_ocd_$(method)"] = @. exp(df[!, pred_col(mod, "SOCdensity", method)] / scalers["SOCdensity"])
    end
    return df
end

function scaled_joint_pairs(method, models)
    pairs = [["bd", "soc"]]
    for mod in models
        push!(pairs, ["$(mod)_bd_$(method)", "$(mod)_soc_$(method)"])
    end
    return pairs
end

scaled_probes(models) = method -> [pred_col(mod, target, method)
    for mod in models for target in ("SOCconc", "CF", "BD", "SOCdensity")]
