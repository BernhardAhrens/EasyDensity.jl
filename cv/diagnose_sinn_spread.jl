# Why the SiNN ensemble is wider on SOC density than UniNN / MultiNN.
#
# The hybrid equation fixes SOC density as SOC × BD × (1 − CF).
# Bulk density itself is the organic/mineral mixing formula in oBD and mBD.
# This script does not retrain. It decomposes the saved ensembles.
#
#   julia --project=. cv/diagnose_sinn_spread.jl

using CSV, DataFrames, Statistics, Random, LinearAlgebra, EasyHybrid

const ROOT = dirname(@__DIR__)
const VERSION = "v20261002"
const MODELS = ("UniNN", "MultiNN", "SiNN")
const TARGETS = ("SOCconc", "CF", "BD", "SOCdensity")
const OBS = Dict("SOCconc" => :soc, "CF" => :cf, "BD" => :bd, "SOCdensity" => :ocd)
const SCALERS = (SOCconc = 0.151, CF = 0.263, BD = 0.529, SOCdensity = 0.167)
const OBD_LO, OBD_HI = 0.05, 0.40
const MBD_LO, MBD_HI = 0.75, 2.0

finite_value(v) = v isa Real && isfinite(Float64(v))

function write_out(name, df)
    dir = joinpath(ROOT, "results")
    mkpath(dir)
    path = joinpath(dir, "sinn_spread_$(name)_$(VERSION).csv")
    CSV.write(path, df)
    @info "wrote" path
    show(df; allrows = true, allcols = true)
    println()
    return df
end

function assign_study_folds()
    ids = CSV.read(joinpath(ROOT, "data", "lucas_preprocessed_v20251125.csv"), DataFrame; select = [:row_id]).row_id
    Random.seed!(42)
    folds = make_folds(DataFrame(row_id = ids), k = 5, shuffle = true)
    return Dict(Int(id) => Int(f) for (id, f) in zip(ids, folds))
end

function load_obs()
    df = CSV.read(joinpath(ROOT, "eval", "all_cv.pred_with.lc_v20251209.csv"), DataFrame;
        select = [:row_id, :soc, :cf, :bd, :ocd, :LC_group])
    df.row_id = Int.(df.row_id)
    return df
end

function observed_ids(obs, target)
    col = obs[!, OBS[target]]
    return Int[id for (id, v) in zip(obs.row_id, col) if finite_value(v)]
end

function load_members(model)
    path = joinpath(ROOT, "eval", "cv_ensemble_members_$(model)_$(VERSION).csv")
    df = CSV.read(path, DataFrame; select = [:row_id, :target, :seed, :prediction],
        types = Dict(:row_id => Int, :seed => Int, :prediction => Float64))
    mats = Dict{String,Tuple{Vector{Int},Matrix{Float64}}}()
    for sub in groupby(df, :target)
        name = String(sub.target[1])
        sub = sort(sub, [:row_id, :seed])
        seed_ids = sort(unique(sub.seed))
        nseed = length(seed_ids)
        nrow(sub) % nseed == 0 || error("$model $name is not a multiple of $nseed seeds")
        n = nrow(sub) ÷ nseed
        seeds = reshape(sub.seed, nseed, n)
        ids = reshape(sub.row_id, nseed, n)
        all(seeds .== reshape(seed_ids, nseed, 1)) || error("$model $name seeds are not the same ordered set in each row")
        all(ids .== ids[1:1, :]) || error("$model $name row ids are not constant within a seed block")
        mats[name] = (vec(ids[1, :]), reshape(sub.prediction, nseed, n))
    end
    return mats
end

function align(mats, name, ids)
    have, M = mats[name]
    pos = Dict(id => i for (i, id) in enumerate(have))
    idx = [pos[id] for id in ids]
    return M[:, idx]
end

function column_std(M)
    return vec(std(M; dims = 1))
end

function column_mean(M)
    return vec(mean(M; dims = 1))
end

function scaled_matrix(target, M)
    target == "BD" && return M .* SCALERS.BD
    target == "SOCdensity" && return SCALERS.SOCdensity .* log.(M)
    target == "SOCconc" && return SCALERS.SOCconc .* log.(M .+ 1)
    target == "CF" && return SCALERS.CF .* log.(M .+ 1)
    return M
end

function median_finite(x)
    y = filter(isfinite, x)
    return isempty(y) ? NaN : median(y)
end

function mean_offdiag_cor(P)
    nseed = size(P, 1)
    C = cor(collect(transpose(P)))
    acc = Float64[]
    for i in 1:nseed, j in (i + 1):nseed
        v = C[i, j]
        isfinite(v) && push!(acc, v)
    end
    return isempty(acc) ? NaN : mean(acc)
end

function product_matrix(soc, bd, cf)
    return soc .* bd .* (1 .- cf ./ 100)
end

function bd_scalar(soc, obd, mbd)
    soct = max(soc / 1000, 1.0e-7)
    som = clamp(1.724 * soct, 0.0, 1.0)
    denom = som * mbd + (1 - som) * obd
    return (obd * mbd) / denom
end

function ocd_scalar(soc, bd, cf)
    return soc * bd * (1 - cf / 100)
end

function numeric_grad(f, x)
    g = similar(x)
    for i in eachindex(x)
        e = 1.0e-5 * max(1.0, abs(x[i]))
        xp, xm = copy(x), copy(x)
        xp[i] += e
        xm[i] -= e
        g[i] = (f(xp) - f(xm)) / (2 * e)
    end
    return g
end

function quad_form(g, vars)
    k = length(vars)
    Σ = Matrix{Float64}(undef, k, k)
    for i in 1:k, j in 1:k
        Σ[i, j] = i == j ? var(vars[i]) : cov(vars[i], vars[j])
    end
    indep = sum(g[i]^2 * Σ[i, i] for i in 1:k)
    full = dot(g, Σ * g)
    return full, indep
end

function pooled_share(parts)
    den = sum(p -> p.tot, parts)
    den == 0 && return ntuple(_ -> NaN, length(parts[1]) - 1)
    keys = propertynames(parts[1])
    return Tuple(sum(p -> getproperty(p, k), parts) / den for k in keys if k != :tot)
end

function headline(models, obs_ids, fold_of)
    rows = NamedTuple[]
    fold_rows = NamedTuple[]
    for model in MODELS
        @info "headline" model
        mats = models[model]
        for target in TARGETS
            ids = obs_ids[target]
            M = align(mats, target, ids)
            σ = column_std(M)
            μ = column_mean(M)
            cv = σ ./ abs.(μ)
            σs = column_std(scaled_matrix(target, M))
            push!(rows, (;
                model, target, n = length(ids),
                median_sd = median(σ), median_cv = median_finite(cv), median_scaled_sd = median(σs),
            ))
            folds = [fold_of[id] for id in ids]
            for f in 1:5
                sel = folds .== f
                Mf = M[:, sel]
                raw = column_std(Mf)
                seed_means = vec(mean(Mf; dims = 2))
                shape = column_std(Mf .- reshape(seed_means, :, 1))
                push!(fold_rows, (;
                    model, target, test_fold = f, n = count(sel),
                    median_sd = median(raw),
                    seed_offset_sd = std(seed_means),
                    median_shape_sd = median(shape),
                    seed_cor = mean_offdiag_cor(Mf),
                    seed_cor_log = mean_offdiag_cor(scaled_matrix(target, Mf)),
                ))
            end
        end
        models[model] = mats
    end
    return DataFrame(rows), DataFrame(fold_rows)
end

function decompose_products(models, ids, soc_obs)
    summary = NamedTuple[]
    bin_rows = NamedTuple[]
    som_ref = nothing
    for model in MODELS
        @info "product decomposition" model
        mats = models[model]
        soc = align(mats, "SOCconc", ids)
        bd = align(mats, "BD", ids)
        cf = align(mats, "CF", ids)
        ocd = align(mats, "SOCdensity", ids)
        prod = product_matrix(soc, bd, cf)
        rel = abs.(prod .- ocd) ./ max.(abs.(ocd), 1e-8)
        μsoc = reshape(column_mean(soc), 1, :)
        μbd = reshape(column_mean(bd), 1, :)
        μcf = reshape(column_mean(cf), 1, :)
        nseed = size(soc, 1)
        only_soc = product_matrix(soc, repeat(μbd, nseed, 1), repeat(μcf, nseed, 1))
        only_bd = product_matrix(repeat(μsoc, nseed, 1), bd, repeat(μcf, nseed, 1))
        only_cf = product_matrix(repeat(μsoc, nseed, 1), repeat(μbd, nseed, 1), cf)
        n = length(ids)
        shares = NamedTuple[]
        ratio_indep = Vector{Float64}(undef, n)
        sd_direct = column_std(ocd)
        sd_prod = column_std(prod)
        sd_soc = column_std(only_soc)
        sd_bd = column_std(only_bd)
        sd_cf = column_std(only_cf)
        cor_soc_bd = Vector{Float64}(undef, n)
        cor_soc_cf = Vector{Float64}(undef, n)
        cor_bd_cf = Vector{Float64}(undef, n)
        for j in 1:n
            sj, bj, cj = soc[:, j], bd[:, j], cf[:, j]
            a, b, c = log.(sj), log.(bj), log.(1 .- cj ./ 100)
            if all(isfinite, a) && all(isfinite, b) && all(isfinite, c)
                va, vb, vc = var(a), var(b), var(c)
                cab, cac, cbc = cov(a, b), cov(a, c), cov(b, c)
                tot = var(a .+ b .+ c)
                push!(shares, (; va, vb, vc, cab = 2cab, cac = 2cac, cbc = 2cbc, tot))
            end
            μ = [mean(sj), mean(bj), mean(cj)]
            g = numeric_grad(x -> ocd_scalar(x[1], x[2], x[3]), μ)
            full, indep = quad_form(g, (sj, bj, cj))
            ratio_indep[j] = full > 0 ? sqrt(indep / full) : NaN
            cor_soc_bd[j] = std(sj) > 0 && std(bj) > 0 ? cor(sj, bj) : NaN
            cor_soc_cf[j] = std(sj) > 0 && std(cj) > 0 ? cor(sj, cj) : NaN
            cor_bd_cf[j] = std(bj) > 0 && std(cj) > 0 ? cor(bj, cj) : NaN
        end
        pool = pooled_share(shares)
        push!(summary, (;
            model, n,
            median_sd_direct = median(sd_direct),
            median_sd_product = median(sd_prod),
            median_rel_product_error = median(vec(rel)),
            median_sd_only_soc = median(sd_soc),
            median_sd_only_bd = median(sd_bd),
            median_sd_only_cf = median(sd_cf),
            logvar_soc = pool[1], logvar_bd = pool[2], logvar_cf = pool[3],
            logvar_cov_soc_bd = pool[4], logvar_cov_soc_cf = pool[5], logvar_cov_bd_cf = pool[6],
            median_indep_over_correlated = median_finite(ratio_indep),
            median_cor_soc_bd = median_finite(cor_soc_bd),
            median_cor_soc_cf = median_finite(cor_soc_cf),
            median_cor_bd_cf = median_finite(cor_bd_cf),
        ))
        if model == "SiNN"
            som_ref = clamp.(1.724 .* column_mean(soc) ./ 1000, 0, 1)
        end
        append!(bin_rows, soc_bins(model, soc_obs, sd_direct, sd_prod))
    end
    som_rows = som_bins(models, ids, som_ref)
    return DataFrame(summary), DataFrame(bin_rows), som_rows
end

function soc_bins(model, soc_mean, sd_direct, sd_prod)
    edges = quantile(soc_mean, [0, 0.25, 0.5, 0.75, 1])
    rows = NamedTuple[]
    for i in 1:4
        lo, hi = edges[i], edges[i + 1]
        sel = i == 4 ? (soc_mean .>= lo) .& (soc_mean .<= hi) : (soc_mean .>= lo) .& (soc_mean .< hi)
        push!(rows, (;
            model, bin = "SOC q$i", lo, hi, n = count(sel),
            median_sd_direct = median(sd_direct[sel]),
            median_sd_product = median(sd_prod[sel]),
        ))
    end
    return rows
end

function som_bins(models, ids, som)
    edges = [0.0, 0.02, 0.05, 0.10, 1.01]
    rows = NamedTuple[]
    for model in MODELS
        mats = models[model]
        bd = column_std(align(mats, "BD", ids))
        ocd = column_std(align(mats, "SOCdensity", ids))
        obd = model == "SiNN" ? column_std(align(mats, "oBD", ids)) : fill(NaN, length(ids))
        mbd = model == "SiNN" ? column_std(align(mats, "mBD", ids)) : fill(NaN, length(ids))
        for i in 1:4
            lo, hi = edges[i], edges[i + 1]
            sel = (som .>= lo) .& (som .< hi)
            count(sel) == 0 && continue
            push!(rows, (;
                model, som_lo = lo, som_hi = hi, n = count(sel),
                median_sd_bd = median(bd[sel]),
                median_sd_ocd = median(ocd[sel]),
                median_sd_obd = median_finite(obd[sel]),
                median_sd_mbd = median_finite(mbd[sel]),
            ))
        end
    end
    return DataFrame(rows)
end

function latent_report(mats, ids)
    soc = align(mats, "SOCconc", ids)
    bd = align(mats, "BD", ids)
    obd = align(mats, "oBD", ids)
    mbd = align(mats, "mBD", ids)
    n = length(ids)
    recon = Vector{Float64}(undef, n)
    ratio = Vector{Float64}(undef, n)
    cor_lat = Vector{Float64}(undef, n)
    only_lat = Vector{Float64}(undef, n)
    only_soc = Vector{Float64}(undef, n)
    for j in 1:n
        sj, oj, mj, bj = soc[:, j], obd[:, j], mbd[:, j], bd[:, j]
        nseed = length(sj)
        hat = [bd_scalar(sj[i], oj[i], mj[i]) for i in 1:nseed]
        recon[j] = median(abs.(hat .- bj) ./ max.(abs.(bj), 1e-8))
        μs, μo, μm = mean(sj), mean(oj), mean(mj)
        hat_lat = [bd_scalar(μs, oj[i], mj[i]) for i in 1:nseed]
        hat_soc = [bd_scalar(sj[i], μo, μm) for i in 1:nseed]
        only_lat[j] = std(hat_lat)
        only_soc[j] = std(hat_soc)
        g = numeric_grad(x -> bd_scalar(x[1], x[2], x[3]), [μs, μo, μm])
        full, indep = quad_form(g, (sj, oj, mj))
        ratio[j] = full > 0 ? sqrt(indep / full) : NaN
        cor_lat[j] = std(oj) > 0 && std(mj) > 0 ? cor(oj, mj) : NaN
    end
    function occupancy(M, lo, hi)
        u = (M .- lo) ./ (hi - lo)
        near = (u .< 0.05) .| (u .> 0.95)
        both = vec(any(u .< 0.05; dims = 1) .& any(u .> 0.95; dims = 1))
        return (median_fraction_near_bound = median(vec(mean(near; dims = 1))),
            fraction_rows_spanning_both_bounds = mean(both))
    end
    obd_occ = occupancy(obd, OBD_LO, OBD_HI)
    mbd_occ = occupancy(mbd, MBD_LO, MBD_HI)
    return DataFrame([(;
        n,
        median_sd_bd = median(column_std(bd)),
        median_sd_obd = median(column_std(obd)),
        median_sd_mbd = median(column_std(mbd)),
        median_rel_bd_recon_error = median(recon),
        median_sd_bd_if_only_latents_vary = median(only_lat),
        median_sd_bd_if_only_soc_varies = median(only_soc),
        median_indep_over_correlated = median_finite(ratio),
        median_cor_obd_mbd = median_finite(cor_lat),
        obd_median_fraction_near_bound = obd_occ.median_fraction_near_bound,
        obd_fraction_rows_spanning_both_bounds = obd_occ.fraction_rows_spanning_both_bounds,
        mbd_median_fraction_near_bound = mbd_occ.median_fraction_near_bound,
        mbd_fraction_rows_spanning_both_bounds = mbd_occ.fraction_rows_spanning_both_bounds,
    )])
end

function attach_hyperparams!(fold_df)
    pieces = DataFrame[]
    for model in MODELS
        hp = CSV.read(joinpath(ROOT, "eval", "cv_$(model)_hyperparams_$(VERSION).csv"), DataFrame)
        if model == "UniNN"
            hp.target = String.(hp.target)
        else
            rows = DataFrame[]
            for target in TARGETS
                piece = copy(hp)
                piece.target .= target
                push!(rows, piece)
            end
            hp = vcat(rows...)
        end
        push!(pieces, select(hp, :target, :test_fold, :h, :bs, :lr, :act, :best_epoch))
    end
    hp = vcat((piece for piece in pieces)...)
    # model is not in hp; join per model below
    out = DataFrame[]
    for model in MODELS
        sub = fold_df[fold_df.model .== model, :]
        hsub = pieces[findfirst(==(model), MODELS)]
        push!(out, leftjoin(sub, hsub, on = [:target, :test_fold]))
    end
    return vcat(out...)
end

function main()
    @info "loading folds and observations"
    fold_of = assign_study_folds()
    obs = load_obs()
    obs_ids = Dict(target => observed_ids(obs, target) for target in TARGETS)
    ok = trues(nrow(obs))
    for target in TARGETS
        ok .&= map(finite_value, obs[!, OBS[target]])
    end
    complete = obs.row_id[ok]
    soc_obs = Float64.(obs.soc[ok])
    @info "observed rows" SOCconc = length(obs_ids["SOCconc"]) complete = length(complete)

    models = Dict{String,Any}()
    for model in MODELS
        @info "reading members" model
        models[model] = load_members(model)
    end

    head, folds = headline(models, obs_ids, fold_of)
    folds = attach_hyperparams!(folds)
    prod, socq, som = decompose_products(models, complete, soc_obs)
    lat = latent_report(models["SiNN"], complete)

    write_out("headline", head)
    write_out("folds", folds)
    write_out("product", prod)
    write_out("soc_quartiles", socq)
    write_out("som_bins", som)
    write_out("latents", lat)

    sinn = only(eachrow(prod[prod.model .== "SiNN", :]))
    multi = only(eachrow(prod[prod.model .== "MultiNN", :]))
    uni = only(eachrow(prod[prod.model .== "UniNN", :]))
    println()
    @info "SOC density SD, product of the three factors versus the stored ensemble"
    @info "ratios" sinn_over_multi = sinn.median_sd_direct / multi.median_sd_direct multi_product_over_direct = multi.median_sd_product / multi.median_sd_direct sinn_over_multi_product = sinn.median_sd_product / multi.median_sd_product uni_product_over_direct = uni.median_sd_product / uni.median_sd_direct
end

main()
