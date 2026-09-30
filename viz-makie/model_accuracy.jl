
using CSV, DataFrames
using Statistics, Random
using CairoMakie
CairoMakie.activate!()

include(joinpath(@__DIR__, "uq_columns.jl"))

#  "CET-L19"
cmap_list = ["#abdda4", "#ffffbf", "#fdae61", "#d7191c"]

version = "v20251209"
targets = ["SOCconc", "CF", "BD", "SOCdensity"]
labels = ["SOC content", "CF", "BD", "SOC density"]
models = ["UniNN", "MultiNN", "SiNN"]

df = attach_uq(CSV.read(joinpath(@__DIR__, "../eval/all_cv.pred_with.lc_$(version).csv"), DataFrame), joinpath(@__DIR__, ".."))
const N_MC = 100

function compute_apply_mask(y_pred, y_target)
    mask = map((a, b) -> !ismissing(a) && !ismissing(b) && isfinite(a) && isfinite(b), y_pred, y_target)
    return replace(y_pred[mask], missing => NaN), replace(y_target[mask], missing => NaN)
end

function compute_r2_rmse_bias_sd(y_pred, y_target)
    err = y_pred .- y_target
    _rmse = sqrt(mean(err .^ 2))
    denom = sum((y_target .- mean(y_target)) .^ 2)
    _r2 = 1 - sum(err .^ 2) / denom
    _bias = mean(err)
    _sd = std(err)
    return _r2, _rmse, _bias, _sd
end

function mc_member_mean_sd(y_pred, y_std, y_target; n_mc = N_MC)
    mask = map((a, b, s) -> !ismissing(a) && !ismissing(b) && !ismissing(s) && isfinite(a) && isfinite(b) && isfinite(s), y_pred, y_target, y_std)
    p = Float64.(y_pred[mask])
    s = Float64.(y_std[mask])
    o = Float64.(y_target[mask])
    r2s = Vector{Float64}(undef, n_mc)
    rmses = Vector{Float64}(undef, n_mc)
    for m in 1:n_mc
        member = p .+ s .* randn(length(p))
        r2s[m], rmses[m], _, _ = compute_r2_rmse_bias_sd(member, o)
    end
    return mean(r2s), std(r2s), mean(rmses), std(rmses)
end

CairoMakie.activate!() # uncomment this to save pdf files.
mkpath(joinpath(@__DIR__, "../figures_png/"))

function accuracy_figures(df, method)
Random.seed!(42)

with_theme(theme_latexfonts()) do
    for (k, t) in enumerate(targets)
        y_target = df[!, t]
        fig = Figure(; size = (1400, 460), fontsize=15)
        axs = [Axis(fig[1, j], aspect= 1, xlabel = "Prediction", ylabel = "Observation",
            xlabelsize = 16, ylabelsize=16, xticklabelsize = 16, yticklabelsize=16,
            titlefont = :regular, titlesize = 12)
            for j in 1:3]
        plt = nothing
        set_upper_count = 1000 # ? we could have different bounds for different targets, but having one to compare among all is good, unless we want to highlight the difference also on the amount of samples per variable. 
        for (i, model) in enumerate(models)
            # @show "$(model)_$t"
            y_pred = df[!, pred_col(model, t, method)]
            y_std = df[!, pred_col(model, t, method, "_std")]
            y_pred_new, y_target_new = compute_apply_mask(y_pred, y_target);
            # @show size(y_target_new), size(y_pred_new)
            _r2, _rmse, _bias, _sd = compute_r2_rmse_bias_sd(y_pred_new, y_target_new)
            mr2, sr2, mrmse, srmse = mc_member_mean_sd(y_pred, y_std, y_target)
            println("$method $model $t  R2=$(round(_r2, digits=3))  RMSE=$(round(_rmse, digits=4))  Bias=$(round(_bias, digits=4))  SD=$(round(_sd, digits=4))  meanR2=$(round(mr2, digits=3))±$(round(sr2, digits=3))  meanRMSE=$(round(mrmse, digits=4))±$(round(srmse, digits=4))  n=$(length(y_target_new))")

            plt = hexbin!(axs[i], y_pred_new, y_target_new; cellsize = 0.025, threshold = 1,
                colormap = cmap_list, colorscale=log10,
                colorrange = (1, set_upper_count), highclip = :grey20, lowclip=:transparent)
            # ! one to one line
            lines!(axs[i], [Point2f(0,0), Point2f(1,1)], color = :grey15)
            # ! title
            axs[i].title = rich(rich("$model  ", font=:bold,),
                rich("R", superscript("2"), " = $(round(_r2, digits=2))", color=:orangered),
                rich("  RMSE = $(round(_rmse, digits=3))", color=:black),
                rich("\nmean R", superscript("2"), " ± SD = $(round(mr2, digits=2)) ± $(round(sr2, digits=2))", color=:orangered),
                rich("\nmean RMSE ± SD = $(round(mrmse, digits=3)) ± $(round(srmse, digits=3))", color=:black),
                rich("\nBias ± SD = $(round(_bias, digits=3)) ± $(round(_sd, digits=3))", color=:dodgerblue),
                )
        end
        # ! label panel
        [Label(fig[1, j, TopLeft()], k,
            fontsize = 22,
            padding = (0, 10, 5, 0),
            halign = :right
            ) for (j, k) in enumerate(["(a)", "(b)", "(c)"])]

        cb = Colorbar(fig[1, 4], plt;
            label = rich(rich("Count", font=:bold), rich(
                "\n\n Cross-validation performance of\n $(labels[k]) predictions."), # for\n other survey years using models trained\n on 2018 data.
                # rich("\n\nTemporal transferability", font=:bold)
                ),
            labelrotation = 0,
            minorticksvisible=true,
            minorticks=IntervalsBetween(9),
            ticks = [1, 10, 100, 1000],
            scale=log10)
        limits!.(axs, 0, 1, 0, 1)
        [axs[j].xticks = (0:0.25:1, ["0", "0.25", "0.5", "0.75", "1"]) for j in 1:3]
        [axs[j].yticks = (0:0.25:1, ["0", "0.25", "0.5", "0.75", "1"]) for j in 1:3]
        hideydecorations!.(axs[2:end], ticks=false, grid=false)
        hidespines!.(axs, :t, :r)
        fig
        save(joinpath(@__DIR__, "../figures_png/model_accuracy_$(t)_$(method).png"), fig)
    end
end
end

for method in present_methods(df, method -> [pred_col(model, t, method, suffix) for model in models for t in targets for suffix in ("", "_std")])
    accuracy_figures(df, method)
end