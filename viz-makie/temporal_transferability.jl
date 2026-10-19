using CSV, DataFrames
using Statistics
using CairoMakie
CairoMakie.activate!()

include(joinpath(@__DIR__, "uq_columns.jl"))

cmap_list = ["#abdda4", "#ffffbf", "#fdae61", "#d7191c"]

targets = ["SOCconc", "CF", "BD", "SOCdensity"]
labels = ["SOC content", "CF", "BD", "SOC density"]
observed = [:soc, :cf, :bd, :ocd]
upper = [625, 100, 2, 160]
units = ["g/kg", "%", "g/cm³", "kg/m³"]

function compute_apply_mask(y_pred, y_target)
    mask = map((a, b) -> !ismissing(a) && !ismissing(b) && isfinite(a) && isfinite(b), y_pred, y_target)
    return replace(y_pred[mask], missing => NaN), replace(y_target[mask], missing => NaN)
end

function compute_r2_rmse_bias(y_pred, y_target)
    err = y_pred .- y_target
    _rmse = sqrt(mean(err .^ 2))
    denom = sum((y_target .- mean(y_target)) .^ 2)
    _r2 = 1 - sum(err .^ 2) / denom
    _bias = mean(err)
    return _r2, _rmse, _bias
end

axis_quantity(kind, unit) = unit == "" ? kind : "$kind ($unit)"

function accuracy_obs(frame, target, obs, scale)
    scale == "scaled" && return frame[!, target]
    target == "CF" && return frame[!, obs] .* 100
    return frame[!, obs]
end

function accuracy_pred(frame, model, target, scale)
    scale == "unscaled" && return frame[!, value_col(model, target, "median")]
    return frame[!, "$(ens_col(model, target))_scaled"]
end

function load_transfer(root)
    obs = CSV.read(joinpath(root, "data", "lucas_preprocessed_v20251125.csv"), DataFrame;
        select = [:row_id, :time, :soc, :cf, :bd, :ocd, :SOCconc, :CF, :BD, :SOCdensity],
        normalizenames = true)
    obs = obs[obs.time .!= 2018, :]
    for model in MODELS
        path = joinpath(results_dir(root), "tc_ensemble_$(model).csv")
        isfile(path) || error("missing transfer ensemble $path")
        piece = CSV.read(path, DataFrame)
        fresh = setdiff(propertynames(piece), (:row_id, propertynames(obs)...))
        obs = leftjoin(obs, select(piece, :row_id, fresh...), on = :row_id)
    end
    add_scaled_medians!(obs)
    return obs
end

root = joinpath(@__DIR__, "..")
df = load_transfer(root)
figdir = figure_results(root)

with_theme(theme_latexfonts()) do
    for scale in ("scaled", "unscaled")
        for (k, t) in enumerate(targets)
            hi = scale == "scaled" ? 1 : upper[k]
            y_target = accuracy_obs(df, t, observed[k], scale)
            pred_label = scale == "scaled" ? "Prediction" : axis_quantity("Prediction", units[k])
            obs_label = scale == "scaled" ? "Observation" : axis_quantity("Observation", units[k])
            fig = Figure(; size = (1400, 460), fontsize = 15)
            axs = [Axis(fig[1, j], aspect = 1, xlabel = pred_label, ylabel = obs_label,
                xlabelsize = 16, ylabelsize = 16, xticklabelsize = 16, yticklabelsize = 16,
                titlefont = :regular, titlesize = 12)
                for j in 1:3]
            plt = nothing
            set_upper_count = 1000
            for (i, model) in enumerate(MODELS)
                y_pred = accuracy_pred(df, model, t, scale)
                y_pred_new, y_target_new = compute_apply_mask(y_pred, y_target)
                _r2, _rmse, _bias = compute_r2_rmse_bias(y_pred_new, y_target_new)
                println("$scale $model $t  R2=$(round(_r2, digits=3))  RMSE=$(round(_rmse, digits=4))  Bias=$(round(_bias, digits=4))  n=$(length(y_target_new))")

                plt = hexbin!(axs[i], y_pred_new, y_target_new; cellsize = hi / 40, threshold = 1,
                    colormap = cmap_list, colorscale = log10,
                    colorrange = (1, set_upper_count), highclip = :grey20, lowclip = :transparent)
                lines!(axs[i], [Point2f(0, 0), Point2f(hi, hi)], color = :grey15)
                axs[i].title = rich(rich("$model  ", font = :bold,),
                    rich("R", superscript("2"), " = $(round(_r2, digits=2))", color = :orangered),
                    rich("  RMSE = $(round(_rmse, digits=3))", color = :black),
                    rich("\nBias = $(round(_bias, digits=3))", color = :dodgerblue),
                )
            end
            [Label(fig[1, j, TopLeft()], lab,
                fontsize = 22,
                padding = (0, 10, 5, 0),
                halign = :right
            ) for (j, lab) in enumerate(["(a)", "(b)", "(c)"])]

            Colorbar(fig[1, 4], plt;
                label = rich(rich("Count", font = :bold), rich(
                    "\n\n Accuracy of\n $(labels[k]) predictions for\n other survey years using models\n trained on 2018 data."),
                    rich("\n\nTemporal transferability", font = :bold)
                ),
                labelrotation = 0,
                minorticksvisible = true,
                minorticks = IntervalsBetween(9),
                ticks = [1, 10, 100, 1000],
                scale = log10)
            limits!.(axs, 0, hi, 0, hi)
            if scale == "scaled"
                tickpos = (0:0.25:1, ["0", "0.25", "0.5", "0.75", "1"])
                [axs[j].xticks = tickpos for j in 1:3]
                [axs[j].yticks = tickpos for j in 1:3]
            end
            hideydecorations!.(axs[2:end], ticks = false, grid = false)
            hidespines!.(axs, :t, :r)
            save(joinpath(figdir, "temporal_transferability_$(t)_$(scale).png"), fig)
        end
    end
end
