using CSV, DataFrames
using Statistics
using CairoMakie
CairoMakie.activate!()

include(joinpath(@__DIR__, "uq_columns.jl"))

#  "CET-L19"
cmap_list = ["#abdda4", "#ffffbf", "#fdae61", "#d7191c"]

targets = ["SOCconc", "CF", "BD", "SOCdensity"]
labels = ["SOC content", "CF", "BD", "SOC density"]
observed = [:soc, :cf, :bd, :ocd]
upper = [625, 100, 2, 160]
units = ["g/kg", "%", "g/cm³", "kg/m³"]

df = load_figure_frame(joinpath(@__DIR__, ".."))
"median" in prediction_sources() && "scaled" in scale_modes() && add_scaled_medians!(df)

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

function accuracy_pred(frame, model, target, source, scale)
    scale == "unscaled" && return frame[!, value_col(model, target, source)]
    source == "tune" && return frame[!, "$(model)_$(target)"]
    return frame[!, "$(ens_col(model, target))_scaled"]
end

CairoMakie.activate!() # uncomment this to save pdf files.
figdir = figure_results()

with_theme(theme_latexfonts()) do
    for source in prediction_sources(), scale in scale_modes()
    for (k, t) in enumerate(targets)
        hi = scale == "scaled" ? 1 : upper[k]
        y_target = accuracy_obs(df, t, observed[k], scale)
        pred_label = scale == "scaled" ? "Prediction" : axis_quantity("Prediction", units[k])
        obs_label = scale == "scaled" ? "Observation" : axis_quantity("Observation", units[k])
        fig = Figure(; size = (1400, 460), fontsize=15)
        axs = [Axis(fig[1, j], aspect= 1, xlabel = pred_label, ylabel = obs_label,
            xlabelsize = 16, ylabelsize=16, xticklabelsize = 16, yticklabelsize=16,
            titlefont = :regular, titlesize = 12)
            for j in 1:3]
        plt = nothing
        set_upper_count = 1000 # ? we could have different bounds for different targets, but having one to compare among all is good, unless we want to highlight the difference also on the amount of samples per variable.
        for (i, model) in enumerate(MODELS)
            y_pred = accuracy_pred(df, model, t, source, scale)
            y_pred_new, y_target_new = compute_apply_mask(y_pred, y_target)
            _r2, _rmse, _bias = compute_r2_rmse_bias(y_pred_new, y_target_new)
            println("$source $scale $model $t  R2=$(round(_r2, digits=3))  RMSE=$(round(_rmse, digits=4))  Bias=$(round(_bias, digits=4))  n=$(length(y_target_new))")

            plt = hexbin!(axs[i], y_pred_new, y_target_new; cellsize = hi / 40, threshold = 1,
                colormap = cmap_list, colorscale=log10,
                colorrange = (1, set_upper_count), highclip = :grey20, lowclip=:transparent)
            # ! one to one line
            lines!(axs[i], [Point2f(0, 0), Point2f(hi, hi)], color = :grey15)
            # ! title
            axs[i].title = rich(rich("$model  ", font=:bold,),
                rich("R", superscript("2"), " = $(round(_r2, digits=2))", color=:orangered),
                rich("  RMSE = $(round(_rmse, digits=3))", color=:black),
                rich("\nBias = $(round(_bias, digits=3))", color=:dodgerblue),
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
                "\n\n Cross-validation performance of\n $(labels[k]) predictions."),
                ),
            labelrotation = 0,
            minorticksvisible=true,
            minorticks=IntervalsBetween(9),
            ticks = [1, 10, 100, 1000],
            scale=log10)
        limits!.(axs, 0, hi, 0, hi)
        if scale == "scaled"
            tickpos = (0:0.25:1, ["0", "0.25", "0.5", "0.75", "1"])
            [axs[j].xticks = tickpos for j in 1:3]
            [axs[j].yticks = tickpos for j in 1:3]
        end
        hideydecorations!.(axs[2:end], ticks=false, grid=false)
        hidespines!.(axs, :t, :r)
        save(joinpath(figdir, "model_accuracy_$(t)_$(source)_$(scale).png"), fig)
    end
    end
end
