using CSV, DataFrames, Statistics
using CairoMakie
CairoMakie.activate!()

include(joinpath(@__DIR__, "uq_columns.jl"))

df = attach_ensemble(
    CSV.read(joinpath(@__DIR__, "../eval/all_cv.pred_with.lc_v20251209.csv"), DataFrame),
    joinpath(@__DIR__, ".."),
)

finite_value(v) = v isa Real && isfinite(Float64(v))
colors = [:grey15, :dodgerblue3, :firebrick]

function finite_column(frame, name)
    col = frame[!, name]
    return Float64.(col[map(finite_value, col)])
end

mkpath(joinpath(@__DIR__, "../figures"))

with_theme(theme_latexfonts()) do
    fig = Figure(; size = (900, 700), fontsize = 15)
    axs = [Axis(fig[1, i], title = rich(model, font = :bold),
            ylabel = i == 1 ? "Ensemble SD (kg/m³)" : "")
           for (i, model) in enumerate(MODELS)]
    for (i, model) in enumerate(MODELS)
        y = finite_column(df, ens_col(model, "SOCdensity", "sd"))
        x = fill(1, length(y))
        violin!(axs[i], x, y; width = 0.7, color = (colors[i], 0.25), strokecolor = colors[i])
        boxplot!(axs[i], x, y; width = 0.18, color = :white, strokecolor = colors[i],
            mediancolor = :orangered, whiskercolor = colors[i], show_outliers = false)
        hidexdecorations!(axs[i])
        hidespines!(axs[i], :t, :r, :b)
    end

    ax = Axis(fig[2, 1:3],
        xlabel = "Predicted SOC density (kg/m³)",
        ylabel = "Ensemble SD (kg/m³)")
    for (i, model) in enumerate(MODELS)
        med = df[!, ens_col(model, "SOCdensity", "median")]
        sd = df[!, ens_col(model, "SOCdensity", "sd")]
        keep = map(finite_value, med) .& map(finite_value, sd)
        scatter!(ax, Float64.(med[keep]), Float64.(sd[keep]);
            color = (colors[i], 0.2), markersize = 3, label = model)
    end
    axislegend(ax; position = :lt, framevisible = false)
    hidespines!(ax, :t, :r)
    save(joinpath(@__DIR__, "../figures/ensemble_uncertainty_SOCdensity.png"), fig)
end
