using CSV, DataFrames
using Statistics
using CairoMakie
CairoMakie.activate!()

include(joinpath(@__DIR__, "uq_columns.jl"))

#  "CET-L19"
cmap_list = ["#abdda4", "#ffffbf", "#fdae61", "#d7191c"]

# targets = ["SOCconc", "CF", "BD", "SOCdensity"]
# labels = ["SOC content", "CF", "BD", "SOC density"]
models_raw = ["UniNN", "MultiNN", "SiNN"]

df = load_figure_frame(joinpath(@__DIR__, ".."))

md_preds = rich.(rich.(models_raw, font=:bold), " prediction")
titles = ["Observation", md_preds...]

function compute_apply_mask(y_pred, y_target)
    mask = map((a, b) -> !ismissing(a) && !ismissing(b) && isfinite(a) && isfinite(b), y_pred, y_target)
    return replace(y_pred[mask], missing => NaN), replace(y_target[mask], missing => NaN)
end

CairoMakie.activate!() # uncomment this to save pdf files.
figdir = figure_results()

# ! filter outliers !
df = subset(
    df,
    :bd  => ByRow(<(0.2)),
    :soc => ByRow(>(200));
    skipmissing = true
)

function draw_outliers(df, source)
xy_to = joint_pairs(source)

with_theme(theme_latexfonts()) do

        fig = Figure(; size = (1200, 350), fontsize=15)
        axs = [Axis(fig[1, j], aspect= 1, xlabel = "BD (g/cm3)", ylabel = "SOC content (g/kg)",
            xlabelsize = 16, ylabelsize=16, xticklabelsize = 16, yticklabelsize=16,
            titlefont = :regular)
            for j in 1:4]
        plt = nothing
        set_upper_count = 500 
        for (i, model_combo) in enumerate(xy_to)
            @show model_combo
            y_x = df[!, "$(model_combo[1])"]
            y_y = df[!, "$(model_combo[2])"]

            y_x_new, y_y_new = compute_apply_mask(y_x, y_y);
            @show size(y_x_new), size(y_y_new)

            plt = hexbin!(axs[i], y_x_new, y_y_new; cellsize = (0.1, 33), threshold = 1,
                colormap = cmap_list, colorscale=log10,
                colorrange = (1, set_upper_count), highclip = :grey20, lowclip=:transparent)

            axs[i].title = titles[i]
        end
        # ! label panel
        [Label(fig[1, j, TopLeft()], k,
            fontsize = 22,
            padding = (0, 10, -30, 0),
            halign = :right
            ) for (j, k) in enumerate(["(a)", "(b)", "(c)", "(d)"])]

        cb = Colorbar(fig[1, 5], plt;
            label = rich(rich("Count", font=:bold),
                ),
            # labelrotation = 0,
            minorticksvisible=true,
            minorticks=IntervalsBetween(9),
            ticks = [1, 10, 100, 500],
            scale=log10)
        limits!.(axs, 0, 2, 0, 625)
        hideydecorations!.(axs[2:end], ticks=false, grid=false)
        hidespines!.(axs, :t, :r)
        fig
        save(joinpath(figdir, "joint_distribution_outliers_$(source).png"), fig)
end
end

for source in prediction_sources()
    draw_outliers(df, source)
end