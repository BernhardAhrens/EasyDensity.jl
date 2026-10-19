using CSV, DataFrames
using Statistics
using CairoMakie
using AlgebraOfGraphics
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

# ! filter rows where `bd` and `soc` are non-missing
df = dropmissing(df, [:bd, :soc])

CairoMakie.activate!() # uncomment this to save pdf files.
figdir = figure_results()

function draw_joint(df, source)
xy_to = joint_pairs(source)

with_theme(theme_latexfonts()) do

        fig = Figure(; size = (1200, 350), fontsize=15)
        axs = [Axis(fig[1, j], aspect= 1, xlabel = "BD (g/cm3)", ylabel = "SOC content (g/kg)",
            xlabelsize = 16, ylabelsize=16, xticklabelsize = 16, yticklabelsize=16,
            titlefont = :regular)
            for j in 1:4]
        plt = nothing
        set_upper_count = 1000
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
            ticks = [1, 10, 100, 1000],
            scale=log10)
        limits!.(axs, 0, 2, 0, 625)
        hideydecorations!.(axs[2:end], ticks=false, grid=false)
        hidespines!.(axs, :t, :r)
        fig
        save(joinpath(figdir, "joint_distribution_$(source).png"), fig)
end

# do a normal density heatmap!
with_theme(theme_latexfonts()) do

        fig = Figure(; size = (1200, 350), fontsize=15)
        axs = [Axis(fig[1, j], aspect= 1, xlabel = "BD (g/cm3)", ylabel = "SOC content (g/kg)", titlefont = :regular)
            for j in 1:4]
        plt_aog = nothing
        set_upper_count = 10000 
        for (i, model_combo) in enumerate(xy_to)
            @show model_combo
            y_x = df[!, "$(model_combo[1])"]
            y_y = df[!, "$(model_combo[2])"]

            y_x_new, y_y_new = compute_apply_mask(y_x, y_y);
            @show size(y_x_new), size(y_y_new)

            df_aog = (x=y_x_new, y=y_y_new)
            plt = data(df_aog) * mapping(:x, :y) * histogram(bins=(30, 30))
            plt_aog = draw!(axs[i], plt, scales(Color = (; colormap = cmap_list, scale=log10, colorrange = (1, set_upper_count), highclip = :grey20, lowclip=:transparent )))
            axs[i].title = titles[i]
        end
        # ! label panel
        [Label(fig[1, j, TopLeft()], k,
            fontsize = 22,
            padding = (0, 10, -30, 0),
            halign = :right
            ) for (j, k) in enumerate(["(a)", "(b)", "(c)", "(d)"])]
        # @show plt_aog[1].axis.plt
        cb = Colorbar(fig[1, 5], limits = (1, set_upper_count),
            colormap = cmap_list,
            label = rich(rich("Count", font=:bold),
                ),
            # labelrotation = 0,
            minorticksvisible=true,
            minorticks=IntervalsBetween(9),
            ticks = [1, 10, 100, 1000, 10_000],
             highclip = :grey20, lowclip=:transparent,
            scale=log10)
        limits!.(axs, 0, 2, 0, 625)
        hideydecorations!.(axs[2:end], ticks=false, grid=false)
        hidespines!.(axs, :t, :r)
        fig
        save(joinpath(figdir, "joint_distribution_density_heatmap_$(source).png"), fig)
end


# ! check numbers
# variables to check (same as python)
vars_to_check = ocd_cols(source)

df_filtered = copy(df)

# ---- 1. COUNT IDS BY NUMBER OF TIME POINTS ----
id_counts = combine(groupby(df_filtered, :id), nrow => :count)

num_ids_all3  = count(==(3), id_counts.count)
num_ids_not3  = count(!=(3), id_counts.count)

println("IDs with all 3 time points: ", num_ids_all3)
println("IDs WITHOUT all 3 time points: ", num_ids_not3)

# ---- 2. KEEP ONLY IDS THAT HAVE ALL 3 ROWS ----
valid_ids = id_counts.id[id_counts.count .== 3]

df3 = filter(row -> row.id in valid_ids, df_filtered)

println("Filtered dataframe shape: ", size(df3))

# ---- 3. TEMPORAL STABILITY USING IQR AND RANGE ----
# Compute min and max per id for each variable
stability = combine(
    groupby(df3, :id),
    [
        var => minimum => Symbol(var * "_min") for var in vars_to_check
    ]...,
    [
        var => maximum => Symbol(var * "_max") for var in vars_to_check
    ]...
)

# Compute range (max - min)
for var in vars_to_check
    stability[!, Symbol(var * "_range")] =
        stability[!, Symbol(var * "_max")] .-
        stability[!, Symbol(var * "_min")]
end

val_labels = ["UniNN","MultiNN","SiNN"]

rows = NamedTuple[]
for (ii, var) in zip(val_labels, vars_to_check)
    col = skipmissing(stability[!, Symbol(var * "_range")])

    q05 = quantile(col, 0.05)
    med = median(col)
    q95 = quantile(col, 0.95)

    println(source, " ", ii, ": ", q05, "  ", med, "  ", q95)
    push!(rows, (; source, model = ii, q05, median = med, q95, n = length(col)))
end
return DataFrame(rows)
end

tables = DataFrame[]
for source in prediction_sources()
    push!(tables, draw_joint(df, source))
end
quantiles = vcat(tables...)
quantile_path = joinpath(figdir, "temporal_range_quantiles.csv")
CSV.write(quantile_path, quantiles)
println("wrote ", quantile_path)