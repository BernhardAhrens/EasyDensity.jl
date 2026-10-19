using CSV, DataFrames
using Statistics
using CairoMakie
CairoMakie.activate!()

include(joinpath(@__DIR__, "uq_columns.jl"))

df = load_figure_frame(joinpath(@__DIR__, ".."))

function porosity_figure(df, source)
src_obd = latent_value_col("oBD", source)
src_mbd = latent_value_col("mBD", source)
# cleaning
cols = [
    :row_id, :time, :lat, :lon, :id, :nuts0, :maxdiff, :bd, :clay,
    :sand, :silt, :cf, :ocd, :soc, :SOCconc, :CF, :BD, :SOCdensity,
    :ndvi, :ndwi, :lst_night, :lst_day, :precipitation, :peat,
    Symbol(src_obd), Symbol(src_mbd), :LC1, :LC_group
]

hbd = select(df, cols)
src_obd == "pred_oBD" || rename!(hbd, Symbol(src_obd) => :pred_oBD, Symbol(src_mbd) => :pred_mBD)
println(size(hbd))

# statistics

hbd.som = hbd.soc .* 1.724 ./ 1000
hbd.som = min.(hbd.som, 1)

hbd.isom = 1 .- hbd.som

# constants
mPD = 2.7
oPD = 1.4
# porosity
hbd.poro = 1 .- (
    (hbd.som ./ oPD .+ hbd.isom ./ mPD) ./
    (hbd.som ./ hbd.pred_oBD .+ hbd.isom ./ hbd.pred_mBD)
)

# non-missing values
hbd_clean = dropmissing(hbd, [:LC_group, :poro])
stats = combine(
    groupby(hbd_clean, :LC_group),
    :poro => length   => :count,
    :poro => mean     => :mean,
    :poro => std      => :std,
    :poro => minimum  => :min,
    :poro => median   => :median,
    :poro => maximum  => :max,
)

println(stats)

# plot

lc_order = [
    "artificial",
    "bareland",
    "cropland",
    "grassland",
    "shrubland",
    "woodland",
    "wetland",
]

# only keep LC that actually exist
lc_present = unique(skipmissing(hbd.LC_group))
lc_groups = [lc for lc in lc_order if lc in lc_present]
lc_groups = lc_groups[1:min(7, length(lc_groups))]

hbd_sub = dropmissing(hbd[!, [:LC_group, :soc, :poro]])

hbd_g = groupby(hbd_sub, [:LC_group])

hbd.copa = hbd.pred_mBD ./ mPD
hbd_clean_copa = dropmissing(hbd, [:LC_group, :copa])

stats_copa = combine(
    groupby(hbd_clean_copa, :LC_group),
    :copa => length   => :count,
    :copa => mean     => :mean,
    :copa => std      => :std,
    :copa => minimum  => :min,
    :copa => median   => :median,
    :copa => maximum  => :max,
)

println(stats_copa)

hbd_copa_g = groupby(hbd_clean_copa, [:LC_group])

CairoMakie.activate!() # uncomment this to save pdf files.
figdir = figure_results()
with_theme(theme_latexfonts()) do
    fig = Figure(; figure_padding=(5,15,0,15), size = (1200, 600), fontsize=15)
    axs = [Axis(fig[i,j], xlabelsize = 16, ylabelsize=16, xticklabelsize = 16, yticklabelsize=16,) for i in 1:2 for j in 1:4]

    for (i, lc) in enumerate(lc_groups)
        soc  = hbd_g[("$(lc)",)][!, :soc]
        poro = hbd_g[("$(lc)",)][!, :poro]
        scatter!(
            axs[i],
            soc,
            poro;
            markersize = 8,
            color=:transparent,
            strokewidth=0.35,
            strokecolor=:grey25,
            rasterize = 2,
        )

        poro_mean = mean(skipmissing(poro))
        poro_std  = std(skipmissing(poro))

        text!(
            axs[i],
            0.55, 0.35,
            text = "Porosity:\nmean=$(round(poro_mean, digits=3))\nstd=$(round(poro_std, digits=3))",
            space = :relative,   # like ax.transAxes
            align = (:left, :top),
            fontsize = 15
        )

        axs[i].title = rich(string(lc), font=:regular)
        axs[i].xlabel = "SOC (g/kg)"
    end
    Box(fig[1:2,0], color=:grey95, strokevisible=false)
    Label(fig[1:2, 0], "Porosity", rotation = π/2, fontsize = 16)
    # Box(fig[3, 1:3], color=:grey95, strokevisible=false)
    # Label(fig[3, 1:3], "SOC (g/kg)", fontsize = 16)
    linkaxes!(axs[1:7])
    ylims!.(axs[1:7], 0.4, 1)
    hideydecorations!.(axs[2:4], grid=false, ticks=false)
    hideydecorations!.(axs[6:7], grid=false, ticks=false)
    # hidexdecorations!.(axs[1:3], grid=false, ticks=false)
    hidespines!.(axs[1:4], :r, :t)
    hidespines!.(axs[5:7], :r, :t)
    # boxplot
    colors = repeat([:grey15], length(lc_order))
    for (indx, f) in enumerate(lc_order)
        datam = hbd_copa_g[("$(f)",)][!, :copa]
        # filter just in case
        datam = filter(x -> x !== missing, datam)
        datam = replace(datam, missing => NaN)

        a = fill(indx, length(datam))
        boxplot!(axs[8], a, datam; whiskerwidth = 0.65, width = 0.35,
            color=:transparent, strokewidth = 1.25, outlierstrokewidth=0.85,
            outlierstrokecolor = (colors[indx], 0.45),
            strokecolor = (colors[indx], 0.85), whiskercolor = (colors[indx], 1),
            # mediancolor = :black,
            markersize=5,
            mediancolor= :orangered,
            medianlinewidth = 0.85,
            )
    end
    panel_labels=["(a)" "(b)" "(c)" "(d)"; "(e)" "(f)" "(g)" "(e)"]
    [Label(fig[i, j, TopLeft()], panel_labels[i,j],
                fontsize = 18,
                padding = (0, 5, 15, 0),
                halign = :right
                ) for i in 1:2 for j in 1:4]
    axs[8].ylabel = "Compaction indicator"
    axs[8].title = rich("Land Cover", font=:regular)
    axs[8].xticks = (1:length(lc_order), lc_order)
    axs[8].xticklabelrotation = pi/4
    axs[8].xticklabelalign = (:center, :top)
    hidespines!.(axs[8])
    colgap!(fig.layout, 10)
    # rowgap!(fig.layout, 5)
    fig
    save(joinpath(figdir, "porosity_$(source).png"), fig)
end
end

for source in prediction_sources()
    porosity_figure(df, source)
end