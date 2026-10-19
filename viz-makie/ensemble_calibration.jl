using CSV, DataFrames, Statistics
using CairoMakie
CairoMakie.activate!()

include(joinpath(@__DIR__, "uq_columns.jl"))

const TARGETS = ("SOCconc", "CF", "BD", "SOCdensity")
const LABELS = ("SOC content", "CF", "BD", "SOC density")
const UNITS = ("g/kg", "%", "g/cm³", "kg/m³")
const COLORS = [:grey15, :dodgerblue3, :firebrick]
const TICKS = 10:10:90

root = joinpath(@__DIR__, "..")
figdir = figure_results(root)
calibration = CSV.read(joinpath(figdir, "cv_ensemble_calibration.csv"), DataFrame)
piw = CSV.read(joinpath(figdir, "cv_ensemble_piw.csv"), DataFrame)

function series(frame, target, model)
    sub = filter(r -> r.target == target && r.model == model, frame)
    return sort(sub, :level_pct)
end

# Width of each y-bin follows its count, so the point cloud has a violin silhouette.
# Values above ymax stay at the center and are left to the axis limit.
function binned_swarm(y; width = 0.36, nbins = 50, ymax = maximum(y))
    x = zeros(length(y))
    isempty(y) && return x
    span = ymax - 0.0
    span == 0 && return x
    inside = (y .>= 0) .& (y .<= ymax)
    bin = fill(0, length(y))
    bin[inside] = clamp.(floor.(Int, y[inside] ./ span .* nbins) .+ 1, 1, nbins)
    counts = [count(==(b), bin) for b in 1:nbins]
    maxc = maximum(counts)
    maxc == 0 && return x
    for b in 1:nbins
        idx = findall(==(b), bin)
        m = length(idx)
        m == 0 && continue
        half = width * m / maxc
        order = sortperm(@view y[idx])
        xs = m == 1 ? [0.0] : collect(range(-half, half; length = m))
        x[idx[order]] = xs
    end
    return x
end

with_theme(theme_latexfonts()) do
    fig = Figure(; size = (1180, 640), fontsize = 14)
    for (row, metric, ylabel, xlabel) in (
        (1, :qcp, "QCP (%)", "Quantile level (%)"),
        (2, :picp, "PICP (%)", "PI level (%)"),
    )
        for (k, target) in enumerate(TARGETS)
            ax = Axis(fig[row, k];
                title = row == 1 ? LABELS[k] : "",
                xlabel = row == 2 ? xlabel : "",
                ylabel = k == 1 ? ylabel : "",
                xticks = TICKS, yticks = TICKS,
            )
            limits!(ax, 0, 100, 0, 100)
            lines!(ax, [0, 100], [0, 100]; color = :grey70, linestyle = :dash)
            for (i, model) in enumerate(MODELS)
                sub = series(calibration, target, model)
                lines!(ax, sub.level_pct, 100 .* sub[!, metric];
                    color = COLORS[i], linewidth = 2, label = model)
                scatter!(ax, sub.level_pct, 100 .* sub[!, metric];
                    color = COLORS[i], markersize = 7)
            end
            hidespines!(ax, :t, :r)
            k == length(TARGETS) && row == 1 && axislegend(ax; position = :lt, framevisible = false)
        end
    end
    save(joinpath(figdir, "ensemble_qcp_picp_deciles.png"), fig)

    fig = Figure(; size = (1100, 420), fontsize = 14)
    for (k, target) in enumerate(TARGETS)
        ax = Axis(fig[1, k];
            title = LABELS[k],
            xlabel = "PI level (%)",
            ylabel = "Mean PIW ($(UNITS[k]))",
            xticks = TICKS,
        )
        for (i, model) in enumerate(MODELS)
            sub = series(calibration, target, model)
            lines!(ax, sub.level_pct, sub.mean_piw; color = COLORS[i], linewidth = 2, label = model)
        end
        hidespines!(ax, :t, :r)
        k == length(TARGETS) && axislegend(ax; position = :lt, framevisible = false)
    end
    save(joinpath(figdir, "ensemble_sharpness_diagram.png"), fig)

    fig = Figure(; size = (1100, 420), fontsize = 14)
    for (k, target) in enumerate(TARGETS)
        ax = Axis(fig[1, k];
            title = LABELS[k],
            ylabel = "PIW ($(UNITS[k]))",
            xticks = (1:length(MODELS), collect(MODELS)),
            xticklabelrotation = π / 8,
        )
        ys = [Float64.(filter(r -> r.target == target && r.model == model, piw).piw) for model in MODELS]
        yhi = quantile(vcat(ys...), 0.99)
        for (i, y) in enumerate(ys)
            scatter!(ax, i .+ binned_swarm(y; ymax = yhi), y;
                color = (COLORS[i], 0.35), markersize = 2.5, strokewidth = 0)
            med = median(y)
            lines!(ax, [i - 0.28, i + 0.28], [med, med]; color = :black, linewidth = 1.6)
        end
        ylims!(ax, 0, yhi)
        hidespines!(ax, :t, :r)
    end
    save(joinpath(figdir, "ensemble_piw_beeswarm.png"), fig)
end
